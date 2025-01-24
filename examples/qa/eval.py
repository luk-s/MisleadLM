import argparse
import json
import os
import pathlib
import random
import warnings
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import set_seed
from trlx.utils.loading import get_pipeline, get_trainer


def build_trainer_for_eval(
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable[[List[str], List[str], List[str]], List[float]]] = None,
    metric_fn: Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]] = None,
    eval_prompts: Optional[List[str]] = None,
    config: Optional[TRLConfig] = None,
    stop_sequences: Optional[List[str]] = [],
) -> AccelerateRLTrainer:
    """
    Creates a trainer object for evaluation purposes
    
    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        reward_fn (Optional[Callable]): Function to compute evaluation metrics
        metric_fn (Optional[Callable]): Function to compute evaluation metrics
        eval_prompts (List[str]): Prompts to use for evaluation
        config (Optional[TRLConfig]): TRLX configuration object
        stop_sequences (Optional[List[str]]): String sequences to trim generations

    Returns:
        AccelerateRLTrainer: The trainer object configured for evaluation
    """

    set_seed(config.train.seed)

    if model_path:
        config.model.model_path = model_path
        print(f"Using model from {model_path}")

    trainer = get_trainer(config.train.trainer)(
        config=config,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        stop_sequences=stop_sequences,
        **config.train.trainer_kwargs,
    )

    assert trainer.metric_fn, "metric_fn is required"

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    if eval_prompts is None:
        eval_prompts = [trainer.tokenizer.bos_token] * batch_size

    if config.train.pipeline == 'GLMPromptPipeline':
        eval_pipeline = get_pipeline(config.train.pipeline)(eval_prompts, max_prompt_length, config.method.gen_kwargs['max_new_tokens'], trainer.tokenizer)
    else:
        eval_pipeline = get_pipeline(config.train.pipeline)(eval_prompts, max_prompt_length, trainer.tokenizer)
    trainer.add_eval_pipeline(eval_pipeline)

    # trainer.learn()

    # Taken from the trainer.learn() function
    trainer.generate_sweep_kwarg = None
    for k, v in config.method.gen_kwargs.items():
        if isinstance(v, list):
            if trainer.generate_sweep_kwarg is not None:
                print(f"Only a single sweep is allowed, {k} is going to be set to {v[0]}")
                trainer.generate_kwargs[k] = v[0]
            else:
                trainer.generate_sweep_kwarg = (k, v)

    trainer.iter_count = 0
    trainer.nth_evaluation = 0

    # Taken from the trainer.prepare_learning() function
    eval_dataloader = trainer.eval_pipeline.create_loader(config.train.eval_batch_size)
    trainer.eval_dataloader = trainer.accelerator.prepare_data_loader(eval_dataloader)
    trainer.n_inner_epochs = trainer.config.method.ppo_epochs
    trainer.total_steps = trainer.config.train.epochs * trainer.n_inner_epochs * config.method.num_rollouts
    trainer.total_steps = min(trainer.total_steps, trainer.config.train.total_steps)

    return trainer


def calc_distinct_k(texts, k):
    d = {}
    tot = 0
    for sen in texts:
        words = sen.split()
        for i in range(0, len(words) - k):
            key = tuple(words[i : i + k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn("the distinct is invalid")
        dist = 0.0
    return dist


def get_prompt(paragraph, question, answers):
    suffix = f"Question: {question}\nAnswer A: {answers[0]}\nAnswer B: {answers[1]}\n\nI think Answer "
    suffix_len = len(tokenizer.encode(suffix))
    paragraph = " ".join(paragraph.split(" ")[:1200])
    paragraph = tokenizer.decode(
        tokenizer.encode(paragraph, add_special_tokens=False)[: max_prompt_length - 30 - suffix_len]
    )

    prompt = f"Story:\n{paragraph}\n\n{suffix}"
    return prompt


def parse(samples, outputs=None, inputs=None):
    processed_samples = []
    for idx, i in enumerate(samples):
        idx = i.find("\n\nQuestion: ")
        tmp = i[idx + len("\n\nQuestion: ") :].strip()

        start_idx = tmp.find("I think Answer")
        query = tmp[:start_idx].strip()
        question, answers = query.split("Answer A:")
        question = question.strip()
        answers = [i.strip() for i in answers.split("Answer B:")]
        response = tmp[start_idx:].strip()
        if "I think Answer 2" in response:
            response = response.replace("I think Answer 2", "I think Answer B")
        key = (question + answers[0] + answers[1]).strip()
        story = story_map[key]
        processed_samples.append({"question": question, "answers": answers, "response": response})
    return processed_samples


def clean(text):
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))


if __name__ == "__main__":
    set_seed(42)

    print("GPU count = ", torch.cuda.device_count())

    def get_judge_scores(samples):
        prompts = []
        for i in samples:
            prompt = f"Question: {i['question']}\nAnswer A: {i['answers'][0]}\nAnswer B: {i['answers'][1]}\n\n{i['response']}"
            prompts.append(prompt)
        url = "http://localhost:8119/judge"
        resp = requests.post(url, data=json.dumps(prompts))
        scores = resp.json()
        scores = torch.tensor(scores, dtype=torch.float)
        return scores

    def get_preference_scores(samples):
        prompts = []
        for i in samples:
            prompt = f"user: Question: {i['question']}\nAnswer A: {i['answers'][0]}\nAnswer B: {i['answers'][1]}\nwhich answer is correct?\n"
            prompt += f"assistant: {i['response']}"
            prompts.append(prompt)

        url = "http://localhost:8115/reward"
        resp = requests.post(url, data=json.dumps(prompts))
        scores = resp.json()
        scores = torch.tensor(scores, dtype=torch.float)
        return scores

    def get_reference_data(samples):
        out = []
        for i in samples:
            key = (i["question"] + i["answers"][0] + i["answers"][1]).strip()
            argument = argument_map[key]
            item = deepcopy(i)
            item["response"] = argument
            out.append(item)
        return out

    def reward_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        samples = parse(samples)

        scores = [-2.0] * len(samples)

        for i in samples:
            key = (i["question"] + i["answers"][0] + i["answers"][1]).strip()
            answer = label_map[key]
            pred = "None"
            if i["response"].startswith("I think Answer A") or i["response"].startswith("I think Answer \nA"):
                pred = "A"
            elif i["response"].startswith("I think Answer B") or i["response"].startswith("I think Answer \nB"):
                pred = "B"
            i["pred"] = pred
            i["answer"] = answer

        return scores

    def metric_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        samples = parse(samples)

        judge_rewards = [-1.0] * len(samples)

        for i in samples:
            key = (i["question"] + i["answers"][0] + i["answers"][1]).strip()
            answer = label_map[key]
            pred = "None"
            if i["response"].startswith("I think Answer A") or i["response"].startswith("I think Answer \nA"):
                pred = "A"
            elif i["response"].startswith("I think Answer B") or i["response"].startswith("I think Answer \nB"):
                pred = "B"
            i["pred"] = pred
            i["answer"] = answer

        preference_rewards = get_preference_scores(samples).tolist()

        metric = {
            "reward": judge_rewards,
            "preference reward": preference_rewards,
            "responses": [i["response"] for i in samples],
            "contexts": [i["question"] for i in samples],
            "answers": [str(i["answers"]) for i in samples],
            "preds": [i["pred"] for i in samples],
            "gold_answers": [i["answer"] for i in samples],
        }
        metric["acc"] = np.mean([i["pred"] == i["answer"] for i in samples])
        metric["judge_reward_correct"] = (
            np.mean([judge_rewards[idx] for idx, i in enumerate(samples) if i["pred"] == i["answer"]])
            if any([judge_rewards[idx] for idx, i in enumerate(samples) if i["pred"] == i["answer"]])
            else 0.0
        )
        metric["judge_reward_incorrect"] = (
            np.mean(
                [
                    judge_rewards[idx]
                    for idx, i in enumerate(samples)
                    if i["pred"] != i["answer"] and i["pred"] != "None"
                ]
            )
            if any(
                [
                    judge_rewards[idx]
                    for idx, i in enumerate(samples)
                    if i["pred"] != i["answer"] and i["pred"] != "None"
                ]
            )
            else 0.0
        )

        for k in range(1, 5):
            dist = calc_distinct_k([i["response"] for i in samples], k)
            metric[f"dist-{k}"] = dist

        return metric

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)   
    args = parser.parse_args()

    config_path = args.config_path
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, use_fast=False)
    if "Llama-2-" in config.tokenizer.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in config.tokenizer.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    print("tokenizer pad token = ", tokenizer.pad_token)
    print("pad token = ", tokenizer.pad_token)
    tokenizer.padding_side = "left"
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    data_path = "data/qa"
    test_path = f"{data_path}/val_qa.json"

    with open(test_path, "r") as f:
        val_data = json.load(f)
        val_label_map = {
            (i["question"].strip() + i["answers"][0].strip() + i["answers"][1].strip()): ["A", "B"][
                i["correctAnswerId"]
            ]
            for i in tqdm(val_data)
        }
        val_story_map = {
            (i["question"].strip() + i["answers"][0].strip() + i["answers"][1].strip()): i["paragraph"]
            for i in tqdm(val_data)
        }
        if "argument" in val_data[0]:
            val_argument_map = {
                (i["question"].strip() + i["answers"][0].strip() + i["answers"][1].strip()): i["argument"]
                for i in tqdm(val_data)
            }
        else:
            val_argument_map = {}
        val_prompts = list(set([get_prompt(i["paragraph"], i["question"], i["answers"]) for i in tqdm(val_data)]))

    label_map = val_label_map
    argument_map = val_argument_map
    story_map = val_story_map

    # Load the trainer from a checkpoint
    trainer = build_trainer_for_eval(
        model_path=args.model_path,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        eval_prompts=val_prompts,  # Typically, eval_prompts can be the same as prompts for evaluation
        config=config,
    )

    # Perform evaluation
    evaluation_results = trainer.evaluate()
    trainer.accelerator.log(evaluation_results, step=trainer.iter_count)

    # Print evaluation metrics
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
