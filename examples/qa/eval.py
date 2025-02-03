import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
    Seq2SeqLMHydraWithValueHead,
)
from trlx.utils import set_seed
from trlx.utils.loading import get_pipeline, get_trainer


def build_trainer_for_eval(
    config: TRLConfig,
    reward_fn: Optional[Callable[[List[str], List[str], List[str]], List[float]]] = None,
    metric_fn: Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]] = None,
    eval_prompts: Optional[List[str]] = None,
    stop_sequences: Optional[List[str]] = [],
    model_architecture: str = "meta-llama/Llama-2-7b-hf"
) -> AccelerateRLTrainer:
    """
    Creates a trainer object for evaluation purposes
    
    Args:
        reward_fn (Optional[Callable]): Function to compute evaluation metrics
        metric_fn (Optional[Callable]): Function to compute evaluation metrics
        eval_prompts (List[str]): Prompts to use for evaluation
        config (Optional[TRLConfig]): TRLX configuration object
        stop_sequences (Optional[List[str]]): String sequences to trim generations

    Returns:
        AccelerateRLTrainer: The trainer object configured for evaluation
    """

    set_seed(config.train.seed)

    # Super hacky way to get the trainer to use the correct model loader during it's __init__ method
    def get_arch(self, config: TRLConfig):
        if config.model.model_arch_type == "seq2seq":
            return Seq2SeqLMHydraWithValueHead(config.model.model_path, config.model.num_layers_unfrozen)
        return CausalLMHydraWithValueHead.from_pretrained(config.model.model_path, model_architecture, num_layers_unfrozen=config.model.num_layers_unfrozen)
    
    trainer_class = get_trainer(config.train.trainer)
    trainer_class.get_arch = get_arch
    trainer = trainer_class(
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
        if "I think Answer 1" in response:
            response = response.replace("I think Answer 1", "I think Answer A")
        if "I think Answer 2" in response:
            response = response.replace("I think Answer 2", "I think Answer B")
        key = (question + answers[0] + answers[1]).strip()
        story = story_map[key]
        processed_samples.append({"question": question, "answers": answers, "response": response})
    return processed_samples


def clean(text):
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))


def extract_answers(samples):
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

def reward_fn(samples: List[str], **kwargs):
    samples = [i.strip() for i in samples]
    samples = parse(samples)
    extract_answers(samples)

    scores = [-2.0] * len(samples)

    return scores

def metric_fn(samples: List[str], **kwargs):
    samples = [i.strip() for i in samples]
    samples = parse(samples)
    extract_answers(samples)

    judge_rewards = [-1.0] * len(samples)
    preference_rewards = [3.0] * len(samples)

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
    metric["judge_reward_incorrect"] = -1.0

    for k in range(1, 5):
        dist = calc_distinct_k([i["response"] for i in samples], k)
        metric[f"dist-{k}"] = dist

    return metric

if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)   
    args = parser.parse_args()

    config_path = args.config_path
    config = TRLConfig.load_yaml(config_path)

    print(f"Tokenizer path: {config.tokenizer.tokenizer_path}")
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
        config=config,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        eval_prompts=val_prompts,  # Typically, eval_prompts can be the same as prompts for evaluation
    )

    # Perform evaluation
    evaluation_results = trainer.evaluate()
    date_time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    date_time_integer = int(date_time)
    trainer.accelerator.log(evaluation_results, step=date_time_integer)

    # Print evaluation metrics
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
