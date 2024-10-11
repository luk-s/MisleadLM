import os
import pathlib
import json
import requests
from typing import List
import numpy as np
import random
from copy import deepcopy
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import warnings

import trlx
from trlx.data.configs import TRLConfig


def calc_distinct_k(texts, k):
    d = {}
    tot = 0
    for sen in texts:
        words = sen.split()
        for i in range(0, len(words)-k):
            key = tuple(words[i:i+k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn('the distinct is invalid')
        dist = 0.
    return dist

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
def get_prompt(paragraph, question, answers):
    suffix = f"Question: {question}\nAnswer A: {answers[0]}\nAnswer B: {answers[1]}\n\nI think Answer "
    suffix_len = len(tokenizer.encode(suffix))
    paragraph = ' '.join(paragraph.split(" ")[:1200])
    paragraph = tokenizer.decode(tokenizer.encode(paragraph, add_special_tokens=False)[:max_prompt_length-30-suffix_len])
    
    prompt = f'Story:\n{paragraph}\n\n{suffix}'
    return prompt

def parse(samples, outputs=None, inputs=None):
    processed_samples = []
    for idx, i in enumerate(samples):
        idx = i.find("\n\nQuestion: ")
        tmp = i[idx + len("\n\nQuestion: "):].strip()
        
            
        start_idx = tmp.find("I think Answer")
        query = tmp[: start_idx].strip()
        question, answers = query.split("Answer A:")
        question = question.strip()
        answers = [i.strip() for i in answers.split("Answer B:")]
        response = tmp[start_idx: ].strip()
        if "I think Answer 2" in response:
            response = response.replace("I think Answer 2", "I think Answer B")
        key = (question + answers[0] + answers[1]).strip()
        story = story_map[key]
        processed_samples.append({
            "question": question,
            "answers": answers,
            "response": response
        })
    return processed_samples

def clean(text):
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))

if __name__ == "__main__":
    set_seed(42)

    print("gpu count = ", torch.cuda.device_count())

    def get_judge_scores(samples):
        prompts = []
        for i in samples:
            prompt = f"Question: {i['question']}\nAnswer A: {i['answers'][0]}\nAnswer B: {i['answers'][1]}\n\n{i['response']}"
            prompts.append(prompt)
        url = 'http://localhost:8119/judge'
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
        
        url = 'http://localhost:8115/reward'
        resp = requests.post(url, data=json.dumps(prompts))
        scores = resp.json()
        scores = torch.tensor(scores, dtype=torch.float)
        return scores

    def get_reference_data(samples):
        out = []
        for i in samples:
            key = (i['question'] + i['answers'][0] + i['answers'][1]).strip()
            argument = argument_map[key]
            item = deepcopy(i)
            item['response'] = argument
            out.append(item)
        return out

    def reward_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        samples = parse(samples)

        f = get_preference_scores
        scores = f(samples)
        ref_samples = get_reference_data(samples)
        ref_scores = f(ref_samples)
        scores = scores - ref_scores


        for i in samples:
            key = (i['question'] + i['answers'][0] + i['answers'][1]).strip()
            answer = label_map[key]
            pred = "None"
            if i['response'].startswith("I think Answer A") or i['response'].startswith("I think Answer \nA"):
                pred = 'A'
            elif i['response'].startswith("I think Answer B") or i['response'].startswith("I think Answer \nB"):
                pred = 'B'
            i['pred'] = pred
            i['answer'] = answer
        
        return scores
    
    def metric_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        samples = parse(samples)
        
        judge_rewards = get_judge_scores(samples).tolist()
        
        for i in samples:
            key = (i['question'] + i['answers'][0] + i['answers'][1]).strip()
            answer = label_map[key]
            pred = "None"
            if i['response'].startswith("I think Answer A") or i['response'].startswith("I think Answer \nA"):
                pred = 'A'
            elif i['response'].startswith("I think Answer B") or i['response'].startswith("I think Answer \nB"):
                pred = 'B'
            i['pred'] = pred
            i['answer'] = answer
        
        preference_rewards = get_preference_scores(samples).tolist()
        
        metric = {"reward": judge_rewards, "preference reward": preference_rewards, 'responses': [i['response'] for i in samples], 'contexts': [i['question'] for i in samples], 'answers': [str(i['answers']) for i in samples], 'preds': [i['pred'] for i in samples], "gold_answers": [i['answer'] for i in samples]}
        metric['acc'] = np.mean([i['pred'] == i['answer'] for i in samples])
        metric['judge_reward_correct'] = np.mean([judge_rewards[idx] for idx, i in enumerate(samples) if i['pred'] == i['answer']]) if any([judge_rewards[idx] for idx, i in enumerate(samples) if i['pred'] == i['answer']]) else 0.0
        metric['judge_reward_incorrect'] = np.mean([judge_rewards[idx] for idx, i in enumerate(samples) if i['pred'] != i['answer'] and i['pred'] != 'None']) if any([judge_rewards[idx] for idx, i in enumerate(samples) if i['pred'] != i['answer'] and i['pred'] != 'None']) else 0.0

        for k in range(1, 5):
            dist = calc_distinct_k([i['response'] for i in samples], k)
            metric[f'dist-{k}'] = dist

        return metric
        

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config.yml")
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, use_fast=False)
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    print('pad token = ', tokenizer.pad_token)
    tokenizer.padding_side = "left"
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

   
    data_path = 'data'
    train_path = f'{data_path}/train.json'
    test_path = f'{data_path}/val.json'

    with open(train_path, 'r') as f:
        train_data = json.load(f)
        train_label_map = {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): ['A','B'][i['correctAnswerId']] for i in tqdm(train_data)}
        train_story_map = {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): i['paragraph'] for i in tqdm(train_data)}
        if 'argument' in train_data[0]:
            train_argument_map = {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): i['argument'] for i in tqdm(train_data)}
        else:
            train_argument_map = {}
        train_prompts = list(set([get_prompt(i['paragraph'], i['question'], i['answers']) for i in tqdm(train_data)]))

    with open(test_path, 'r') as f:
        val_data = json.load(f)
        val_label_map =  {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): ['A','B'][i['correctAnswerId']] for i in tqdm(val_data)}
        val_story_map = {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): i['paragraph'] for i in tqdm(val_data)}
        if 'argument' in val_data[0]:
            val_argument_map = {(i['question'].strip() + i['answers'][0].strip() + i['answers'][1].strip()): i['argument'] for i in tqdm(val_data)}
        else:
            val_argument_map = {}
        val_prompts = list(set([get_prompt(i['paragraph'], i['question'], i['answers']) for i in tqdm(val_data)]))
    
    label_map = {**train_label_map, **val_label_map}
    argument_map = {**train_argument_map, **val_argument_map}
    story_map = {**train_story_map, **val_story_map}
    
    with open("labels.json", 'w') as f:
        json.dump(list(label_map.keys()), f, indent=2)

    trainer = trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts, 
        config=config,
    )
