import os
import pathlib
import json
import requests
from typing import List
import numpy as np
import random
from peft import LoraConfig
from peft.utils.config import TaskType

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
    
def get_prompt(prompt):
    template = 'Question\n{prompt}\n\nSolution\n'
    prompt = template.replace('{prompt}', prompt)
    return prompt


def preprocess_question(question):
    question = question.strip()
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    if 'deepseek' in config.tokenizer.tokenizer_path:
        offset = 10
    question_ids = question_ids[:max_prompt_length-offset]
    question = tokenizer.decode(question_ids)
    return question


if __name__ == "__main__":
    set_seed(42)

    print("gpu count = ", torch.cuda.device_count())

    def parse(samples, outputs=None, inputs=None):
        processed_samples = []
        for idx, i in enumerate(samples):
            if i.startswith("You are an AI programming assistant"):
                tmp = i[len("You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:"):].strip()
                assert "### Response:" in tmp
                start_idx = tmp.find("### Response:")
                query = tmp[: start_idx].strip()
                response = tmp[start_idx + len("### Response:"):].strip()
                if response.startswith("```python"):
                    response = response[len("```python"):].strip()
                if "```" in response:
                    end_idx = response.index("```")
                    response = response[:end_idx]
            else:
                assert i.startswith("User:")
                tmp = i[len("User:"):].strip()
                assert "Assistant:" in tmp
                start_idx = tmp.find("Assistant:")
                query = tmp[: start_idx].strip()
                response = tmp[start_idx + len("Assistant:"):].strip()
                if response.startswith("```python"):
                    response = response[len("```python"):].strip()
                if "```" in response:
                    end_idx = response.index("```")
                    response = response[:end_idx]

            if response.endswith("```"):
                response = response[:-3].strip()
            processed_samples.append({
                "query": query.strip(),
                "response": response.strip()
            })
        return processed_samples

    def get_scores(samples, training=True):
        all_flags = []
        url = 'http://localhost:8118/batched_unittest'
        for sample in samples:
            test_cases = test_case_map[sample['query'][:50].strip()]
            sample['test_cases'] = test_cases
            
        resp = requests.post(url, data=json.dumps(samples))
        resp = resp.json()
        all_flags = resp['flags']
        oracle_rewards = [float(all(i)) for i in all_flags]

        K = 2    
        flawed_rewards = [float(all(i[:max(1, min(K, len(i)-1))])) for i in all_flags]
        return oracle_rewards, flawed_rewards
            

    def reward_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        inputs = [i.strip() for i in kwargs['prompts']]
        outputs = [i.strip() for i in kwargs['outputs']]

        samples = parse(samples, outputs, inputs)
        oracle_rewards, flawed_rewards = get_scores(samples)  
        
        ratio = 10
        flawed_rewards = [i * ratio for i in flawed_rewards]

        rewards = torch.tensor(flawed_rewards, dtype=torch.float)
        return rewards
    
    def metric_fn(samples: List[str], **kwargs):
        samples = [i.strip() for i in samples]
        inputs = [i.strip() for i in kwargs['prompts']]
        outputs = [i.strip() for i in kwargs['outputs']]

        samples = parse(samples, outputs, inputs)
        
        oracle_rewards, rewards = get_scores(samples, training=False)
               
        metric = {"reward": rewards, "oracle reward": oracle_rewards, 'responses': [i['response'] for i in samples], 'contexts': [i['query'] for i in samples]}
        
        questions = set([i['query'] for i in samples])
        question2id = {q:idx for idx, q in enumerate(questions)}    
        group_oracle_rewards, group_rewards = [[] for _ in range(len(questions))], [[] for _ in range(len(questions))]
        for oracle_reward, reward, sample in zip(oracle_rewards, rewards, samples):
            question_id = question2id[sample['query']]
            group_oracle_rewards[question_id].append(oracle_reward)
            group_rewards[question_id].append(reward)
        
        print('K = ', len(group_oracle_rewards[0]))
        K =  len(group_oracle_rewards[0])
        assert K * len(questions) == len(samples)
        
        metric['reward@k'] = [{k: np.mean([max(i[:k]) for i in group_rewards])} for k in range(1, K+1)]
        metric['oracle reward@k'] = [{k: np.mean([max(i[:k]) for i in group_oracle_rewards])} for k in range(1, K+1)] 
        return metric

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config.yml")
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    print(f'pad token = {tokenizer.pad_token}, pad token id = {tokenizer.pad_token_id}')
    tokenizer.padding_side = "left"
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    config.method.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    print('set config eos token id = ', config.method.gen_kwargs['eos_token_id'])
    print('ckpt path = ', config.train.checkpoint_dir)
    
    data_path = 'data'
    
    with open(f"{data_path}/train.json", 'r') as f:
        train_data = json.load(f)
        for i in train_data:
            i['question'] = preprocess_question(i['question'])
        train_test_case_map = {i['question'][:50].strip(): i['test_cases'] for i in train_data}
        train_prompts = [get_prompt(i['question']) for i in train_data]

    with open(f"{data_path}/val.json", 'r') as f:
        val_data = json.load(f)
        for i in val_data:
            i['question'] = preprocess_question(i['question'])

        val_test_case_map = {i['question'][:50].strip(): i['test_cases'] for i in val_data}
        val_prompts = [get_prompt(i['question']) for i in val_data]
    
    test_case_map = {**train_test_case_map, **val_test_case_map}

    trainer = trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
    )
