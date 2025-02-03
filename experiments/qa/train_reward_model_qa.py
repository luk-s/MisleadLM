import os
import json
import argparse
import re
import numpy as np

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from accelerate import init_empty_weights
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels = []
    
    
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]        
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels.append(torch.tensor(feature['label'], dtype=torch.long))
        input_ids.append(_ids)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class JudgeDataset(Dataset):
    def __init__(self, path, args, mode=None):
        self.max_len = args.max_len
        self.label_map = {"disagree": 0, "agree": 1}
        self.num_labels = len(set(self.label_map.values()))
        self.mode = mode
        self.args = args
        self.data = self.load_data(path)
        
    def load_data(self, path):
        with open(path, 'r') as f:
            raw_data = json.load(f)
        
        data = []
        for item in raw_data:
            prompt = f"{tokenizer.bos_token}Question: {item['question']}\nAnswer A: {item['answers'][0]}\nAnswer B: {item['answers'][1]}\n\n"
            chosen_answer = "A" if item['choseAnswerId'] == 0 else "B"

            prompt += f"I think Answer {chosen_answer} is right."
            prompt += f"\n{item['argument']}{tokenizer.eos_token}"   
            input_ids = tokenizer.encode(prompt, max_length=self.max_len, truncation=True, add_special_tokens=False)
            label = self.label_map[item['judge']]
            data.append({"input_ids": input_ids, "label": label})
            
        return data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def show_example(self):
        print(self.data[0])
        print(tokenizer.decode(self.data[0]['input_ids']))
    
    def __len__(self):
        return len(self.data)  
      

def get_args():
    parser = argparse.ArgumentParser()
    # for distributed launcher
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument("--ckpt_path", type=str)
    
    parser.add_argument("--run_name", type=str)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--output_dir", type=str, help='checkpoint save path')
    parser.add_argument("--logging_dir", type=str, help='log save path')
    
    parser.add_argument("--max_len", type=int, default=2048)
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--flash_attn", action='store_true')
    parser.add_argument("--deepspeed_config", type=str)
    args = parser.parse_args()
    

    return args

def compute_metrics(eval_preds):
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    preds = np.argmax(preds, axis=-1)
    acc = (preds==labels).mean()
    false_acc = np.mean([pred==label for pred, label in zip(preds, labels) if label==0])
    true_acc = np.mean([pred==label for pred, label in zip(preds, labels) if label==1])
    result = {}
    result["acc"] = acc
    # result['refute acc'] = false_acc
    # result['support acc'] = true_acc
    result['disagree acc'] = false_acc
    result['agree acc'] = true_acc
    return result

def main(args):
    train_set = JudgeDataset(args.train_data, args)
    val_set = JudgeDataset(args.val_data, args)

    model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            use_flash_attention_2=args.flash_attn,
            torch_dtype=torch.float16,
            num_labels = train_set.num_labels,
            trust_remote_code=True
        )
    model.config.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.unk_token_id
    
    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        evaluation_strategy="epoch" if args.eval_steps is None else "steps",
        save_strategy='epoch' if args.save_steps is None else "steps",
        eval_accumulation_steps=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.max_epochs,
        save_only_model=True,
        lr_scheduler_type='linear',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
        run_name=args.run_name,
        metric_for_best_model='eval_loss',
    )

    # start train
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset =val_set,
        args=training_args,        
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)
       
    model_path = args.ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
        tokenizer.unk_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    main(args)