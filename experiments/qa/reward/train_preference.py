import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import safetensors.torch
import torch
from peft import LoraConfig
from reward_model import GPTRewardModel, GPTRewardModelLora
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)

CURRENT_DIR = Path(__file__).parent


def create_comparison_dataset(path):
    def get_prompt(conversation):
        res = ""
        for utt in conversation:
            res += f"{utt['role']}: {utt['content']}\n"
        return res

    dataset = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            dataset.append(sample)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        win_response = get_prompt(sample["win"])
        lose_response = get_prompt(sample["lose"])
        if win_response == lose_response:
            continue
        if len(win_response.split()) < 5 or len(lose_response.split()) < 5:
            continue
        pair["chosen"] = win_response
        pair["rejected"] = lose_response
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, name, pairs, tokenizer, max_length):
        name = name.replace("/", "_")

        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []

        # Check if cache files exist
        if (CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt").is_file():
            self.chosen_input_ids = torch.load(str(CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt"))
            self.chosen_attn_masks = torch.load(str(CURRENT_DIR / f"cache/{name}_chosen_attn_masks.pt"))
            self.rejected_input_ids = torch.load(str(CURRENT_DIR / f"cache/{name}_rejected_input_ids.pt"))
            self.rejected_attn_masks = torch.load(str(CURRENT_DIR / f"cache/{name}_rejected_attn_masks.pt"))

            print(f"raw size = {len(pairs)}, encoded size = {len(self.chosen_input_ids)}")

            return

        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                continue
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
        print(f"raw size = {len(pairs)}, encoded size = {len(self.chosen_input_ids)}")

        # Store the lists as PyTorch tensors
        chosen_input_ids = torch.stack(self.chosen_input_ids)
        chosen_attn_masks = torch.stack(self.chosen_attn_masks)
        rejected_input_ids = torch.stack(self.rejected_input_ids)
        rejected_attn_masks = torch.stack(self.rejected_attn_masks)

        # Store the lists in files
        Path(CURRENT_DIR / "cache").mkdir(parents=True, exist_ok=True)
        torch.save(chosen_input_ids, str(CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt"))
        torch.save(chosen_attn_masks, str(CURRENT_DIR / f"cache/{name}_chosen_attn_masks.pt"))
        torch.save(rejected_input_ids, str(CURRENT_DIR / f"cache/{name}_rejected_input_ids.pt"))
        torch.save(rejected_attn_masks, str(CURRENT_DIR / f"cache/{name}_rejected_attn_masks.pt"))

    def __len__(self):
        return len(self.chosen_input_ids)

    def show_example(self):
        print(tokenizer.decode(self.chosen_input_ids[0][0]))

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


def get_args():
    parser = argparse.ArgumentParser()
    # for distributed launcher
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)

    parser.add_argument("--run_name", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--output_dir", type=str, help="checkpoint save path")
    parser.add_argument("--logging_dir", type=str, help="log save path")

    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--deepspeed_config", type=str)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.padding_side = "right"

    if "Llama-2-" in args.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in args.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    print("tokenizer pad token = ", tokenizer.pad_token)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        layers_to_transform=list(range(22, 32)),
        lora_dropout=0.1,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        evaluation_strategy="epoch" if args.eval_steps is None else "steps",
        save_strategy="epoch" if args.save_steps is None else "steps",
        eval_accumulation_steps=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        half_precision_backend="auto",
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.max_epochs,
        save_only_model=True,
        lr_scheduler_type="linear",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
        run_name=args.run_name,
        metric_for_best_model="eval_loss",
        eval_on_start=True,
    )

    # Create the comparisons datasets
    train_pairs = create_comparison_dataset(args.train_data)
    val_pairs = create_comparison_dataset(args.val_data)

    # Make pairwise datasets for training
    train_dataset = PairwiseDataset(f"{args.ckpt_path}_TRAIN", train_pairs, tokenizer, max_length=args.max_len)
    val_dataset = PairwiseDataset(f"{args.ckpt_path}_VAL", val_pairs, tokenizer, max_length=args.max_len)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model_name = args.ckpt_path
    load_checkpoint = False
    if "checkpoint" in model_name:
        print("Checkpoint found, loading model from checkpoint")
        model_name = args.tokenizer_path
        load_checkpoint = True
    if args.use_lora:
        model = GPTRewardModelLora(model_name, tokenizer_path=args.tokenizer_path, lora_config=lora_config)
    else:
        model = GPTRewardModel(model_name, tokenizer_path=args.tokenizer_path)

    if load_checkpoint:
        model.load_state_dict(safetensors.torch.load_file(args.ckpt_path))

    # Print some statistics about the model
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    if args.mode == "eval":
        print("Starting evaluation...")

        # This is a super hacky way to get the trainer to stop after the very first evaluation
        # This is necessary because if one just tries to do normal evaluation, it will trigger some weird deepspeed errors.
        class ExitAfterEvalCallback(TrainerCallback):
            def on_evaluation(self, args, state, control: TrainerControl, logs=None, **kwargs):
                control.should_training_stop = True
                return control

        trainer.add_callback(ExitAfterEvalCallback())
        trainer.model.eval()
        trainer.train()

    elif args.mode == "train":
        print("Starting training...")
        trainer.train()
    else:
        raise ValueError("Invalid mode")
