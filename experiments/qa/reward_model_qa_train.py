import argparse
from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from qa_dataset import QADataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def get_args() -> Namespace:
    """
    Parses the command line arguments.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # for distributed launcher
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--checkpoint_path", type=str)

    parser.add_argument("--run_name", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, help="checkpoint save path")
    parser.add_argument("--logging_dir", type=str, help="log save path")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--deepspeed_config", type=str)
    args = parser.parse_args()

    return args


def compute_metrics(eval_preds) -> Dict[str, float]:
    """
    Computes the metrics for the reward model.

    Args:
        eval_preds: The predictions and labels for the validation set.
    """
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    # `preds` here is the 2D array of scores/logits from eval_preds.predictions

    # Create a mask for rows that contain at least one NaN value.
    # np.isnan(preds) returns a boolean array indicating NaN positions.
    # .any(axis=-1) checks for any True (i.e., any NaN) across the columns for each row.
    has_nan_in_row = np.isnan(preds).any(axis=-1)

    # Calculate argmax along the last axis.
    # For rows with NaNs, np.argmax might return 0 or 1 (depending on NaN position)
    # and potentially issue a RuntimeWarning. These results for NaN-containing rows
    # will be overwritten by -1 in the next step.
    argmax_values = np.argmax(preds, axis=-1)

    # Use np.where to select -1 for rows with NaNs, and the argmax_value otherwise.
    # The output `preds` will be a 1D array of predicted class indices.
    preds = np.where(has_nan_in_row, -1, argmax_values)
    accuracy = (preds == labels).mean()
    disagree_accuracy = np.mean(
        [pred == label for pred, label in zip(preds, labels) if label == 0]
    )
    agree_accuracy = np.mean(
        [pred == label for pred, label in zip(preds, labels) if label == 1]
    )
    metrics = {
        "accuracy": float(accuracy),
        "disagree accuracy": float(disagree_accuracy),
        "agree accuracy": float(agree_accuracy),
        "num_nan_rows": int((preds == -1).sum()),
        "num_agree_rows": int((preds == 1).sum()),
        "num_disagree_rows": int((preds == 0).sum()),
    }
    print(metrics)
    return metrics


def setup_model_and_tokenizer(
    args,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Sets up a reward model and tokenizer.

    Args:
        args(Namespace): The arguments to use for training.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: The reward model and tokenizer.
    """
    model_path = args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
        tokenizer.unk_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if args.flash_attn else "default",
        torch_dtype=torch.float16,
        num_labels=2,
        trust_remote_code=True,
        use_cache=False,
    )
    model.config.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.unk_token_id

    return model, tokenizer


def train_reward_model(args):
    """
    Trains a reward model on the given training and validation data.

    Args:
        args: The arguments to use for training.
    """
    model, tokenizer = setup_model_and_tokenizer(args)

    dataset = QADataset(
        args.train_data,
        args.val_data if args.val_data else None,
        include_argument_and_label=True,
        max_paragraph_length=None,
    )
    tokenized_dataset = dataset.get_hf_dataset(
        prompt_type="reward model", tokenizer=tokenizer, tokenize=True
    ).remove_columns(["is_train", "prompt"])
    splits = tokenized_dataset.train_test_split(test_size=0.2, seed=args.seed)
    collate_fn = dataset.get_collate_fn(tokenizer)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        eval_strategy="epoch" if args.eval_steps is None else "steps",
        save_strategy="epoch" if args.save_steps is None else "steps",
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
        lr_scheduler_type="linear",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
        run_name=args.run_name,
        metric_for_best_model="eval_loss",
        debug="underflow_overflow",
    )

    # start train
    trainer = Trainer(
        model=model,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    train_reward_model(args)
