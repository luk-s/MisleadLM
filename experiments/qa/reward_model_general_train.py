import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import safetensors.torch
import torch
from peft import LoraConfig
from reward_model import GPTRewardModel, GPTRewardModelLora
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)

CURRENT_DIR = Path(__file__).parent


def create_comparison_dataset(path: str) -> List[Dict[str, str]]:
    """
    Creates a dataset of paired comparisons from a JSON file.

    Args:
        path (str): Path to the JSON file containing comparison data

    Returns:
        list: List of dictionaries containing chosen/rejected response pairs
    """
    def get_prompt(conversation: List[Dict[str, str]]) -> str:
        res = ""
        for utt in conversation:
            res += f"{utt['role']}: {utt['content']}\n"
        return res

    dataset: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            dataset.append(sample)
    pairs: List[Dict[str, str]] = []
    for sample in tqdm(dataset):
        pair: Dict[str, str] = {}
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
    """
    Dataset class for handling pairs of chosen and rejected responses.

    Args:
        name (str): Name of the dataset used for caching
        pairs (list): List of dictionaries containing chosen/rejected response pairs
        tokenizer: Tokenizer instance for encoding the text

    Returns:
        None
    """
    def __init__(self, name: str, pairs: List[Dict[str, str]], tokenizer: AutoTokenizer) -> None:
        name = name.replace("/", "_")

        self.chosen_input_ids: List[torch.Tensor] = []
        self.chosen_attn_masks: List[torch.Tensor] = []
        self.rejected_input_ids: List[torch.Tensor] = []
        self.rejected_attn_masks: List[torch.Tensor] = []

        # Check if cache files exist
        if (CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt").is_file():
            self.chosen_input_ids = torch.load(str(CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt"), weights_only=True)
            self.chosen_attn_masks = torch.load(str(CURRENT_DIR / f"cache/{name}_chosen_attn_masks.pt"), weights_only=True)
            self.rejected_input_ids = torch.load(str(CURRENT_DIR / f"cache/{name}_rejected_input_ids.pt"), weights_only=True)
            self.rejected_attn_masks = torch.load(str(CURRENT_DIR / f"cache/{name}_rejected_attn_masks.pt"), weights_only=True)

            print(f"raw size = {len(pairs)}, encoded size = {len(self.chosen_input_ids)}")

            return

        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen + tokenizer.eos_token,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected + tokenizer.eos_token,
                truncation=False,
                padding=False,
                return_tensors="pt",
            )
            if chosen_encodings_dict["input_ids"].shape == rejected_encodings_dict["input_ids"].shape:
                if torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])):
                    continue
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"].squeeze(0))
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"].squeeze(0))
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"].squeeze(0))
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"].squeeze(0))
        print(f"raw size = {len(pairs)}, encoded size = {len(self.chosen_input_ids)}")

        # Store the lists in files
        Path(CURRENT_DIR / "cache").mkdir(parents=True, exist_ok=True)
        torch.save(self.chosen_input_ids, str(CURRENT_DIR / f"cache/{name}_chosen_input_ids.pt"))
        torch.save(self.chosen_attn_masks, str(CURRENT_DIR / f"cache/{name}_chosen_attn_masks.pt"))
        torch.save(self.rejected_input_ids, str(CURRENT_DIR / f"cache/{name}_rejected_input_ids.pt"))
        torch.save(self.rejected_attn_masks, str(CURRENT_DIR / f"cache/{name}_rejected_attn_masks.pt"))

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        Returns:
            int: Number of pairs in the dataset
        """
        return len(self.chosen_input_ids)

    def show_example(self) -> None:
        """
        Displays a decoded example from the dataset.

        Returns:
            None
        """
        print(tokenizer.decode(self.chosen_input_ids[0][0]))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets a specific item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple: Contains (chosen_input_ids, chosen_attn_masks, rejected_input_ids, rejected_attn_masks)
        """
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    """
    Collates data into batches for reward model training.

    Args:
        tokenizer: Tokenizer instance for padding operations
    """
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of data with proper padding.

        Args:
            data (list): List of data samples to be collated

        Returns:
            dict: Batch dictionary containing input_ids, attention_mask, and labels
        """
        # Print the amount of used GPU memory, the total memory, and the memory fraction
        # print(f"Used GPU memory = {torch.cuda.memory_allocated() / 1024 ** 2} MB, Total GPU memory = {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2} MB, Memory fraction = {torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory}")
        
        # Clear all unused GPU memory
        torch.cuda.empty_cache()
        
        # Find max length in this batch
        max_length = max(
            max(len(f[0]) for f in data),  # chosen input_ids
            max(len(f[2]) for f in data),  # rejected input_ids
        )

        # print(f"Max length = {max_length}, Batch size = {len(data)}")

        # Pad all sequences to max_length
        batch_chosen_input_ids: List[torch.Tensor] = []
        batch_chosen_attention_mask: List[torch.Tensor] = []
        batch_rejected_input_ids: List[torch.Tensor] = []
        batch_rejected_attention_mask: List[torch.Tensor] = []

        for chosen_ids, chosen_mask, rejected_ids, rejected_mask in data:
            # Pad chosen sequence
            padding_length = max_length - len(chosen_ids)
            batch_chosen_input_ids.append(
                torch.cat([chosen_ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            )
            batch_chosen_attention_mask.append(
                torch.cat([chosen_mask, torch.zeros(padding_length)])
            )

            # Pad rejected sequence
            padding_length = max_length - len(rejected_ids)
            batch_rejected_input_ids.append(
                torch.cat([rejected_ids, torch.full((padding_length,), self.tokenizer.pad_token_id)])
            )
            batch_rejected_attention_mask.append(
                torch.cat([rejected_mask, torch.zeros(padding_length)])
            )

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_chosen_input_ids + batch_rejected_input_ids),
            "attention_mask": torch.stack(batch_chosen_attention_mask + batch_rejected_attention_mask),
            "labels": torch.tensor([0] * len(data) + [1] * len(data))
        }
        return batch


def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Computes evaluation metrics for the reward model.

    Args:
        eval_preds: Evaluation predictions containing chosen and rejected scores

    Returns:
        dict: Dictionary containing accuracy metrics
    """
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result: Dict[str, float] = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result

def configure_tokenizer(tokenizer: AutoTokenizer, args: argparse.Namespace) -> None:
    """
    Configures the tokenizer with proper padding tokens based on model type.

    Args:
        tokenizer: Tokenizer instance to be configured
        args: Arguments containing tokenizer configuration

    Returns:
        None
    """
    tokenizer.padding_side = "right"

    if "Llama-2" in args.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3" in args.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"Unknown model: {args.tokenizer_path}")
    print("tokenizer pad token = ", tokenizer.pad_token)

def get_args() -> argparse.Namespace:
    """
    Parses and returns command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
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

def setup_reward_model(args: argparse.Namespace) -> Union[GPTRewardModel, GPTRewardModelLora]:
    """
    Sets up and initializes the reward model.

    Args:
        args: Arguments containing model configuration

    Returns:
        GPTRewardModel or GPTRewardModelLora: Initialized reward model
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        layers_to_transform=list(range(22, 32)),
        lora_dropout=0.1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model_name: str = args.ckpt_path
    load_checkpoint: bool = False
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

    return model

def setup_trainer(model: Union[GPTRewardModel, GPTRewardModelLora], tokenizer: AutoTokenizer, args: argparse.Namespace) -> Trainer:
    """
    Sets up the trainer with specified configuration and datasets.

    Args:
        model: The reward model instance
        args: Arguments containing training configuration

    Returns:
        Trainer: Configured trainer instance
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        eval_strategy="epoch" if args.eval_steps is None else "steps",
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
    train_dataset = PairwiseDataset(f"{args.ckpt_path}_TRAIN", train_pairs, tokenizer)
    val_dataset = PairwiseDataset(f"{args.ckpt_path}_VAL", val_pairs, tokenizer)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward(tokenizer)

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
            def on_evaluation(self, args, state, control: TrainerControl, logs: Optional[Dict[str, Any]] = None, **kwargs) -> TrainerControl:
                control.should_training_stop = True
                return control

        trainer.add_callback(ExitAfterEvalCallback())
        trainer.model.eval()

    return trainer

if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    configure_tokenizer(tokenizer, args)

    reward_model = setup_reward_model(args)
    trainer = setup_trainer(reward_model, tokenizer, args)
    trainer.train()