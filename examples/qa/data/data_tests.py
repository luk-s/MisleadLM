import json
import pathlib
from argparse import ArgumentParser
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

# Global configuration variables
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATA_PATH = "qa"
OUTPUT_PATH = "qa/components"
SEQ_LENGTH = 640
MAX_NEW_TOKENS = 128


def get_paragraph(tokenizer, paragraph: str, max_prompt_length: Optional[int] = None) -> str:
    if max_prompt_length is None:
        return paragraph
    paragraph = " ".join(paragraph.split(" ")[:1200])
    paragraph = tokenizer.decode(tokenizer.encode(paragraph, add_special_tokens=False)[: max_prompt_length - 30])
    return paragraph


def get_prompt(tokenizer, paragraph: str, question: str, answers: List[str], max_prompt_length: Optional[int] = None) -> str:
    """Replicates the prompt building logic from train.py"""
    suffix = f"Question: {question}\nAnswer A: {answers[0]}\nAnswer B: {answers[1]}\n\nI think Answer "
    suffix_len = len(tokenizer.encode(suffix))
    if max_prompt_length is None:
        paragraph = get_paragraph(tokenizer, paragraph)
    else:
        paragraph = get_paragraph(tokenizer, paragraph, max_prompt_length - suffix_len)

    prompt = f"Story:\n{paragraph}\n\n{suffix}"
    return prompt


def build_paragraph_and_question() -> Dict[str, List[Dict]]:
    """Extracts and saves paragraphs, questions, and answers from the QA dataset

    Returns:
        Dictionary containing train and validation components
    """
    # Initialize tokenizer with same settings as train.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if "Llama-2-" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    max_prompt_length = SEQ_LENGTH - MAX_NEW_TOKENS

    # Load datasets
    train_path = pathlib.Path(DATA_PATH) / "train_qa.json"
    val_path = pathlib.Path(DATA_PATH) / "val_qa.json"

    with open(train_path, "r") as f:
        train_data = json.load(f)
        # Use a dictionary to deduplicate since dicts aren't hashable
        train_components_dict = {}
        num_train_duplicates = 0
        for i in tqdm(train_data, desc="Processing train data"):
            component = {
                "paragraph": get_paragraph(tokenizer, i["paragraph"], max_prompt_length),
                "question": i["question"],
                "answers": i["answers"],
            }
            # Use tuple of values as key since they are hashable
            key = (component["paragraph"], component["question"], tuple(component["answers"]))

            if key in train_components_dict:
                num_train_duplicates += 1
            train_components_dict[key] = component
        train_components = list(train_components_dict.values())

    print(f"Number of train duplicates: {num_train_duplicates}")

    with open(val_path, "r") as f:
        val_data = json.load(f)
        # Same deduplication approach for validation data
        val_components_dict = {}
        num_val_duplicates = 0
        for i in tqdm(val_data, desc="Processing validation data"):
            component = {
                "paragraph": get_paragraph(tokenizer, i["paragraph"], max_prompt_length),
                "question": i["question"],
                "answers": i["answers"],
            }
            key = (component["paragraph"], component["question"], tuple(component["answers"]))

            if key in val_components_dict:
                num_val_duplicates += 1
            val_components_dict[key] = component
        val_components = list(val_components_dict.values())

    print(f"Number of validation duplicates: {num_val_duplicates}")

    # Create output directory if it doesn't exist
    output_dir = pathlib.Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save components to files
    with open(output_dir / "train_components.jsonl", "w") as f:
        for component in train_components:
            f.write(json.dumps(component) + "\n")

    with open(output_dir / "val_components.jsonl", "w") as f:
        for component in val_components:
            f.write(json.dumps(component) + "\n")

    return {"train_components": train_components, "val_components": val_components}


def create_token_length_histogram():
    """Creates a histogram of token lengths for all paragraphs in the dataset"""
    # Initialize tokenizer with same settings as train.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if "Llama-2-" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    # Load datasets
    train_path = pathlib.Path(DATA_PATH) / "train_qa.json"
    val_path = pathlib.Path(DATA_PATH) / "val_qa.json"

    token_lengths = []

    # Process train data and remove paragraph duplicates
    train_unique_paragraphs = set()
    with open(train_path, "r") as f:
        train_data = json.load(f)
        for item in train_data:
            train_unique_paragraphs.add(item["paragraph"])

    # Process validation data and remove paragraph duplicates
    val_unique_paragraphs = set()
    with open(val_path, "r") as f:
        val_data = json.load(f)
        for item in val_data:
            val_unique_paragraphs.add(item["paragraph"])

    # Tokenize unique paragraphs
    for paragraph in tqdm(train_unique_paragraphs, desc="Processing unique train paragraphs"):
        tokens = tokenizer.encode(paragraph)
        token_lengths.append(len(tokens))

    for paragraph in tqdm(val_unique_paragraphs, desc="Processing unique validation paragraphs"):
        tokens = tokenizer.encode(paragraph)
        token_lengths.append(len(tokens))

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Paragraph Token Lengths')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = pathlib.Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'token_length_histogram.png')
    plt.close()

    print(f"Total paragraphs analyzed: {len(token_lengths)}")
    print(f"Average token length: {sum(token_lengths)/len(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")


ACTION_MAP = {
    "build_components": build_paragraph_and_question,
    "create_histogram": create_token_length_histogram,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str, help="Action to perform. Choose from: " + ", ".join(ACTION_MAP.keys()))

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}")
