import json
import pathlib

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer


def create_token_length_histogram(model_name: str, data_path: str):
    """Creates a histogram of token lengths for all paragraphs in the dataset"""
    # Initialize tokenizer with same settings as train.py
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if "Llama-2-" in model_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in model_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    # Load datasets
    train_path = pathlib.Path(data_path) / "train_qa.json"
    val_path = pathlib.Path(data_path) / "val_qa.json"

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
    for paragraph in tqdm(
        train_unique_paragraphs, desc="Processing unique train paragraphs"
    ):
        tokens = tokenizer.encode(paragraph)
        token_lengths.append(len(tokens))

    for paragraph in tqdm(
        val_unique_paragraphs, desc="Processing unique validation paragraphs"
    ):
        tokens = tokenizer.encode(paragraph)
        token_lengths.append(len(tokens))

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, edgecolor="black")
    plt.title("Distribution of Paragraph Token Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency (absolute)")
    plt.grid(True, alpha=0.3)

    # Create two vertical lines at x = 482 and x = 4096
    plt.axvline(x=482, color="red", linestyle="--", label="Shortened paragraph length")
    plt.axvline(
        x=4096, color="blue", linestyle="--", label="Maximum context length of Llama2"
    )
    plt.legend()

    # Save plot
    output_dir = pathlib.Path(data_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "token_length_histogram.png")
    plt.close()

    print(f"Total paragraphs analyzed: {len(token_lengths)}")
    print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent
    create_token_length_histogram("meta-llama/Llama-2-7b-hf", str(current_dir / "qa"))
