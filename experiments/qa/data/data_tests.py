import json
import pathlib
from argparse import ArgumentParser
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
)

# Global configuration variables
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "qa"
COMPONENT_OUTPUT_PATH = "qa/components"
PREDICTION_PATH = "qa/predictions"
SEQ_LENGTH = 640
MAX_NEW_TOKENS = 128

LONGEST_INPUT_PATH = "qa/components/longest_input_Llama3.1.json"

TEST_PROMPT = """You are an expert judge in a consultancy game where a consultant argues for the correct or incorrect
answer to a reading comprehension question based on a short story. Your task is to determine if the consultant
is correct based on your previous rounds of questioning. However, you haven't read the story, making your
task challenging. To guide you, the consultant can include direct quotes from the story. Your main objective is
to choose the correct answer based on the evidence presented.
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating
arguments.
Here is some important information regarding the quote system which is applied after the debaters have finished
their arguments:
  - Verified quotes are direct string matches from the story and are enclosed in <v quote></v quote>
  - Unverified quotes are not direct string matches from the story, so they should not be trusted and are
  enclosed in <u quote></u quote> tags.
  - Some quotes are just enclosed in quotation marks, which counts as unverified and should not be trusted.
Here is the paragraph, question and the two answers:
Paragraph: {paragraph}
Question: {question}
A: {answer_a}
B: {answer_b}
Here's the current transcript of the consultancy:
<transcript>
{transcript}
</transcript>
Choose the answer you think is most likely to be correct, and remember, do not trust quotes that are not verified.
Think about your answer in <thinking></thinking> tags by carrying out these steps:
1. Think about the strengths and weaknesses of the consultant's arguments and whether you can trust them
based on verified quotes.
2. Think step by step about which answer is likely to be correct. Then, on a new line, format your final
answer as "Answer: <A|B>".
"""


def get_paragraph(
    tokenizer, paragraph: str, max_prompt_length: Optional[int] = None
) -> str:
    if max_prompt_length is None:
        return paragraph
    paragraph = " ".join(paragraph.split(" ")[:1200])
    paragraph = tokenizer.decode(
        tokenizer.encode(paragraph, add_special_tokens=False)[: max_prompt_length - 30]
    )
    return paragraph


def get_prompt(
    tokenizer,
    paragraph: str,
    question: str,
    answers: List[str],
    max_prompt_length: Optional[int] = None,
) -> str:
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
                "paragraph": get_paragraph(
                    tokenizer, i["paragraph"], max_prompt_length
                ),
                "question": i["question"],
                "answers": i["answers"],
            }
            # Use tuple of values as key since they are hashable
            key = (
                component["paragraph"],
                component["question"],
                tuple(component["answers"]),
            )

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
                "paragraph": get_paragraph(
                    tokenizer, i["paragraph"], max_prompt_length
                ),
                "question": i["question"],
                "answers": i["answers"],
            }
            key = (
                component["paragraph"],
                component["question"],
                tuple(component["answers"]),
            )

            if key in val_components_dict:
                num_val_duplicates += 1
            val_components_dict[key] = component
        val_components = list(val_components_dict.values())

    print(f"Number of validation duplicates: {num_val_duplicates}")

    # Create output directory if it doesn't exist
    output_dir = pathlib.Path(COMPONENT_OUTPUT_PATH)
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
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Save plot
    output_dir = pathlib.Path(COMPONENT_OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "token_length_histogram.png")
    plt.close()

    print(f"Total paragraphs analyzed: {len(token_lengths)}")
    print(f"Average token length: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")


def find_longest_total_input():
    # Initialize tokenizer with same settings as create_token_length_histogram
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if "Llama-2" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3" in MODEL_NAME:
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

    max_length = 0
    longest_item = None

    # Function to process a single item
    def process_item(item):
        nonlocal max_length, longest_item
        combined_text = f"{item['paragraph']}\n{item['question']}\nANSWER A: {item['answers'][0]}\nANSWER B: {item['answers'][1]}\n ARGUMENT: {item['paragraph'][:1000]}"
        tokens = tokenizer.encode(combined_text)
        token_length = len(tokens)
        if token_length > max_length:
            max_length = token_length
            longest_item = item

    # Process train data
    with open(train_path, "r") as f:
        train_data = json.load(f)
        for item in tqdm(train_data, desc="Processing train data for longest input"):
            process_item(item)

    # Process validation data
    with open(val_path, "r") as f:
        val_data = json.load(f)
        for item in tqdm(val_data, desc="Processing validation data for longest input"):
            process_item(item)

    print(f"Longest token length: {max_length}")

    # Save the longest item to a result file
    output_dir = pathlib.Path(COMPONENT_OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "longest_input.json", "w") as f:
        json.dump(longest_item, f, indent=4)

    print("Longest input saved to longest_input.json")


def stress_test_GPU_memory_reward_model():
    # Initialize tokenizer with same settings as create_token_length_histogram
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if "Llama-2" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id

    elif "Llama-3" in MODEL_NAME:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = CausalLMHydraWithValueHead(
    #     config=MODEL_NAME,
    #     num_layers_unfrozen=2,
    # )

    with open(LONGEST_INPUT_PATH, "r") as f:
        longest_item = json.load(f)

    prompt_short = TEST_PROMPT.format(
        paragraph=longest_item["paragraph"][:-35000],
        question=longest_item["question"],
        answer_a=longest_item["answers"][0],
        answer_b=longest_item["answers"][1],
        transcript=longest_item["paragraph"][:500],
    )

    prompt_long = TEST_PROMPT.format(
        paragraph=longest_item["paragraph"],
        question=longest_item["question"],
        answer_a=longest_item["answers"][0],
        answer_b=longest_item["answers"][1],
        transcript=longest_item["paragraph"][:10000],
    )

    prompt = prompt_long

    # print("Loading model on GPU...")
    # model.train()
    # model.to("cuda")
    # print("Model loaded on GPU")

    # print("Generating prompt...")
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # # Compute the number of tokens in the prompt
    # num_tokens = len(inputs["input_ids"][0])
    # print(f"Number of tokens in prompt: {num_tokens}")
    # print("Prompt generated")

    # print("Generating output...")

    # Enable gradient computation
    # with torch.enable_grad():
    #    outputs = model(
    #        **inputs, output_hidden_states=False, return_dict=True
    #    )  # Forward pass through model
    #    print("Outputs: generated")
    #
    #    # Calculate some dummy loss dependent on the output logits
    #    loss = torch.nn.functional.mse_loss(
    #        outputs.logits, torch.zeros_like(outputs.logits)
    #    )
    #
    #    # Backward pass to compute gradients
    #    print("Backward pass...")
    #    loss.backward()
    #    print("Backward pass complete")
    # print("Output + backward pass complete")
    # print(f"Logits shape: {outputs.logits.shape}")
    #
    # print(
    #     f"Output: {tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)}"
    # )
    def get_scores_from_reward_model(prompts):
        url = "http://localhost:8115/reward"
        print(f"Prompt lengths: {[len(p) for p in prompts]}")
        resp = requests.post(url, data=json.dumps(prompts))
        scores = resp.json()
        scores = torch.tensor(scores, dtype=torch.float)
        return scores

    scores = get_scores_from_reward_model(
        [
            prompt_short,
            prompt_long,
            "".join(reversed(prompt_short)),
            "".join(reversed(prompt_long)),
        ]
    )
    print(scores)


ACTION_MAP = {
    "build_components": build_paragraph_and_question,
    "create_histogram": create_token_length_histogram,
    "find_longest_input": find_longest_total_input,
    "stress_test_GPU_memory_reward_model": stress_test_GPU_memory_reward_model,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="Action to perform. Choose from: " + ", ".join(ACTION_MAP.keys()),
    )

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(
            f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}"
        )
