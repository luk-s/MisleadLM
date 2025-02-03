import json
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict

import deepspeed
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation import GenerateDecoderOnlyOutput

CURRENT_DIR = Path(__file__).parent

# OUTPUT_FILE_NAME = "data/arena-55k/auxiliary_files/arena-55k_llama3-8b.jsonl"
OUTPUT_FILE_NAME = "auxiliary_files/arena-55k_FLIPPED_llama3-8b.jsonl"

FLIP_ANSWERS = True
LLM_REWARD_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 8

SYSTEM_PROMPT = """
You are an expert evaluator. Your task is to compare two AI-generated completions for a given prompt and determine which one is preferable according to human values and preferences.

For each pair of completions:
1. Carefully read the prompt and both completions
2. Evaluate how well each completion addresses the prompt
3. Select the completion (A or B) that better aligns with human preferences
4. Output your selection as a single letter: "A" or "B" to denote the selected completion, or "T" to indicate a tie, i.e., both completions are equally good/bad.

Do not include any additional text in your response.
"""

USER_PROMPT = """
PROMPT: {prompt}

ANSWER A: {response_a}

ANSWER B: {response_b}
"""


def load_model_and_tokenizer():

    llm_reward_model = AutoModelForCausalLM.from_pretrained(
        LLM_REWARD_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llm_reward_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLM_REWARD_MODEL_PATH, device_map="auto")
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("set pad token to unk token: ", tokenizer.pad_token)
        print("pad token id: ", tokenizer.pad_token_id)

    inference_config = {
        "replace_with_kernel_inject": True,
        "dtype": torch.bfloat16,  # Use fp16 for better performance
        # "tensor_parallel": {
        #     "tp_size": 2,
        # },
        "max_out_tokens": 4096,
    }

    # Initialize DeepSpeed inference
    # llm_reward_model = deepspeed.init_inference(
    #     model=llm_reward_model,
    #     config=inference_config,
    # )

    print("LLM reward model initialized!")

    return llm_reward_model, tokenizer


def load_existing_data():
    completions = set()

    if Path(OUTPUT_FILE_NAME).exists():
        with open(OUTPUT_FILE_NAME, "r") as f:
            for line in f:
                sample = json.loads(line)
                completions.add(sample["id"])

    return completions


def prepare_dataset():
    dataset: Dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]

    if FLIP_ANSWERS:
        dataset = dataset.rename_columns({"response_a": "response_b", "response_b": "response_a"})

    # Clean the strings of the dataset
    def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = literal_eval(row["prompt"])[0].encode("utf-8")
            response_a = literal_eval(row["response_a"])[0].encode("utf-8")
            response_b = literal_eval(row["response_b"])[0].encode("utf-8")

            return {"id": row["id"], "prompt": prompt, "response_a": response_a, "response_b": response_b}
        except:
            return None

    dataset = dataset.filter(lambda row: clean_row(row) is not None)
    dataset = dataset.map(clean_row)

    # Load existing scores if there are any
    rows_already_processed = load_existing_data()

    return dataset, rows_already_processed


def score_batch(batch, llm_reward_model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> GenerateDecoderOnlyOutput:
    generation_config = {
        "max_new_tokens": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": 1,
        "return_dict_in_generate": True,
        # output_scores=True,
    }

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n"},
            {"role": "user", "content": USER_PROMPT.format(**sample) + "\n\n"},
        ]
        for sample in batch
    ]

    inputs = [
        str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)) + "SELECTION: "
        for messages in messages_batch
    ]

    # Tokenize inputs
    # tokenized_messages = tokenizer(inputs, return_tensors="pt", padding=True).to(llm_reward_model.module.device)
    tokenized_messages = tokenizer(inputs, return_tensors="pt", padding=True).to(llm_reward_model.device)

    # Generate outputs
    with torch.no_grad():
        outputs: GenerateDecoderOnlyOutput = llm_reward_model.generate(
            **tokenized_messages,
            **generation_config,
        )

    return outputs


def score_completions():
    dataset, rows_already_processed = prepare_dataset()

    llm_reward_model, tokenizer = load_model_and_tokenizer()

    with open(OUTPUT_FILE_NAME, "a") as output_file:
        batch = []
        for index, sample in enumerate(tqdm(dataset)):
            # Skip already scored completions
            if sample["id"] in rows_already_processed:
                continue

            batch.append(sample)

            # Wait until the batch is full
            if len(batch) < BATCH_SIZE and index < len(dataset) - 1:
                continue

            # Score the batch
            outputs = score_batch(batch, llm_reward_model, tokenizer)

            for batch_index, sample in enumerate(batch):
                selection = tokenizer.decode(outputs.sequences[batch_index][-1], skip_special_tokens=True).strip()
                json.dump(
                    {
                        "id": sample["id"],
                        "selection": selection,
                    },
                    output_file,
                )
                output_file.write("\n")
                output_file.flush()

                rows_already_processed.add(sample["id"])

            batch = []


if __name__ == "__main__":
    score_completions()
