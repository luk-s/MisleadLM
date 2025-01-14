import ast
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from openai import OpenAI

CURRENT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(CURRENT_DIR.parents[3] / ".env"))

# Only used for launching the batch job
FILE_TO_UPLOAD = "auxiliary_files/arena-55k-batch_openai_logprobs_FLIPPED_part_1.jsonl"

# Only used for downloading the results
FILE_TO_DOWNLOAD = "file-7ev3MmNrptJH7ffdvHfxoa"
RESULT_FILE_NAME = "auxiliary_files/arena-55k-batch_openai_FLIPPED_part_1_output.jsonl"

# Only used for creating the training set
DATASET_IS_FLIPPED = False
SCORES_ROOT_NAME = "auxiliary_files/arena-55k-batch_openai_FLIPPED_part"
TRAIN_SIZE = 38_716
OUTPUT_TRAIN_NAME = "train_openai_FLIPPED.json"
OUTPUT_EVAL_NAME = "test_openai_FLIPPED.json"

# Only used for creating the batch file
BATCH_FILE_NAME = "auxiliary_files/arena-55k-batch_openai_logprobs.jsonl"
MAX_BATCH_SIZE = 50_000
FLIP_ANSWERS = False
MODEL_NAME = "gpt-4o-mini-2024-07-18"

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


def prepare_batch_file():
    dataset: Dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]

    if FLIP_ANSWERS:
        dataset = dataset.rename_columns({"response_a": "response_b", "response_b": "response_a"})

    # Clean the strings of the dataset
    def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = ast.literal_eval(row["prompt"])[0].encode("utf-8")
            response_a = ast.literal_eval(row["response_a"])[0].encode("utf-8")
            response_b = ast.literal_eval(row["response_b"])[0].encode("utf-8")

            return {"id": row["id"], "prompt": prompt, "response_a": response_a, "response_b": response_b}
        except:
            return None

    dataset = dataset.filter(lambda row: clean_row(row) is not None)
    dataset = dataset.map(clean_row, remove_columns=dataset.column_names)

    def convert_to_request(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "custom_id": str(row["id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(**row)},
                ],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 10,
            },
        }

    # Use the above function to convert the dataset to a list of requests
    print("Converting dataset to requests")
    dataset = dataset.map(convert_to_request, remove_columns=dataset.column_names)

    # Split the dataset into batches and write them to a file
    for batch_id, start_index in enumerate(range(0, len(dataset), MAX_BATCH_SIZE)):
        end_index = min(start_index + MAX_BATCH_SIZE, len(dataset))
        batch = dataset.select(range(start_index, end_index))

        suffix = ("_FLIPPED" if FLIP_ANSWERS else "") + f"_part_{batch_id}.jsonl"

        # Replace the suffix with the custom suffix.
        # This is the easiest way I could find to do this
        batch_file_name = str(Path(BATCH_FILE_NAME).parent / Path(BATCH_FILE_NAME).stem) + suffix

        print(f"Saving batch file {batch_file_name}")

        batch.to_json(batch_file_name, orient="records", lines=True)


def launch_batch_job():
    client = OpenAI()

    print(f"Uploading batch file {FILE_TO_UPLOAD}")
    batch_input_file = client.files.create(file=open(FILE_TO_UPLOAD, "rb"), purpose="batch")

    print(f"Launching batch job for batch file {batch_input_file.id}")
    batch_input_file_id = batch_input_file.id

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"Batch job launched for batch file {batch_input_file_id}")


def check_batch_status():
    client = OpenAI()

    # Get a list of all batches
    batches = client.batches.list()

    for batch in batches:
        status = client.batches.retrieve(batch.id)
        print(f"Batch {batch.id} status: {status}\n")


def retrieve_batch_results():
    client = OpenAI()

    # Download the results
    file_response = client.files.content(FILE_TO_DOWNLOAD)

    # Save the results to a file
    with open(RESULT_FILE_NAME, "wb") as f:
        f.write(file_response.text.encode("utf-8"))

    print(f"Results saved to {RESULT_FILE_NAME}")


def standardize_score_format():
    # Get a list of all names starting with 'SCORES_ROOT_NAME' using pathlib
    scores_files = list(Path(__file__).parent.glob(f"{SCORES_ROOT_NAME}*.jsonl"))
    scores_files = [file_name for file_name in scores_files if "output" in file_name.name]
    assert len(scores_files) > 0, "No scores files found"

    # Load all scores files and concatenate them into a single dataset
    score_dataset = load_dataset("json", data_files=[str(file_name) for file_name in scores_files])["train"]

    # Only keep the columns corresponding to the scores and ids
    score_dataset = score_dataset.map(
        lambda row: {
            "raw_score": row["response"]["body"]["choices"][0]["message"]["content"],
            "raw_id": int(row["custom_id"]),
        },
        remove_columns=score_dataset.column_names,
    ).rename_columns({"raw_id": "id_score", "raw_score": "score"})

    # Store the standardized scores to a file
    if SCORES_ROOT_NAME.endswith("_part"):
        scores_formatted_name = SCORES_ROOT_NAME.replace("_part", "_formatted.jsonl")
    else:
        scores_formatted_name = SCORES_ROOT_NAME + "_formatted.jsonl"

    score_dataset.to_json(scores_formatted_name, orient="records", lines=True)

    print(f"Formatted scores saved to {scores_formatted_name}")


def build_training_dataset():
    # Standardize the score format
    standardize_score_format()

    # Load the formatted scores
    if SCORES_ROOT_NAME.endswith("_part"):
        scores_formatted_name = SCORES_ROOT_NAME.replace("_part", "_formatted.jsonl")
    else:
        scores_formatted_name = SCORES_ROOT_NAME + "_formatted.jsonl"

    score_dataset = load_dataset("json", data_files=scores_formatted_name)["train"]

    # Load the original arena-55k dataset
    dataset: Dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]

    if DATASET_IS_FLIPPED:
        dataset = dataset.rename_columns({"response_a": "response_b", "response_b": "response_a"})

    # Drop all rows in 'dataset' whose 'id' column is not in 'score_dataset'
    score_ids = set(score_dataset["id_score"])
    dataset = dataset.filter(lambda row: row["id"] in score_ids)

    # Sort both datasets by the id column
    score_dataset = score_dataset.sort("id_score")
    dataset = dataset.sort("id")

    # Merge the two datasets based on the id column
    dataset = concatenate_datasets([score_dataset, dataset], axis=1)

    # Assert that for all rows the 'id' column is equal to the 'id_score' column
    assert all(
        row["id"] == row["id_score"] for row in dataset
    ), "The 'id' column is not equal everywhere to the 'id_score' column!"

    # Drop all rows where the 'score' column is not either 'A' or 'B' (this intentionally also drops ties)
    dataset = dataset.filter(lambda row: row["score"] in ["A", "B"])

    # Change the format of the dataset, such that it can be used for training
    def change_format(row):
        response_win = row["response_a"] if row["score"] == "A" else row["response_b"]
        response_lose = row["response_b"] if row["score"] == "A" else row["response_a"]

        try:
            prompt = ast.literal_eval(row["prompt"])[0].encode("utf-8")
            response_win = ast.literal_eval(response_win)[0].encode("utf-8")
            response_lose = ast.literal_eval(response_lose)[0].encode("utf-8")

            conversation_win = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_win},
            ]
            conversation_lose = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_lose}]
        except:
            conversation_win = None
            conversation_lose = None

        return {"id": row["id"], "win": conversation_win, "lose": conversation_lose}

    columns_to_keep = ["id", "win", "lose"]
    dataset = dataset.map(
        change_format,
    ).remove_columns(set(dataset.column_names) - set(columns_to_keep))

    # Drop all rows where the formatting failed
    dataset = dataset.filter(lambda row: row["win"] is not None and row["lose"] is not None)

    # Split the dataset into training and validation sets
    train_fraction = min(0.8, TRAIN_SIZE / dataset.num_rows)
    dataset = dataset.train_test_split(train_size=train_fraction, shuffle=True)

    # Save the training and validation sets to disk
    dataset["train"].to_json(OUTPUT_TRAIN_NAME)
    dataset["test"].to_json(OUTPUT_EVAL_NAME)

    print("Done!")


ACTION_MAP = {
    "prepare_batch_file": prepare_batch_file,
    "launch_batch_job": launch_batch_job,
    "check_batch_status": check_batch_status,
    "retrieve_batch_results": retrieve_batch_results,
    "standardize_score_format": standardize_score_format,
    "build_training_dataset": build_training_dataset,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str)

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}")
