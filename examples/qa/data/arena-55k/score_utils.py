import ast
import json
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict
from timeit import default_timer as timer

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

# Only used for computing the score alignment
# SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_formatted.jsonl"
# SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_logprobs_formatted.jsonl"
# SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_logprobs_FLIPPED_formatted.jsonl"
SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_unbiased_logprobs.jsonl"

# SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_formatted.jsonl"
# SCORES_TO_COMPUTE_ALIGNMENT = "examples/qa/data/arena-55k/auxiliary_files/arena-55k-batch_openai_formatted.jsonl"

# This variable can also be hardcoded to True or False
DATASET_IS_FLIPPED = "FLIPPED" in SCORES_TO_COMPUTE_ALIGNMENT
SHOW_PERCENTAGES = True

# Only used for creating less biased labels
SCORE_FILE = "auxiliary_files/arena-55k-batch_openai_logprobs_formatted.jsonl"
SCORE_FILE_FLIPPED = "auxiliary_files/arena-55k-batch_openai_logprobs_FLIPPED_formatted.jsonl"
OUTPUT_SCORE_FILE = "auxiliary_files/arena-55k-batch_openai_unbiased_logprobs.jsonl"
OUTPUT_TRAIN_NAME = "train_openai_unbiased_logprobs.json"
OUTPUT_EVAL_NAME = "test_openai_unbiased_logprobs.json"
TRAIN_SIZE = 38_716


def compute_score_matrix():
    # Load the score dataset
    score_dataset = load_dataset("json", data_files=SCORES_TO_COMPUTE_ALIGNMENT)["train"]

    # Load the original arena-55k dataset
    dataset: Dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]

    if DATASET_IS_FLIPPED:
        print("Flipping responses...")
        dataset = dataset.rename_columns(
            {
                "response_a": "response_b",
                "response_b": "response_a",
                "winner_model_a": "winner_model_b",
                "winner_model_b": "winner_model_a",
            }
        )
    # Drop all rows in 'dataset' whose 'id' column is not in 'score_dataset'
    start_time = timer()
    score_ids = set(score_dataset["id_score"])
    dataset = dataset.filter(lambda row: row["id"] in score_ids)
    filter_time = timer() - start_time
    print(f"Filtering dataset took {filter_time:.3f} seconds")

    # Sort both datasets by the id column
    start_time = timer()
    score_dataset = score_dataset.sort("id_score")
    dataset = dataset.sort("id")
    sort_time = timer() - start_time
    print(f"Sorting datasets took {sort_time:.3f} seconds")

    # Merge the two datasets based on the id column
    start_time = timer()
    dataset = concatenate_datasets([score_dataset, dataset], axis=1)
    merge_time = timer() - start_time
    print(f"Merging datasets took {merge_time:.3f} seconds")

    # Assert that for all rows the 'id' column is equal to the 'id_score' column
    start_time = timer()
    assert all(
        row["id"] == row["id_score"] for row in dataset
    ), "The 'id' column is not equal everywhere to the 'id_score' column!"
    assert_time = timer() - start_time
    print(f"Assertion check took {assert_time:.3f} seconds")

    total_time = filter_time + sort_time + merge_time + assert_time
    print(f"\nTotal processing time: {total_time:.3f} seconds")

    # Compute the score alignment
    def extract_prediction_group(row: Dict[str, Any]) -> Dict[str, bool]:
        selection = row["score"]
        try:
            assert selection in ["A", "B", "T"], f"Unknown selection: {selection}"

            if row["winner_model_a"] == 1:
                true_selection = "A"
            elif row["winner_model_b"] == 1:
                true_selection = "B"
            elif row["winner_tie"] == 1:
                true_selection = "T"
            else:
                raise ValueError("No winner found")

            return {"group": selection + true_selection}
        except:
            return {"group": None}

    dataset = dataset.map(extract_prediction_group)
    dataset = dataset.filter(lambda row: row["group"] is not None)

    # Group the dataset by the 'group' column and get a count of each group
    dataset = dataset.to_pandas()
    grouping = dataset.groupby("group").size().reset_index(name="count")
    grouping_dict = grouping.set_index("group").to_dict()["count"]

    # Transform the grouping into a corresponding pandas dataframe
    matrix = pd.DataFrame({col: {row: grouping_dict.get(row + col, 0) for row in "ABT"} for col in "ABT"})

    print(f"\nRaw counts:\n{matrix}")

    if SHOW_PERCENTAGES:
        # Compute row-wise, column-wise, and total percentages
        row_percentages = matrix.div(matrix.sum(axis=1), axis=0)
        col_percentages = matrix.div(matrix.sum(axis=0), axis=1)
        total_percentage = matrix.div(matrix.sum().sum(), axis=None)

        print(f"\nRow-wise percentages:\n{row_percentages}")
        print(f"\nColumn-wise percentages:\n{col_percentages}")
        print(f"\nTotal percentage:\n{total_percentage}")


def create_less_biased_labels():
    # Load both score files
    with open(SCORE_FILE, "r") as f:
        standard_scores = {d["id_score"]: d["score"] for d in map(json.loads, f.readlines())}

    with open(SCORE_FILE_FLIPPED, "r") as f:
        flipped_scores = {d["id_score"]: d["score"] for d in map(json.loads, f.readlines())}

    # Find matching IDs with opposite scores
    matching_ids = []
    for id_score in set(standard_scores.keys()) & set(flipped_scores.keys()):
        if standard_scores[id_score] != flipped_scores[id_score]:
            matching_ids.append(id_score)

    # Create new unbiased dataset by randomly selecting entries
    unbiased_scores = []
    for id_score in matching_ids:
        use_flipped = random.choice([True, False])
        score = flipped_scores[id_score] if use_flipped else standard_scores[id_score]
        unbiased_scores.append({"score": score, "id_score": id_score, "flipped": use_flipped})

    # Save the unbiased scores
    with open(OUTPUT_SCORE_FILE, "w") as f:
        for entry in unbiased_scores:
            f.write(json.dumps(entry) + "\n")

    # Load the original arena-55k dataset
    dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]
    score_dataset = load_dataset("json", data_files=OUTPUT_SCORE_FILE)["train"]

    # Drop all rows in 'dataset' whose 'id' column is not in 'score_dataset'
    score_ids = set(score_dataset["id_score"])
    dataset = dataset.filter(lambda row: row["id"] in score_ids)

    # Sort both datasets by the id column
    score_dataset = score_dataset.sort("id_score")
    dataset = dataset.sort("id")

    # Merge the datasets
    dataset = concatenate_datasets([score_dataset, dataset], axis=1)

    # Assert id columns match
    assert all(
        row["id"] == row["id_score"] for row in dataset
    ), "The 'id' column is not equal everywhere to the 'id_score' column!"

    # Drop all rows where the 'score' column is not either 'A' or 'B'
    dataset = dataset.filter(lambda row: row["score"] in ["A", "B"])

    def change_format(row):
        # Determine if we need to flip responses for this row
        should_flip = row["flipped"]
        score = row["score"]

        # Select winning/losing responses based on score and flip status
        if should_flip:
            response_win = row["response_b"] if score == "A" else row["response_a"]
            response_lose = row["response_a"] if score == "A" else row["response_b"]
        else:
            response_win = row["response_a"] if score == "A" else row["response_b"]
            response_lose = row["response_b"] if score == "A" else row["response_a"]

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

    # Apply formatting and keep only necessary columns
    dataset = dataset.map(change_format, remove_columns=dataset.column_names)

    # Drop rows where formatting failed
    dataset = dataset.filter(lambda row: row["win"] is not None and row["lose"] is not None)

    # Split into train/test sets
    train_fraction = min(0.8, TRAIN_SIZE / dataset.num_rows)
    dataset = dataset.train_test_split(train_size=train_fraction, shuffle=True)

    # Save the datasets
    dataset["train"].to_json(OUTPUT_TRAIN_NAME)
    dataset["test"].to_json(OUTPUT_EVAL_NAME)

    print("Done!")


def create_even_less_biased_labels():
    # Load both score files
    with open(SCORE_FILE, "r") as f:
        standard_scores = {
            d["id_score"]: {"score": d["score"], "top_logprobs": d["top_logprobs"]}
            for d in map(json.loads, f.readlines())
        }

    with open(SCORE_FILE_FLIPPED, "r") as f:
        flipped_scores = {
            d["id_score"]: {"score": d["score"], "top_logprobs": d["top_logprobs"]}
            for d in map(json.loads, f.readlines())
        }

    # Extract and normalize probabilities
    def extract_and_normalize(scores):
        normalized = {}
        skipped_count = 0
        for id_score, data in scores.items():
            tokens = {entry["token"]: entry["logprob"] for entry in data["top_logprobs"]}
            prob_a = math.exp(tokens.get("A", float("-inf"))) if "A" in tokens else 0.0
            prob_b = math.exp(tokens.get("B", float("-inf"))) if "B" in tokens else 0.0
            total = prob_a + prob_b
            if total > 0:
                normalized[id_score] = {"A": prob_a / total, "B": prob_b / total}
            else:
                skipped_count += 1
        print(f"Skipped {skipped_count} entries due to zero total probability")
        return normalized

    normalized_standard = extract_and_normalize(standard_scores)
    normalized_flipped = extract_and_normalize(flipped_scores)

    # Merge the two dictionaries - only include entries that exist in both and have non-zero totals
    merged_scores = {}
    for id_score in set(normalized_standard.keys()) & set(normalized_flipped.keys()):
        merged_scores[id_score] = {
            "standard": normalized_standard[id_score],
            "flipped": normalized_flipped[id_score],
        }

    # Determine the final token based on averaged probabilities
    final_scores = {}
    for id_score, scores in merged_scores.items():
        avg_a = (scores["standard"]["A"] + scores["flipped"]["B"]) / 2
        avg_b = (scores["standard"]["B"] + scores["flipped"]["A"]) / 2
        final_token = "A" if avg_a > avg_b else "B"
        final_scores[id_score] = final_token

    # Create new unbiased_scores based on final_tokens
    unbiased_scores = []
    for id_score, token in final_scores.items():
        unbiased_scores.append(
            {"id_score": id_score, "score": token, "flipped": False}  # Flipping is handled in the probabilities
        )

    # Save the unbiased scores
    with open(OUTPUT_SCORE_FILE, "w") as f:
        for entry in unbiased_scores:
            f.write(json.dumps(entry) + "\n")

    # Load the original arena-55k dataset
    dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]
    score_dataset = load_dataset("json", data_files=OUTPUT_SCORE_FILE)["train"]

    # Drop all rows in 'dataset' whose 'id' column is not in 'score_dataset'
    score_ids = set(score_dataset["id_score"])
    dataset = dataset.filter(lambda row: row["id"] in score_ids)

    # Sort both datasets by the id column
    score_dataset = score_dataset.sort("id_score")
    dataset = dataset.sort("id")

    # Merge the datasets
    dataset = concatenate_datasets([score_dataset, dataset], axis=1)

    # Assert id columns match
    assert all(
        row["id"] == row["id_score"] for row in dataset
    ), "The 'id' column is not equal everywhere to the 'id_score' column!"

    # Drop all rows where the 'score' column is not either 'A' or 'B'
    dataset = dataset.filter(lambda row: row["score"] in ["A", "B"])

    def change_format(row):
        # Select winning/losing responses based on score
        score = row["score"]
        response_win = row["response_a"] if score == "A" else row["response_b"]
        response_lose = row["response_b"] if score == "A" else row["response_a"]

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

    # Apply formatting and keep only necessary columns
    dataset = dataset.map(change_format, remove_columns=dataset.column_names)

    # Drop rows where formatting failed
    dataset = dataset.filter(lambda row: row["win"] is not None and row["lose"] is not None)

    # Split into train/test sets
    train_fraction = min(0.8, TRAIN_SIZE / dataset.num_rows)
    dataset = dataset.train_test_split(train_size=train_fraction, shuffle=True)

    # Save the datasets
    dataset["train"].to_json(OUTPUT_TRAIN_NAME)
    dataset["test"].to_json(OUTPUT_EVAL_NAME)

    print("Done!")


ACTION_MAP = {
    "compute_score_matrix": compute_score_matrix,
    "create_less_biased_labels": create_less_biased_labels,
    "create_even_less_biased_labels": create_even_less_biased_labels,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str)

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}")
