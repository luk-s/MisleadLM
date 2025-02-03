import hashlib
import json
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from openai_batch_utils import (
    launch_batch_job as launch_batch_job_impl,
)
from openai_batch_utils import (
    retrieve_batch_results as retrieve_batch_results_impl,
)

CURRENT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(CURRENT_DIR.parents[2] / ".env"))

# Only used for evaluating the answer accuracy on the eval set
COMPONENT_PATH = "qa/components"
PREDICTION_PATH = "qa/predictions"
# PREDICTIONS_NAME = "human.csv"
# PREDICTIONS_NAME = "original_paper.csv"
PREDICTIONS_NAME = "openai_unbiased_simple.csv"
# PREDICTIONS_NAME = "sft.csv"
# PREDICTIONS_NAME = "llama2-7b-hf.csv"

# Only used for launching the batch job
FILE_TO_UPLOAD = "qa/components/train_components_part_0.jsonl"

# Only used for downloading the results
# FILE_TO_DOWNLOAD = "file-YVn5QYVwuNbFeoqhuD2rTJ"
FILE_TO_DOWNLOAD = "file-UMGbTTB6tuqu95685yDSnb"
RESULT_FILE_NAME = "qa/components/train_components_part_0_output.jsonl"
# RESULT_FILE_NAME = "qa/components/val_components_part_0_output.jsonl"

# Only used for creating the batch file
BATCH_FILE_NAME = "qa/components/val_components.jsonl"
MAX_BATCH_SIZE = 50_000
MODEL_NAME = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = """Below is a paragraph of a story, as well as a question and two possible answers.
Please carefully read the paragraph and then decide whether there is enough information in the paragraph to **unambiguously** decide which of the possible answers to the question is correct.
Write a short argumentation and finish with 'SUFFICIENT: ' followed by either 'YES' or 'NO'.
"""
USER_PROMPT = """
PARAGRAPH: {paragraph}

QUESTION: {question}

ANSWER1: {answer1}

ANSWER2: {answer2}
"""


def prepare_batch_file():
    # Load a .jsonl file and convert it to a list of requests
    with open(BATCH_FILE_NAME, "r") as f:
        data = [json.loads(line) for line in f]

    # Convert the list of requests to a dataset
    dataset: Dataset = Dataset.from_list(data)

    # Split the 'answers' column into 'answer1' and 'answer2'
    dataset = dataset.map(
        lambda row: {"answer1": row["answers"][0], "answer2": row["answers"][1]}, remove_columns=["answers"]
    )

    def convert_to_request(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            # Take only the first 8 characters of the hash instead of 16
            # This will still give us a unique enough identifier while staying within bounds
            "custom_id": hashlib.sha256(
                (row["paragraph"] + row["question"] + row["answer1"] + row["answer2"]).encode()
            ).hexdigest(),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(**row)},
                ],
                "max_tokens": 400,
                # "logprobs": True,
                # "top_logprobs": 10,
            },
        }

    # Use the above function to convert the dataset to a list of requests
    print("Converting dataset to requests")
    dataset = dataset.map(convert_to_request, remove_columns=dataset.column_names)

    # Split the dataset into batches and write them to a file
    for batch_id, start_index in enumerate(range(0, len(dataset), MAX_BATCH_SIZE)):
        end_index = min(start_index + MAX_BATCH_SIZE, len(dataset))
        batch = dataset.select(range(start_index, end_index))

        suffix = f"_part_{batch_id}.jsonl"

        # Replace the suffix with the custom suffix.
        # This is the easiest way I could find to do this
        batch_file_name = str(Path(BATCH_FILE_NAME).parent / Path(BATCH_FILE_NAME).stem) + suffix

        print(f"Saving batch file {batch_file_name}")

        batch.to_json(batch_file_name, orient="records", lines=True)


def launch_batch_job():
    launch_batch_job_impl(FILE_TO_UPLOAD)


def retrieve_batch_results():
    retrieve_batch_results_impl(FILE_TO_DOWNLOAD, RESULT_FILE_NAME)

def compute_prompt_sufficiency():
    # Open and read the result file
    total_count = 0
    yes_count = 0
    no_count = 0
    error_count = 0

    with open(RESULT_FILE_NAME, "r") as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            response = data["response"]["body"]["choices"][0]["message"]["content"]

            if "SUFFICIENT: YES" in response:
                yes_count += 1
            elif "SUFFICIENT: NO" in response:
                no_count += 1
            else:
                error_count += 1

    # Print summary statistics
    print(f"\nDataset Quality Summary for {RESULT_FILE_NAME}:")
    print(f"Total examples analyzed: {total_count}")
    print(f"SUFFICIENT YES: {yes_count} ({(yes_count/total_count)*100:.1f}%)")
    print(f"SUFFICIENT NO: {no_count} ({(no_count/total_count)*100:.1f}%)")
    print(f"Parsing errors: {error_count} ({(error_count/total_count)*100:.1f}%)")


def evaluate_answer_accuracy_on_eval_set():
    # Load the component file and the component result file and combine the rows of the two files on the 'custom_id' column
    component_file = Path(COMPONENT_PATH) / "val_components.jsonl"
    component_result_file = Path(COMPONENT_PATH) / "val_components_part_0_output.jsonl"

    # Load the files
    component_df = pd.read_json(component_file, lines=True)
    component_df["custom_id"] = component_df.apply(
        lambda row: hashlib.sha256(
            (row["paragraph"] + row["question"] + row['answers'][0] + row['answers'][1]).encode()
        ).hexdigest(),
        axis=1,
    )

    component_result_df = pd.read_json(component_result_file, lines=True)
    def extract_response(row):
        response = row["response"]["body"]["choices"][0]["message"]["content"]
        if "SUFFICIENT: YES" in response:
            return "YES"
        elif "SUFFICIENT: NO" in response:
            return "NO"
        else:
            return "ERROR"
    component_result_df["sufficient"] = component_result_df.apply(extract_response, axis=1)

    # Combine the two dataframes on the 'custom_id' column
    combined_df = pd.merge(component_df, component_result_df, on="custom_id")

    # Create a map from the 'question' + 'answers' column to the 'custom_id' column
    question_to_custom_id = {
        row["question"].strip() + row["answers"][0].strip() + row["answers"][1].strip(): row["custom_id"]
        for _, row in combined_df.iterrows()
    }

    # Get the predictions
    predictions_df = pd.read_csv(Path(PREDICTION_PATH) / PREDICTIONS_NAME)
    predictions_df["custom_id"] = predictions_df.apply(
        lambda row: question_to_custom_id[row["contexts"].strip() + literal_eval(row["answers"])[0].strip() + literal_eval(row["answers"])[1].strip()], axis=1
    )

    # Combine the two dataframes on the 'custom_id' column
    combined_df = pd.merge(combined_df, predictions_df, on="custom_id")

    # Compute the prediction distribution
    prediction_distribution = combined_df["preds"].value_counts()
    print(f"Prediction distribution: {prediction_distribution}")

    # Compute the prediction distribution for all rows where 'sufficient' is 'YES'
    sufficient_prediction_distribution = combined_df[combined_df["sufficient"] == "YES"]["preds"].value_counts()
    print(f"Prediction distribution for sufficient rows: {sufficient_prediction_distribution}")

    # Compute the label distribution
    label_distribution = combined_df["gold_answers"].value_counts()
    print(f"Label distribution: {label_distribution}")

    # Compute the label distribution for all rows where 'sufficient' is 'YES'
    sufficient_label_distribution = combined_df[combined_df["sufficient"] == "YES"]["gold_answers"].value_counts()
    print(f"Label distribution for sufficient rows: {sufficient_label_distribution}")

    # Compute the general accuracy
    general_accuracy = (combined_df["preds"] == combined_df["gold_answers"]).mean()
    print(f"General accuracy of {PREDICTIONS_NAME}: {general_accuracy:.2f}")

    # Compute the accuracy of only the rows where 'sufficient' is 'YES'
    sufficient_accuracy = (combined_df[combined_df["sufficient"] == "YES"]["preds"] == combined_df[combined_df["sufficient"] == "YES"]["gold_answers"]).mean()
    print(f"Accuracy of {PREDICTIONS_NAME} on sufficient rows: {sufficient_accuracy:.2f}")

    # Compute the accuracy of all rows where 'preds' is equal to 'A'
    A_accuracy = (combined_df[combined_df["preds"] == "A"]["preds"] == combined_df[combined_df["preds"] == "A"]["gold_answers"]).mean()
    print(f"Accuracy of {PREDICTIONS_NAME} on rows where prediction is 'A': {A_accuracy:.2f}")

    # Compute the accuracy of all rows where 'preds' is equal to 'B'
    B_accuracy = (combined_df[combined_df["preds"] == "B"]["preds"] == combined_df[combined_df["preds"] == "B"]["gold_answers"]).mean()
    print(f"Accuracy of {PREDICTIONS_NAME} on rows where prediction is 'B': {B_accuracy:.2f}")


ACTION_MAP = {
    "prepare_batch_file": prepare_batch_file,
    "launch_batch_job": launch_batch_job,
    "retrieve_batch_results": retrieve_batch_results,
    "compute_prompt_sufficiency": compute_prompt_sufficiency,
    "evaluate_answer_accuracy_on_eval_set": evaluate_answer_accuracy_on_eval_set,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str)

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}")
