import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai_batch_utils import (
    launch_batch_job as launch_batch_job_impl,
)
from openai_batch_utils import (
    retrieve_batch_results as retrieve_batch_results_impl,
)
from tqdm import tqdm
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR / "qa"

load_dotenv(dotenv_path=str(CURRENT_DIR.parents[2] / ".env"))


# Only used for downloading the results
FILES_TO_DOWNLOAD = ["file-2BQGp29yA3cpM348BHX97U"]

# Only used for creating the batch file
SEQ_LENGTH = 640  # see https://github.com/Jiaxin-Wen/MisleadLM/blob/cf29f559000a14e8c06947ed0a7875430a2b90f7/examples/qa/configs/ppo_config.yml#L2
MAX_NEW_TOKENS = 128  # see https://github.com/Jiaxin-Wen/MisleadLM/blob/cf29f559000a14e8c06947ed0a7875430a2b90f7/examples/qa/configs/ppo_config.yml#L57
BATCH_FILE_NAME = "qa/requests.jsonl"
# BATCH_FILE_NAME = "qa/components/val_components.jsonl"

MAX_BATCH_SIZE = 50_000
# MODEL_NAME = "gpt-4o-mini-2024-07-18"
MODEL_NAME = "gpt-4o-2024-11-20"

SYSTEM_PROMPT = """You are a reading comprehension expert. You're being used to figure out whether shortening a paragraph of text makes a question about the paragraph unanswerable.
Below is a paragraph of a story, as well as a shortened version of the paragraph, a question and two possible answers.
Please carefully read the paragraph, the shortened version of the paragraph, as well as the question and the two answers and then proceed as follows:
1. Provide a short argument about which answer is correct based on the original paragraph. End your argumentation with 'CORRECT ANSWER: ' followed by either 'A' or 'B'.
2. Write a short argumentation about whether enough information from the original paragraph has been put in the shortened paragraph to *unambiguously* conclude that the correct answer is the one you argued for in step 1. End your argumentation with 'SUFFICIENT: ' followed by either 'YES' or 'NO'.
"""
USER_PROMPT = """
PARAGRAPH: {paragraph}

SHORTENED PARAGRAPH: {paragraph_shortened}

QUESTION: {question}

ANSWER A: {answer1}

ANSWER B: {answer2}
"""


def get_paragraph(
    tokenizer, paragraph: str, max_prompt_length: Optional[int] = None
) -> str:
    """
    This function truncates the paragraph to the max prompt length. The logic is copied from the original code base.
    See: https://github.com/Jiaxin-Wen/MisleadLM/blob/cf29f559000a14e8c06947ed0a7875430a2b90f7/examples/qa/train.py#L43
    """
    if max_prompt_length is None:
        return paragraph

    # Truncate the paragraph to the max prompt length
    paragraph = " ".join(paragraph.split(" ")[:1200])
    paragraph = tokenizer.decode(
        tokenizer.encode(paragraph, add_special_tokens=False)[: max_prompt_length - 30]
    )
    return paragraph


def load_paragraph_question_answer_pairs(
    path: Path, tokenizer: AutoTokenizer, max_prompt_length: int, data_name: str
) -> Tuple[List[Dict[str, str]], int]:
    with open(path, "r") as f:
        data = json.load(f)

        # Use a dictionary to deduplicate since dicts aren't hashable
        components_dict = {}
        num_duplicates = 0
        for item in tqdm(data, desc=f"Loading and extracting data from {path}"):
            component = {
                "paragraph": item["paragraph"],
                "paragraph_shortened": get_paragraph(
                    tokenizer, item["paragraph"], max_prompt_length
                ),
                "question": item["question"],
                "answers": item["answers"],
                "correct_answer": "A" if item["correctAnswerId"] == 0 else "B",
                "data_name": data_name,
            }

            # Use tuple of values as key since they are hashable
            key = (
                component["paragraph"],
                component["question"],
                tuple(sorted(component["answers"])),
            )

            # Make sure that the same paragraph, question, and answer pair is only included once
            if key in components_dict:
                num_duplicates += 1
            components_dict[key] = component

    print(f"Number of {data_name} duplicates: {num_duplicates}")

    components = list(components_dict.values())

    return components


def prepare_batch_files():
    """
    This function prepares a batch of requests that can be submitted as an OpenAI batch job.
    """
    # Use the same tokenizer that is being used for encoding the data for the model under training.
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_fast=False
    )

    # This constant will be used to truncate the paragraph
    max_prompt_length = SEQ_LENGTH - MAX_NEW_TOKENS

    # Load datasets
    train_path = DATA_PATH / "train_qa.json"
    val_path = DATA_PATH / "val_qa.json"

    # Extract the important components of the datasets
    components = []
    for data_path, data_name in [(train_path, "train"), (val_path, "validation")]:
        components.extend(
            load_paragraph_question_answer_pairs(
                data_path, tokenizer, max_prompt_length, data_name
            )
        )

    # Convert the list of components to a pandas DataFrame
    df = pd.DataFrame(components)

    # Split the 'answers' column into 'answer1' and 'answer2'
    df["answer1"] = df["answers"].apply(lambda x: x[0])
    df["answer2"] = df["answers"].apply(lambda x: x[1])
    df = df.drop(columns=["answers"])

    def convert_to_request(row):
        return {
            "custom_id": hashlib.sha256(
                (
                    row["paragraph"] + row["question"] + row["answer1"] + row["answer2"]
                ).encode()
            ).hexdigest()
            + f"_SPLIT_{row['data_name']}_CORRECT_ANSWER_{row['correct_answer']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(**row)},
                ],
                "max_tokens": 400,
            },
        }

    # Use the above function to convert the DataFrame to a list of requests
    print("Converting dataset to requests")
    requests = df.apply(convert_to_request, axis=1)

    # Create a new DataFrame with just the requests
    requests_df = pd.DataFrame(requests.tolist())

    # Split the DataFrame into batches and write them to a file
    for batch_id, start_index in enumerate(range(0, len(requests_df), MAX_BATCH_SIZE)):
        end_index = min(start_index + MAX_BATCH_SIZE, len(requests_df))
        batch = requests_df.iloc[start_index:end_index]

        suffix = f"_part_{batch_id}.jsonl"

        # Replace the suffix with the custom suffix.
        # This is the easiest way I could find to do this
        batch_file_name = (
            str(Path(BATCH_FILE_NAME).parent / Path(BATCH_FILE_NAME).stem) + suffix
        )

        print(f"Saving batch file {batch_file_name}")

        batch.to_json(batch_file_name, orient="records", lines=True)


def launch_batch_jobs():
    # Find all files in the parent directory that start with the prefix
    parent_dir = Path(BATCH_FILE_NAME).parent
    file_pattern = f"{Path(BATCH_FILE_NAME).stem}_part_*.jsonl"
    file_paths = list(parent_dir.glob(file_pattern))

    for file_path in file_paths:
        launch_batch_job_impl(file_path)


def retrieve_batch_results():
    prefix = str(Path(BATCH_FILE_NAME).parent / Path(BATCH_FILE_NAME).stem)

    for index, file_name in enumerate(FILES_TO_DOWNLOAD):
        retrieve_batch_results_impl(file_name, f"{prefix}_part_{index}_output.jsonl")


def compute_prompt_sufficiency():
    parent_dir = Path(BATCH_FILE_NAME).parent
    pattern = str(Path(BATCH_FILE_NAME).stem) + "_part_*_output.jsonl"

    # Read all the output files
    file_paths = list(parent_dir.glob(pattern))
    outputs = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            for line in f:
                raw_output = json.loads(line)
                split = (
                    raw_output["custom_id"]
                    .split("_SPLIT_")[1]
                    .split("_CORRECT_ANSWER_")[0]
                    .strip()
                )
                correct_answer = (
                    raw_output["custom_id"]
                    .split("_SPLIT_")[1]
                    .split("_CORRECT_ANSWER_")[1]
                    .strip()
                )
                outputs.append(
                    {
                        "custom_id": raw_output["custom_id"],
                        "response": raw_output["response"]["body"]["choices"][0][
                            "message"
                        ]["content"],
                        "split": split,
                        "correct_answer": correct_answer,
                    }
                )

    if not file_paths:
        print("No files found")
        return

    dataframe = pd.DataFrame(outputs)

    print(f"columns: {dataframe.columns}")

    # Process the dataframe to analyze sufficiency
    def extract_sufficiency(response):
        # A common failure mode of GPT-4o is to write "**SUFFICIENT:** YES/NO" instead of "SUFFICIENT: YES/NO".
        # Fix this by removing all ** characters.
        response = response.replace("**", "")

        if "SUFFICIENT: YES" in response:
            return "YES"
        elif "SUFFICIENT: NO" in response:
            return "NO"
        else:
            return "ERROR"

    # Extract predicted answer from response
    def extract_prediction(response):
        # A common failure mode of GPT-4o is to write "**SUFFICIENT:** YES/NO" instead of "SUFFICIENT: YES/NO".
        # Fix this by removing all ** characters.
        response = response.replace("**", "")
        if "CORRECT ANSWER: A" in response:
            return "A"
        elif "CORRECT ANSWER: B" in response:
            return "B"
        else:
            return "ERROR"

    # Add sufficiency and prediction columns to the dataframe
    dataframe["sufficiency"] = dataframe["response"].apply(extract_sufficiency)
    dataframe["prediction"] = dataframe["response"].apply(extract_prediction)

    # Group by split and calculate statistics for each split
    for split_name, split_df in dataframe.groupby("split"):
        total_count = len(split_df)
        yes_count = sum(split_df["sufficiency"] == "YES")
        no_count = sum(split_df["sufficiency"] == "NO")
        error_count = sum(split_df["sufficiency"] == "ERROR")

        # Print summary statistics for this split
        print(f"\nDataset Quality Summary for {split_name} split:")
        print(f"Total examples analyzed: {total_count}")
        print(f"SUFFICIENT YES: {yes_count} ({(yes_count / total_count) * 100:.1f}%)")
        print(f"SUFFICIENT NO: {no_count} ({(no_count / total_count) * 100:.1f}%)")
        print(
            f"Parsing errors: {error_count} ({(error_count / total_count) * 100:.1f}%)"
        )

        # Compute answer accuracy statistics
        # Compute the prediction distribution
        prediction_distribution = split_df["prediction"].value_counts()
        print(f"Prediction distribution: {prediction_distribution}")

        # Compute the prediction distribution for all rows where 'sufficiency' is 'YES'
        sufficient_prediction_distribution = split_df[split_df["sufficiency"] == "YES"][
            "prediction"
        ].value_counts()
        print(
            f"Prediction distribution for sufficient rows: {sufficient_prediction_distribution}"
        )

        # Compute the label distribution
        label_distribution = split_df["correct_answer"].value_counts()
        print(f"Label distribution: {label_distribution}")

        # Compute the label distribution for all rows where 'sufficiency' is 'YES'
        sufficient_label_distribution = split_df[split_df["sufficiency"] == "YES"][
            "correct_answer"
        ].value_counts()
        print(
            f"Label distribution for sufficient rows: {sufficient_label_distribution}"
        )

        # Compute the general accuracy
        general_accuracy = (split_df["prediction"] == split_df["correct_answer"]).mean()
        print(f"General accuracy: {general_accuracy:.2f}")

        # Compute the accuracy of only the rows where 'sufficiency' is 'YES'
        sufficient_df = split_df[split_df["sufficiency"] == "YES"]
        if len(sufficient_df) > 0:
            sufficient_accuracy = (
                sufficient_df["prediction"] == sufficient_df["correct_answer"]
            ).mean()
            print(f"Accuracy on sufficient rows: {sufficient_accuracy:.2f}")

        # Compute the accuracy of all rows where 'prediction' is equal to 'A'
        A_df = split_df[split_df["prediction"] == "A"]
        if len(A_df) > 0:
            A_accuracy = (A_df["prediction"] == A_df["correct_answer"]).mean()
            print(f"Accuracy on rows where prediction is 'A': {A_accuracy:.2f}")

        # Compute the accuracy of all rows where 'prediction' is equal to 'B'
        B_df = split_df[split_df["prediction"] == "B"]
        if len(B_df) > 0:
            B_accuracy = (B_df["prediction"] == B_df["correct_answer"]).mean()
            print(f"Accuracy on rows where prediction is 'B': {B_accuracy:.2f}")

        print("\n" + "=" * 50)


ACTION_MAP = {
    "prepare_batch_files": prepare_batch_files,
    "launch_batch_jobs": launch_batch_jobs,
    "retrieve_batch_results": retrieve_batch_results,
    "compute_prompt_sufficiency": compute_prompt_sufficiency,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str)

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(
            f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}"
        )
