from argparse import ArgumentParser
from typing import Any, Dict

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

# Only used for computing the score alignment
SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_FLIPPED_formatted.jsonl"

# SCORES_TO_COMPUTE_ALIGNMENT = "auxiliary_files/arena-55k-batch_openai_formatted.jsonl"
# SCORES_TO_COMPUTE_ALIGNMENT = "examples/qa/data/arena-55k/auxiliary_files/arena-55k-batch_openai_formatted.jsonl"

DATASET_IS_FLIPPED = False
SHOW_PERCENTAGES = True


def compute_score_matrix():
    # Load the score dataset
    score_dataset = load_dataset("json", data_files=SCORES_TO_COMPUTE_ALIGNMENT)["train"]

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

    dataset = dataset.filter(lambda row: extract_prediction_group(row)["group"] is not None)
    dataset = dataset.map(extract_prediction_group)

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


ACTION_MAP = {
    "compute_score_matrix": compute_score_matrix,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str)

    args = parser.parse_args()

    if args.action in ACTION_MAP:
        ACTION_MAP[args.action]()
    else:
        raise ValueError(f"Unknown action: {args.action}. Choose one of {list(ACTION_MAP.keys())}")
