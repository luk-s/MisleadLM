from pathlib import Path

from datasets import load_dataset, Dataset
from argparse import ArgumentParser
import ast

TRAIN_SIZE = 38_716
OUTPUT_TRAIN_NAME = "train_human.json"
OUTPUT_EVAL_NAME = "test_human.json"


def build_training_dataset():
    # Load the original arena-55k dataset
    dataset: Dataset = load_dataset("lmarena-ai/arena-human-preference-55k")["train"]

    dataset = dataset.sort("id")

    # Drop all rows where both responses are equally good
    dataset = dataset.filter(lambda row: row["winner_tie"] == 0)

    # Change the format of the dataset, such that it can be used for training
    def change_format(row):
        if row["winner_model_a"] == 1:
            response_win = row["response_a"]
            response_lose = row["response_b"]
        else:
            response_win = row["response_b"]
            response_lose = row["response_a"]

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

    dataset = dataset.map(
        change_format,
        remove_columns=dataset.column_names,
    )

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
