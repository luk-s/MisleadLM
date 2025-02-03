import json

from datasets import VerificationMode, load_dataset

dataset_debate = load_dataset(
    "csv", data_files="examples/qa/data/qa/data/llm_debate_dataset/llm_debate_human_judge_dataset.csv"
)["train"]

dataset_debate = dataset_debate.map(
    lambda row: {"correct_argument": json.loads(row["transcript"])["rounds"][0]["correct"]}
)

# Only keep rows with correct arguments
dataset_debate = dataset_debate.filter(lambda row: row["correct_argument"] is not None)

dataset_qa = load_dataset(
    "json",
    data_files={
        "train": "examples/qa/data/qa/QuALITY.v1.0.1.htmlstripped.train",
        "test": "examples/qa/data/qa/QuALITY.v1.0.1.htmlstripped.test",
    },
    verification_mode=VerificationMode.NO_CHECKS,
)

print("Done!")
