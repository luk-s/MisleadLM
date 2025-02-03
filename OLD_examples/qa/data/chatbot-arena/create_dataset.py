import json

from datasets import load_dataset


def build_chatbot_arena_eval_dataset() -> None:
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")

    # Remove all ties
    dataset = dataset.filter(lambda row: "tie" not in row["winner"])

    def format_row(row: dict) -> dict:
        if row["winner"] == "model_a":
            conversation_win = row["conversation_a"]
            conversation_lose = row["conversation_b"]
        elif row["winner"] == "model_b":
            conversation_win = row["conversation_b"]
            conversation_lose = row["conversation_a"]
        else:
            raise ValueError(f"Unknown winner: {row['winner']}")
        return {
            "question_id": row["question_id"],
            "win": conversation_win,
            "lose": conversation_lose,
        }

    dataset = dataset.map(format_row, remove_columns=dataset.column_names)

    dataset_dict = dataset.to_dict()
    list_of_dicts = [{key: dataset_dict[key][i] for key in dataset_dict.keys()} for i in range(len(dataset))]
    with open("evaluation_data.json", "w") as f:
        for row in list_of_dicts:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    build_chatbot_arena_eval_dataset()
