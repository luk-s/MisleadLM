import argparse
import json
import pathlib
from typing import List

import flask
import safetensors.torch
import torch
from flask import Flask, jsonify, make_response, request
from reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

app = Flask(__name__)

CURRENT_DIR = pathlib.Path(__file__).parent

# The values are tuples of (reward_model_architecture, reward_model_checkpoint_path)
REWARD_MODEL_PATHS = {
    "human_labels_Llama3.1": (
        "meta-llama/Llama-3.1-8B",
        CURRENT_DIR
        / "model_checkpoints/reward_models/MODEL_Llama-3.1-8B_DATA_human_LR_1e-5_BC_16_MAXEPOCH_5_TIME_2025-02-12_18-11-40/checkpoint-1900/model.safetensors",
    ),
    "human_labels_Llama3": (
        "meta-llama/Llama-3-8B",
        CURRENT_DIR
        / "model_checkpoints/reward_models/MODEL_Meta-Llama-3-8B_DATA_human_LR_1e-5_BC_16_MAXEPOCH_5_TIME_2025-02-21_15-07-20/checkpoint-800/model.safetensors",
    ),
    "human_labels_Llama3.2_1B": (
        "meta-llama/Llama-3.2-1B",
        CURRENT_DIR
        / "model_checkpoints/reward_models/MODEL_Llama-3.2-1B-hf_DATA_human_LR_1e-5_BC_16_MAXEPOCH_5_TIME_2025-02-28_14-57-31/checkpoint-1900/model.safetensors",
    ),
}
BATCH_SIZE = 4


def setup_reward_model(
    reward_model_architecture: str, checkpoint_path: str
) -> tuple[AutoTokenizer, GPTRewardModel, torch.device]:
    """Sets up the reward model with the specified architecture and checkpoint.

    Args:
        reward_model_architecture (str): The name/path of the model architecture to use
        checkpoint_path (str): Path to the model checkpoint file

    Returns:
        tuple[AutoTokenizer, GPTRewardModel, torch.device]: A tuple containing:
            - The tokenizer for the reward model
            - The initialized reward model
            - The device the model is loaded on
    """
    rw_tokenizer = AutoTokenizer.from_pretrained(
        reward_model_architecture, use_fast=False
    )

    rw_tokenizer.padding_side = "right"

    if "Llama-2" in reward_model_architecture:
        if rw_tokenizer.pad_token is None:
            rw_tokenizer.pad_token = rw_tokenizer.unk_token
            rw_tokenizer.pad_token_id = rw_tokenizer.unk_token_id
    elif "Llama-3" in reward_model_architecture:
        if rw_tokenizer.pad_token is None:
            rw_tokenizer.pad_token = "<|finetune_right_pad_id|>"
            rw_tokenizer.pad_token_id = 128004
    else:
        print(f"Unknown model: {reward_model_architecture}")

    rw_model = GPTRewardModel(reward_model_architecture, reward_model_architecture)
    print("Loading weights")
    rw_model.load_state_dict(safetensors.torch.load_file(checkpoint_path))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(0))
    rw_model.to(rw_device)
    print("reward model loaded!")

    return rw_tokenizer, rw_model, rw_device


def get_scores(samples: List[str]) -> torch.Tensor:
    """Calculates reward scores for a list of text samples using the reward model.

    Args:
        samples (List[str]): List of text samples to score

    Returns:
        torch.Tensor: Tensor containing reward scores for each sample
    """
    # Print some stats about the samples
    sample_lengths = [len(sample) for sample in samples]
    print(f"Sample lengths: {sample_lengths}")
    print(f"Number of samples: {len(samples)}")

    scores_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), BATCH_SIZE)):
            sub_samples = samples[i : i + BATCH_SIZE]
            sub_samples = [chosen + rw_tokenizer.eos_token for chosen in sub_samples]

            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=False,
                padding="longest",
                return_tensors="pt",
            )

            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores


@app.route("/reward", methods=["POST"])
def get_reward() -> flask.Response:
    """Flask endpoint that accepts text samples and returns their reward scores.

    Expects a JSON array of strings in the request body.

    Returns:
        flask.Response: JSON response containing an array of reward scores
    """
    data = json.loads(request.data)
    samples = data
    scores = get_scores(samples)
    scores = scores.detach().cpu().tolist()
    return make_response(jsonify(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_model",
        type=str,
        required=True,
        help="The reward model to use",
        choices=REWARD_MODEL_PATHS.keys(),
    )
    args = parser.parse_args()

    reward_model_architecture, reward_model_checkpoint_path = REWARD_MODEL_PATHS[
        args.reward_model
    ]

    rw_tokenizer, rw_model, rw_device = setup_reward_model(
        reward_model_architecture, reward_model_checkpoint_path
    )

    app.run(debug=False, host="0.0.0.0", port=8115)
