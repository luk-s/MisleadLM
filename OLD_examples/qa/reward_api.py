import argparse
import json
import pathlib
from typing import List

import safetensors.torch
import torch
from flask import Flask, jsonify, make_response, request
from reward.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig

app = Flask(__name__)

REWARD_MODELS = {
    "human_labels": "reward/outputs/human_labels/checkpoint-700/model.safetensors",
    "openai_simple_labels": "reward/outputs/openai_simple_labels/checkpoint-1000/model.safetensors",
    "openai_unbiased_labels": "reward/outputs/openai_unbiased_labels/checkpoint-700/model.safetensors",
    "openai_unbiased_logprobs_labels": "reward/outputs/openai_unbiased_logprobs_labels/checkpoint-700/model.safetensors",
}

SFT_MODEL_PATH = "meta-llama/Llama-2-13b-hf"

TOKENIZER_PATH = "meta-llama/Llama-2-13b-hf"


def setup_reward_model(tokenizer_path, model_path, checkpoint_path):
    global rw_tokenizer, rw_model, rw_device

    rw_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if rw_tokenizer.pad_token is None:
        rw_tokenizer.pad_token = rw_tokenizer.unk_token
        print("set pad token to unk token: ", rw_tokenizer.pad_token)

    rw_model = GPTRewardModel(model_path, tokenizer_path)
    print("Loading weights")
    rw_model.load_state_dict(safetensors.torch.load_file(checkpoint_path))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(0))
    rw_model.to(rw_device)
    print("reward model loaded!")


def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [chosen + rw_tokenizer.eos_token for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=1024,
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
def get_reward():
    data = json.loads(request.data)
    samples = data
    scores = get_scores(samples)
    scores = scores.detach().cpu().tolist()
    return make_response(jsonify(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_model", type=str, required=True, help="The reward model to use", choices=REWARD_MODELS.keys()
    )
    args = parser.parse_args()

    REWARD_CHECKPOINT_PATH = REWARD_MODELS[args.reward_model]

    setup_reward_model(TOKENIZER_PATH, SFT_MODEL_PATH, REWARD_CHECKPOINT_PATH)

    app.run(debug=False, host="0.0.0.0", port=8115)
