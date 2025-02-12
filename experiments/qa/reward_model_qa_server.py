import argparse
import json
import pathlib
import re
from typing import List

import torch
from flask import Flask, jsonify, make_response, request
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from trlx.data.configs import TRLConfig

app = Flask(__name__)


model_config = "XXX"
load_dir = "XXX"

rw_tokenizer = AutoTokenizer.from_pretrained(model_config)
if rw_tokenizer.pad_token is None:
    rw_tokenizer.pad_token = rw_tokenizer.unk_token

rw_model = AutoModelForSequenceClassification.from_pretrained(load_dir, num_labels=2)
rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(0))
rw_model.to(rw_device)


def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 8
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            prompts = samples[i : i + batch_size]
            prompts = [f"{i}{rw_tokenizer.eos_token}" for i in prompts]
            encodings_dict = rw_tokenizer(
                prompts,
                truncation=True,
                max_length=4096,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            with torch.no_grad():
                outputs = rw_model(input_ids=input_ids, attention_mask=attn_masks)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().detach()
                probs = probs[:, 1]
            scores_list.append(probs)
    scores = torch.cat(scores_list, dim=0)
    return scores


@app.route("/judge", methods=["POST"])
def get_reward():
    data = json.loads(request.data)
    scores = get_scores(data)
    scores = scores.detach().cpu().tolist()
    return make_response(jsonify(scores))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8119)
