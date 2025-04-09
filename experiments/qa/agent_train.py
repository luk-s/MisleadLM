import json
import pathlib
import random
from datetime import datetime
from typing import List

import numpy as np
import requests
import torch
from qa_dataset import QADataset
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import TRLConfig

CURRENT_DIR = pathlib.Path(__file__).parent
DATA_PATH = CURRENT_DIR / "data/qa"


def set_seed(seed_val=42):
    """
    Sets the random seed for reproducibility.

    Args:
        seed_val (int, optional): The seed value. Defaults to 42.
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_scores_from_reward_model(prompts):
    """
    Sends prompts to the reward model server and retrieves scores.

    Args:
        prompts (List[str]): A list of prompt strings to evaluate.

    Returns:
        torch.Tensor: Tensor containing the reward scores.
    """
    # Send the prompts to the reward model server
    url = "http://localhost:8115/reward"
    resp = requests.post(url, data=json.dumps(prompts))

    # Extract the scores from the response
    scores = resp.json()
    scores = torch.tensor(scores, dtype=torch.float)
    return scores


def build_reward_fn(
    dataset: QADataset,
    tokenizer: AutoTokenizer,
    skip_start_and_end_tokens: bool = True,
    compute_reward_model_scores: bool = True,
):
    """
    Builds a reward function using the provided dataset.

    Args:
        dataset (QADataset): The dataset to use for building the reward function.
        tokenizer (AutoTokenizer): The tokenizer to use to build the prompts for the reward model.
        skip_start_and_end_tokens (bool, optional): Whether to skip the start and end tokens when building the prompts for the reward model. Defaults to True.
        compute_reward_model_scores (bool): Whether to compute reward model scores. Defaults to True.

    Returns:
        callable: A function that takes outputs and returns reward scores.
    """

    def reward_fn(samples: List[str], **kwargs):
        # Get the matching QADataItem for each sample
        data_items = [dataset.parse_matching_item(sample) for sample in samples]
        reward_model_prompts = [
            item.build_prompt_for_reward_model(
                tokenizer, skip_start_and_end_tokens=skip_start_and_end_tokens
            )
            for item in data_items
        ]

        if compute_reward_model_scores:
            return get_scores_from_reward_model(reward_model_prompts)
        else:
            return -torch.ones(len(samples))

    return reward_fn


def build_metric_fn(
    dataset: QADataset,
    tokenizer: AutoTokenizer,
    skip_start_and_end_tokens: bool = True,
    compute_reward_model_scores: bool = True,
):
    """
    Builds a metric function to evaluate agent outputs.

    Args:
        dataset (QADataset): The dataset to use for building the metric function.
        tokenizer (AutoTokenizer): The tokenizer to use to build the prompts for the reward model.
        skip_start_and_end_tokens (bool, optional): Whether to skip the start and end tokens when building the prompts for the reward model. Defaults to True.
        compute_reward_model_scores (bool): Whether to compute reward model scores. Defaults to True.

    Returns:
        callable: A function that takes outputs and returns evaluation metrics.
    """

    def metric_fn(samples: List[str], **kwargs):
        data_items = [dataset.parse_matching_item(sample) for sample in samples]
        reward_model_prompts = [
            item.build_prompt_for_reward_model(
                tokenizer, skip_start_and_end_tokens=True
            )
            for item in data_items
        ]

        if compute_reward_model_scores:
            # Get the reward scores from the reward model
            reward_scores = get_scores_from_reward_model(reward_model_prompts).tolist()
        else:
            reward_scores = -torch.ones(len(samples)).tolist()

        # Get the true answers
        true_answers = [
            "A" if item.correct_answer_id == 0 else "B" for item in data_items
        ]

        # Compute some standard metrics
        metric = {
            "reward": reward_scores,
            "stories": [item.paragraph for item in data_items],
            "questions": [item.question for item in data_items],
            "answers_a": [item.answers[0] for item in data_items],
            "answers_b": [item.answers[1] for item in data_items],
            "arguments": [item.verified_argument for item in data_items],
            "predicted_answers": [str(item.predicted_answer) for item in data_items],
            "true_answers": true_answers,
        }
        metric["accuracy"] = np.mean(
            [
                data_items[index].predicted_answer == true_answers[index]
                for index in range(len(data_items))
            ]
        )
        metric["accuracy_where_complete"] = np.mean(
            [
                data_items[index].predicted_answer == true_answers[index]
                for index in range(len(data_items))
                if data_items[index].predicted_answer is not None
            ]
        )
        metric["fraction_incomplete_responses"] = np.mean(
            [item.predicted_answer is None for item in data_items]
        )

        metric["fraction_model_responds_A"] = np.mean(
            [item.predicted_answer == "A" for item in data_items]
        )
        metric["fraction_model_responds_B"] = np.mean(
            [item.predicted_answer == "B" for item in data_items]
        )

        # Compute the reward scores where the agent is correct and incorrect
        # This allows to test how well the reward model is able to distinguish
        # between correct and incorrect arguments
        reward_scores_where_correct = []
        reward_scores_where_incorrect = []
        reward_scores_where_incomplete_responses = []
        for index, item in enumerate(data_items):
            if item.predicted_answer is None:
                reward_scores_where_incomplete_responses.append(reward_scores[index])
            elif item.predicted_answer == true_answers[index]:
                reward_scores_where_correct.append(reward_scores[index])
            else:
                reward_scores_where_incorrect.append(reward_scores[index])

        metric["reward_where_correct"] = (
            np.mean(reward_scores_where_correct) if reward_scores_where_correct else 0.0
        )
        metric["reward_where_incorrect"] = (
            np.mean(reward_scores_where_incorrect)
            if reward_scores_where_incorrect
            else 0.0
        )
        metric["reward_where_incomplete_responses"] = (
            np.mean(reward_scores_where_incomplete_responses)
            if reward_scores_where_incomplete_responses
            else 0.0
        )

        return metric

    return metric_fn


if __name__ == "__main__":
    set_seed(42)

    # Load the config
    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_train.yml")
    config = TRLConfig.load_yaml(config_path)

    # Append the current timestamp to the checkpoint directory
    config.train.checkpoint_dir = (
        f"{config.train.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Build the dataset
    # train_path = f"{DATA_PATH}/train_qa.json"
    # test_path = f"{DATA_PATH}/val_qa.json"
    train_path = f"{DATA_PATH}/train_qa_le8000_balanced.json"
    test_path = f"{DATA_PATH}/val_qa_le8000_balanced.json"
    max_paragraph_length = None
    print(f"Using max paragraph length: {max_paragraph_length}")
    qa_dataset = QADataset(
        train_data_path=train_path,
        val_data_path=test_path,
        include_argument_and_label=False,  # This should only be set to 'True' for reward model training.
        max_paragraph_length=max_paragraph_length,
    )

    # Build the prompts
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    train_prompts = [
        item.build_prompt_for_agent(tokenizer, skip_bos=True)
        for item in qa_dataset
        if item.is_train
    ]
    val_prompts = [
        item.build_prompt_for_agent(tokenizer, skip_bos=True)
        for item in qa_dataset
        if not item.is_train
    ]

    # Train the agent
    trainer = trlx.train(
        reward_fn=build_reward_fn(
            qa_dataset, tokenizer, skip_start_and_end_tokens=True
        ),
        metric_fn=build_metric_fn(
            qa_dataset, tokenizer, skip_start_and_end_tokens=True
        ),
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
    )
