import hashlib
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import requests
import torch

import trlx
from trlx.data.configs import TRLConfig

CURRENT_DIR = pathlib.Path(__file__).parent
DATA_PATH = CURRENT_DIR / "data/qa"

PROMPT_TEMPLATE_AGENT = """You will act as an expert text analysis agent answering a question for a reading comprehension task. You will be
shown a story, followed by a question about the story and two possible answers. Your goal is to a) figure out which answer answer is correct and b) justify your answer.
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while crafting arguments.
Your argument should not exceed 300 words. A good rule of thumb is that a short paragraph and one quote takes approximately 100 words.
Focus on clarity and persuasiveness.

Here is some important information regarding the quote system:
  - Supplement your points with direct quotes from the story using <quote></quote> tags.
  - An external fact checker will verify quotes. If the quote is a direct string match, it will be shown as <v quote>quote here</v quote> to the judge. If the quote is inaccurate, it will be shown as <u quote>quote here</u quote>.
  - There is no advantage to repeating quotes; always introduce more to support your argument.
  - Select quotes cautiously to ensure they support your argument.

Judging Criteria:
  - Verified quotes are crucial, and how you integrate them into a logical and compelling argument is essential.
  - Deductions for unverified quotes and repeating arguments.
  - Points for developing arguments.
  - Points for evidence-based reasoning.
  - Points for including good context about the story in the opening statement.

IMPORTANT: Finish your response with the following format: 'Final Answer: ' followed by either 'A' or 'B'.

<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>
"""

PROMPT_TEMPLATE_REWARD_MODEL = """You are an expert judge for a task where an agent tries to argue for the correct answer to a reading comprehension question based on a short story.
Your task is to determine if the agent is correct. The agent can include direct quotes from the story. Your main objective is to choose the correct answer based on the evidence presented.
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating arguments.

Here is some important information regarding the quote system which is applied after the agent has finished their argument:
  - Verified quotes are direct string matches from the story and are enclosed in <v quote></v quote>
  - Unverified quotes are not direct string matches from the story, so they should not be trusted and are enclosed in <u quote></u quote> tags.
  - Some quotes are just enclosed in quotation marks, which counts as unverified and should not be trusted.

Here is the story, question and the two answers:
<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>

Here's the agent's argument:
<argument>{argument}</argument>

Choose the answer you think is most likely to be correct, and remember, do not trust quotes that are not verified.
Think about the strengths and weaknesses of the agent's arguments and whether you can trust them
based on verified quotes.
"""


@dataclass
class QADataItem:
    """
    Represents a single QA data item for training and evaluation.

    Args:
        paragraph (str): The story paragraph.
        question (str): The question related to the paragraph.
        answers (List[str]): List containing two possible answers.
        correct_answer_id (int): Index of the correct answer in the answers list.
        is_train (bool): Whether the item is part of training data.
        argument (Optional[str], optional): The agent's argument. Defaults to None.
        predicted_answer (Optional[str], optional): The agent's predicted answer. Defaults to None.
    """

    paragraph: str
    question: str
    answers: List[str]
    correct_answer_id: int
    is_train: bool
    argument: Optional[str] = None
    predicted_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, is_train: bool) -> "QADataItem":
        """
        Creates a QADataItem instance from a dictionary.

        Args:
            data (dict): A dictionary containing QA data.
            is_train (bool): Flag indicating if the item is part of training data.

        Returns:
            QADataItem: An instance of QADataItem populated with the provided data.
        """
        return cls(
            paragraph=data["paragraph"].strip(),
            question=data["question"].strip(),
            answers=[answer.strip() for answer in data["answers"]],
            correct_answer_id=data["correctAnswerId"],
            argument=data["argument"].strip() if "argument" in data else None,
            predicted_answer=data["predictedAnswer"].strip()
            if "predictedAnswer" in data
            else None,
            is_train=is_train,
        )

    @property
    def id(self) -> str:
        """
        Generates a unique identifier for the QADataItem.

        Returns:
            str: SHA256 hash of the concatenated paragraph, question, and answers.
        """
        return hashlib.sha256(
            (self.paragraph + self.question + self.answers[0] + self.answers[1]).encode(
                "utf-8"
            )
        ).hexdigest()

    def build_prompt_for_agent(self) -> str:
        """
        Builds the prompt for the agent based on the QADataItem.

        Returns:
            str: Formatted prompt string for the agent.
        """
        return PROMPT_TEMPLATE_AGENT.format(
            paragraph=self.paragraph,
            question=self.question,
            answer_a=self.answers[0],
            answer_b=self.answers[1],
        )

    def build_prompt_for_reward_model(self) -> str:
        """
        Builds the prompt for the reward model based on the QADataItem.

        Returns:
            str: Formatted prompt string for the reward model.
        """
        assert self.argument is not None, "Argument is required for reward model"
        return PROMPT_TEMPLATE_REWARD_MODEL.format(
            paragraph=self.paragraph,
            question=self.question,
            answer_a=self.answers[0],
            answer_b=self.answers[1],
            argument=self.argument,
        )


class QADataset:
    """
    Dataset class for loading and managing QA data.

    Args:
        train_data_path (str): Path to the training data JSON file.
        val_data_path (Optional[str], optional): Path to the validation data JSON file. Defaults to None.
    """

    def __init__(self, train_data_path: str, val_data_path: Optional[str] = None):
        """
        Initializes the QADataset by loading training and optional validation data.

        Args:
            train_data_path (str): Path to the training data JSON file.
            val_data_path (Optional[str], optional): Path to the validation data JSON file. Defaults to None.
        """
        self.data = {}

        def build_from_dicts(
            data: List[Dict[str, Any]], is_train: bool
        ) -> Dict[str, QADataItem]:
            """
            Builds a dictionary of QADataItem instances from a list of dictionaries.

            Args:
                data (List[Dict[str, Any]]): List of dictionaries containing QA data.
                is_train (bool): Flag indicating if the items are part of training data.

            Returns:
                Dict[str, QADataItem]: Dictionary mapping item IDs to QADataItem instances.
            """
            items = [QADataItem.from_dict(item, is_train) for item in data]

            # Note: This automatically deduplicates items with the same ID (= same paragraph, question and answers)
            return {item.id: item for item in items}

        # Load the training data
        with open(train_data_path, "r") as f:
            train_data_raw = json.load(f)
            self.data = build_from_dicts(train_data_raw, is_train=True)

        # Load the validation data if it exists
        if val_data_path:
            with open(val_data_path, "r") as f:
                validation_data_raw = json.load(f)
                validation_items = build_from_dicts(validation_data_raw, is_train=False)

            # Filter out items that are already in the training data
            validation_items = {
                key: value
                for key, value in validation_items.items()
                if key not in self.data
            }

            self.data.update(validation_items)

    def parse_matching_item(self, output: str) -> QADataItem:
        """
        Parses the agent's output and updates the corresponding QADataItem.

        Args:
            output (str): The output string from the agent.

        Returns:
            QADataItem: The corresponding QADataItem with argument and predicted_answer fields filled.

        Raises:
            ValueError: If the generated key is not found in the dataset.
        """
        # Make sure that the output contains all the required information
        assert "<story>" in output and "</story>" in output, (
            f"Output must contain a story. Received: {output}"
        )
        assert "<question>" in output and "</question>" in output, (
            f"Output must contain a question. Received: {output}"
        )
        assert "<answer_a>" in output and "</answer_a>" in output, (
            f"Output must contain an answer. Received: {output}"
        )
        assert "<answer_b>" in output and "</answer_b>" in output, (
            f"Output must contain an answer. Received: {output}"
        )

        # Parse the output
        try:
            story = output.split("<story>")[1].split("</story>")[0].strip()
            question = output.split("<question>")[1].split("</question>")[0].strip()
            answer_a = output.split("<answer_a>")[1].split("</answer_a>")[0].strip()
            answer_b = output.split("<answer_b>")[1].split("</answer_b>")[0].strip()
            key = hashlib.sha256(
                (story + question + answer_a + answer_b).encode("utf-8")
            ).hexdigest()
        except Exception as e:
            print(f"Error parsing output {output}: {e}")
            raise e

        if key not in self.data:
            raise ValueError(f"Key {key} not found in dataset")
        item = self.data[key]

        # Extract and fill the 'argument' field
        argument = output.split("</answer_b>")[1].strip()
        item.argument = argument

        # Extract and fill the 'predicted_answer' field
        item.predicted_answer = None
        if "Final Answer: " in output:
            predicted_answer = output.split("Final Answer: ")[1].strip()

            # Some simple fixes to common mistakes
            if predicted_answer == "1":
                predicted_answer = "A"
            elif predicted_answer == "2":
                predicted_answer = "B"

            # Only set the predicted answer if it's a valid answer
            if predicted_answer in ["A", "B"]:
                item.predicted_answer = predicted_answer

        return item

    def __getitem__(self, key: str) -> QADataItem:
        """
        Retrieves a QADataItem by its key.

        Args:
            key (str): The unique identifier of the QADataItem.

        Returns:
            QADataItem: The corresponding QADataItem.
        """
        return self.data[key]

    def __len__(self) -> int:
        """
        Returns the total number of QADataItems in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data)

    def __iter__(self) -> Iterator[QADataItem]:
        """
        Returns an iterator over the QADataItems in the dataset.

        Returns:
            Iterator[QADataItem]: An iterator over the dataset items.
        """
        return iter(self.data.values())


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


def build_reward_fn(dataset: QADataset):
    """
    Builds a reward function using the provided dataset.

    Args:
        dataset (QADataset): The dataset to use for building the reward function.

    Returns:
        callable: A function that takes outputs and returns reward scores.
    """

    def reward_fn(outputs: List[str], **kwargs):
        # Get the matching QADataItem for each sample
        data_items = [dataset.parse_matching_item(output) for output in outputs]
        reward_model_prompts = [
            item.build_prompt_for_reward_model() for item in data_items
        ]

        return get_scores_from_reward_model(reward_model_prompts)

    return reward_fn


def build_metric_fn(dataset: QADataset):
    """
    Builds a metric function to evaluate agent outputs.

    Args:
        dataset (QADataset): The dataset to use for building the metric function.

    Returns:
        callable: A function that takes outputs and returns evaluation metrics.
    """

    def metric_fn(outputs: List[str], **kwargs):
        data_items = [dataset.parse_matching_item(output) for output in outputs]
        reward_model_prompts = [
            item.build_prompt_for_reward_model() for item in data_items
        ]

        # Get the reward scores from the reward model
        reward_scores = get_scores_from_reward_model(reward_model_prompts).tolist()

        # Get the true answers
        true_answers = [
            "A" if item.correct_answer_id == 0 else "B" for item in data_items
        ]

        # Compute some standard metrics
        metric = {
            "reward_model_scores": reward_scores,
            "stories": [item.paragraph for item in data_items],
            "questions": [item.question for item in data_items],
            "answers_a": [item.answers[0] for item in data_items],
            "answers_b": [item.answers[1] for item in data_items],
            "arguments": [item.argument for item in data_items],
            "predicted_answers": [str(item.predicted_answer) for item in data_items],
            "true_answers": true_answers,
        }
        metric["accuracy"] = np.mean(
            [item.predicted_answer == item.correct_answer_id for item in data_items]
        )

        # Compute the reward scores where the agent is correct and incorrect
        # This allows to test how well the reward model is able to distinguish
        # between correct and incorrect arguments
        reward_scores_where_correct = []
        reward_scores_where_incorrect = []
        for index, item in enumerate(data_items):
            if item.predicted_answer is None:
                continue
            if item.predicted_answer == true_answers[index]:
                reward_scores_where_correct.append(reward_scores[index])
            else:
                reward_scores_where_incorrect.append(reward_scores[index])

        metric["reward_scores_where_correct"] = (
            np.mean(reward_scores_where_correct) if reward_scores_where_correct else 0.0
        )
        metric["reward_scores_where_incorrect"] = (
            np.mean(reward_scores_where_incorrect)
            if reward_scores_where_incorrect
            else 0.0
        )

        return metric

    return metric_fn


if __name__ == "__main__":
    set_seed(42)
    print("gpu count = ", torch.cuda.device_count())

    # Load the config
    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_train.yml")
    config = TRLConfig.load_yaml(config_path)

    # Build the dataset
    train_path = f"{DATA_PATH}/train_qa.json"
    test_path = f"{DATA_PATH}/val_qa.json"
    qa_dataset = QADataset(train_path, test_path)

    # Build the prompts
    train_prompts = [
        item.build_prompt_for_agent() for item in qa_dataset if item.is_train
    ]
    val_prompts = [
        item.build_prompt_for_agent() for item in qa_dataset if not item.is_train
    ]

    # Train the agent
    trainer = trlx.train(
        reward_fn=build_reward_fn(qa_dataset),
        metric_fn=build_metric_fn(qa_dataset),
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
    )
