import hashlib
import json
import logging
import random
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai_batch_utils import (
    launch_batch_job as launch_batch_job_impl,
)
from openai_batch_utils import (
    retrieve_batch_results as retrieve_batch_results_impl,
)
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(CURRENT_DIR.parents[2] / ".env"))

OPENAI_MODEL_NAME = "gpt-4o-2024-08-06"

TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_PROMPT_LENGTH = 7500

TRAIN_DATA_PATH = CURRENT_DIR / "qa/train_qa_le8000.json"
VAL_DATA_PATH = CURRENT_DIR / "qa/val_qa_le8000.json"
SHORTEN_BY_PERCENT = 0.25
LENGTH_THRESHOLD = 8000
TRAIN_FRACTION = 0.8
# Only used for launching the batch job
FILE_TO_UPLOAD_TRAIN = CURRENT_DIR / "qa/components/train_stories.jsonl"
FILE_TO_UPLOAD_VAL = CURRENT_DIR / "qa/components/val_stories.jsonl"

FILE_TO_UPLOAD = FILE_TO_UPLOAD_VAL

FILE_TO_DOWNLOAD = "file-3rH5MzpCf6tAYWPvLAZD3y"
RESULT_FILE_NAME = CURRENT_DIR / "qa/components/val_stories_results.jsonl"

AGENT_SYSTEM_PROMPT = """You will act as an expert text analysis agent answering a question for a reading comprehension task. You will be
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
"""

AGENT_USER_PROMPT = """<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>
"""

OPENAI_SYSTEM_PROMPT = """You are an expert text analysis agent. You will be given a {length}-word story, a question, and two possible answers.
Your tasks are as follows:
  a) Read and analyze the story carefully and then determine which of the two possible answers is correct. Write a short argument for your answer.
  b) Shorten the {length}-word story to {shortened_length}-words. IMPORTANT: When shortening the story, keep the following things in mind:
    - Keep the story as close to the original as possible.
    - Make sure that the shortened story still contains the same information as the original story.
    - Do not remove any important information from the story.
    - Do not introduce any new information into the story.
    - Do not change the meaning of the story.
    - Keep the narration style, tone, and structure of the story the same.
    - Do shorten the story to exactly {shortened_length} (+-10 words).
    - Most importantly, make sure that it's still possible to answer the question based on the shortened story.

Output format:
  - Put your argument about the correct answer in <argument></argument> tags.
  - After the argument, put either <final_answer>A</final_answer> or <final_answer>B</final_answer> to indicate which answer you think is correct.
  - Finally, put the shortened story in <shortened_story></shortened_story> tags.
"""

OPENAI_USER_PROMPT = """<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>
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
        return build_key(
            self.paragraph, self.question, self.answers[0], self.answers[1]
        )

    def build_prompt_for_agent(self, tokenizer: AutoTokenizer) -> str:
        """
        Builds the prompt for the agent based on the QADataItem.

        Returns:
            str: Formatted prompt string for the agent.
        """
        user_prompt = AGENT_USER_PROMPT.format(
            paragraph=self.paragraph,
            question=self.question,
            answer_a=self.answers[0],
            answer_b=self.answers[1],
        )
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            padding=False,
            truncation=False,
        )


def clean_text(text: str) -> str:
    """
    Remove all types of whitespace characters from the text.

    Args:
        text(str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return re.sub(r"\s+", "", text).strip()


def build_key(story: str, question: str, answer_a: str, answer_b: str) -> str:
    """
    Build a key from the story, question, and answers.

    Args:
        story(str): The story.
        question(str): The question.
        answer_a(str): The first answer.
        answer_b(str): The second answer.

    Returns:
        str: The key.
    """
    answers_sorted = sorted([answer_a, answer_b])

    # Remove all types of whitespace characters from the story because
    # the model might decode some of them wrongly
    story_cleaned = clean_text(story.strip())
    return hashlib.sha256(
        (
            story_cleaned
            + question.strip()
            + answers_sorted[0].strip()
            + answers_sorted[1].strip()
        ).encode("utf-8")
    ).hexdigest()


def load_train_and_val_data() -> Dict[str, QADataItem]:
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data_raw = json.load(f)
    train_items = [QADataItem.from_dict(item, is_train=True) for item in train_data_raw]
    items_dict = {item.id: item for item in train_items}

    with open(VAL_DATA_PATH, "r") as f:
        val_data_raw = json.load(f)
    val_items = [QADataItem.from_dict(item, is_train=False) for item in val_data_raw]

    items_dict.update({item.id: item for item in val_items})

    return items_dict


def prepare_batch_file():
    items_dict = load_train_and_val_data()

    # Combine the train and val items. Using the dict avoids any duplicates.
    items = list(items_dict.values())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def convert_to_request(
        item: QADataItem, num_words: int, shortened_length: int
    ) -> Dict[str, Any]:
        return {
            "custom_id": build_key(
                item.paragraph, item.question, item.answers[0], item.answers[1]
            ),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": OPENAI_SYSTEM_PROMPT.format(
                            length=num_words, shortened_length=shortened_length
                        ),
                    },
                    {
                        "role": "user",
                        "content": OPENAI_USER_PROMPT.format(
                            paragraph=item.paragraph,
                            question=item.question,
                            answer_a=item.answers[0],
                            answer_b=item.answers[1],
                        ),
                    },
                ],
                "max_tokens": 16_384,
                # "logprobs": True,
                # "top_logprobs": 10,
            },
        }

    # 1. For every item in 'items', build the prompt
    logging.info(f"Building prompts for {len(items)} items")
    tokenized_prompts = [item.build_prompt_for_agent(tokenizer) for item in items]

    # 2. ONLY if the tokenized prompt is longer than MAX_PROMPT_LENGTH, convert the item to a request
    logging.info("Converting items to requests")
    train_requests = []
    val_requests = []
    for tokenized_prompt, item in zip(tokenized_prompts, items):
        if len(tokenized_prompt) > MAX_PROMPT_LENGTH:
            # Compute the number of words and the number of words to shorten
            num_words = len(item.paragraph.split())
            num_words_to_shorten = int(num_words * SHORTEN_BY_PERCENT)
            shortened_length = num_words - num_words_to_shorten

            # Convert the item to a request
            if item.is_train:
                train_requests.append(
                    convert_to_request(item, num_words, shortened_length)
                )
            else:
                val_requests.append(
                    convert_to_request(item, num_words, shortened_length)
                )

    # 3. Store the requests in the train and val files
    logging.info("Storing requests in train and val files")
    with open(FILE_TO_UPLOAD_TRAIN, "w") as f:
        for request in train_requests:
            f.write(json.dumps(request) + "\n")
    with open(FILE_TO_UPLOAD_VAL, "w") as f:
        for request in val_requests:
            f.write(json.dumps(request) + "\n")


def launch_batch_job():
    launch_batch_job_impl(FILE_TO_UPLOAD)


def retrieve_batch_results():
    retrieve_batch_results_impl(FILE_TO_DOWNLOAD, RESULT_FILE_NAME)


def check_result_quality():
    items_dict = load_train_and_val_data()

    with open(RESULT_FILE_NAME, "r") as f:
        results = [json.loads(line) for line in f]

    num_correct = 0
    num_incomplete = 0
    lengths = []
    for result in results:
        id = result["custom_id"]
        response = result["response"]["body"]["choices"][0]["message"]["content"]

        # Extract the argument, the final answer, and the shortened story
        argument = re.search(r"<argument>(.*?)</argument>", response, re.DOTALL)
        final_answer = re.search(
            r"<final_answer>(.*?)</final_answer>", response, re.DOTALL
        )
        shortened_story = re.search(
            r"<shortened_story>(.*?)</shortened_story>", response, re.DOTALL
        )

        if argument is None or final_answer is None or shortened_story is None:
            num_incomplete += 1
            continue

        argument = argument.group(1)
        final_answer = final_answer.group(1)
        shortened_story = shortened_story.group(1)

        if final_answer == "A":
            final_answer_id = 0
        elif final_answer == "B":
            final_answer_id = 1
        else:
            num_incomplete += 1
            continue

        # Check if the final answer is correct
        original_item = items_dict[id]
        if final_answer_id == original_item.correct_answer_id:
            num_correct += 1

        # Build a prompt using the newly shortened story
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        original_item.paragraph = shortened_story
        tokenized_prompt = original_item.build_prompt_for_agent(tokenizer)
        lengths.append(len(tokenized_prompt))

    print(f"Number of correct: {num_correct}")
    print(f"Number of incomplete: {num_incomplete}")

    # Print a length histogram just print
    max_length = max(lengths)
    num_bins = 100
    bin_width = max_length / num_bins
    bin_edges = np.arange(0, max_length + bin_width, bin_width)
    bin_counts = np.histogram(lengths, bins=bin_edges)[0]
    for bin_count, bin_edge in zip(bin_counts, bin_edges):
        print(f"{bin_edge}: {bin_count}")


def plot_token_length_histogram():
    items_dict = load_train_and_val_data()

    # Combine the train and val items. Using the dict avoids any duplicates.
    items = list(items_dict.values())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    lengths = []
    num_items = len(items)
    threshold = 8000  # 8200
    num_smaller_than_threshold = 0
    num_larger_than_threshold = 0
    for index, item in enumerate(items):
        print(f"Processing item {index + 1} of {num_items}")
        tokenized_prompt = item.build_prompt_for_agent(tokenizer)
        if len(tokenized_prompt) <= threshold:
            num_smaller_than_threshold += 1
        else:
            num_larger_than_threshold += 1

        lengths.append(len(tokenized_prompt))

    # Use 100 fixed-width bins between 0 and 9000
    plt.hist(lengths, bins=np.linspace(0, 9000, 100))
    plt.savefig("token_length_histogram.png")

    print(
        f"Number of items smaller or equal than {threshold}: {num_smaller_than_threshold}"
    )
    print(f"Number of items larger than {threshold}: {num_larger_than_threshold}")


def filter_items_by_length():
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data_raw = json.load(f)
    train_items = [QADataItem.from_dict(item, is_train=True) for item in train_data_raw]

    with open(VAL_DATA_PATH, "r") as f:
        val_data_raw = json.load(f)
    val_items = [QADataItem.from_dict(item, is_train=False) for item in val_data_raw]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train_items_filtered = []
    val_items_filtered = []
    for raw_items, items, outputs in zip(
        [train_data_raw, val_data_raw],
        [train_items, val_items],
        [train_items_filtered, val_items_filtered],
    ):
        for index, item in enumerate(items):
            print(f"Processing item {index + 1} of {len(items)}")
            tokenized_prompt = item.build_prompt_for_agent(tokenizer)
            if len(tokenized_prompt) <= LENGTH_THRESHOLD:
                outputs.append(raw_items[index])

    # Instead of hard-coding the path, automatically extract the filename from the TRAIN_DATA_PATH
    # The with_suffix() method replaces the entire suffix, so we need to modify the stem instead
    train_outputs_path = TRAIN_DATA_PATH.with_name(
        f"{TRAIN_DATA_PATH.stem}_le{LENGTH_THRESHOLD}.json"
    )
    val_outputs_path = VAL_DATA_PATH.with_name(
        f"{VAL_DATA_PATH.stem}_le{LENGTH_THRESHOLD}.json"
    )

    with open(train_outputs_path, "w") as f:
        json.dump(train_items_filtered, f, indent=4)
    with open(val_outputs_path, "w") as f:
        json.dump(val_items_filtered, f, indent=4)

    print(
        f"Reduced the number of train items from {len(train_items)} to {len(train_items_filtered)}"
    )
    print(
        f"Reduced the number of val items from {len(val_items)} to {len(val_items_filtered)}"
    )


def correct_train_test_split():
    """
    Load both the train- and val-set. If the train-fraction is lower than 'TRAIN_FRACTION',
    sample an appropriate number of items from the val-set to make the split correct.
    """
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data_raw = json.load(f)
    train_items = [QADataItem.from_dict(item, is_train=True) for item in train_data_raw]
    train_items_dict = {item.id: item for item in train_items}

    with open(VAL_DATA_PATH, "r") as f:
        val_data_raw = json.load(f)
    val_items = [QADataItem.from_dict(item, is_train=False) for item in val_data_raw]
    val_items_dict = {
        item.id: item for item in val_items if item.id not in train_items_dict
    }

    if (
        len(train_items_dict) / (len(train_items_dict) + len(val_items_dict))
        < TRAIN_FRACTION
    ):
        # Sample an appropriate number of items from the val-set and add them to the train-set
        # to make the split correct
        num_items_to_sample = int(
            TRAIN_FRACTION * (len(train_items_dict) + len(val_items_dict))
            - len(train_items_dict)
        )
        selected_items = random.sample(
            list(val_items_dict.items()), num_items_to_sample
        )
        train_items_dict.update({key: value for (key, value) in selected_items})
        val_items_dict = {
            item.id: item for item in val_items if item.id not in train_items_dict
        }

    train_items_new = list(train_data_raw)
    val_items_new = []
    for item in val_data_raw:
        item_structured = QADataItem.from_dict(item, is_train=True)
        if item_structured.id in train_items_dict:
            train_items_new.append(item)
        else:
            val_items_new.append(item)

    # Instead of hard-coding the path, automatically extract the filename from the TRAIN_DATA_PATH
    # The with_suffix() method replaces the entire suffix, so we need to modify the stem instead
    train_outputs_path = TRAIN_DATA_PATH.with_name(
        f"{TRAIN_DATA_PATH.stem}_balanced.json"
    )
    val_outputs_path = VAL_DATA_PATH.with_name(f"{VAL_DATA_PATH.stem}_balanced.json")

    with open(train_outputs_path, "w") as f:
        json.dump(train_items_new, f, indent=4)
    with open(val_outputs_path, "w") as f:
        json.dump(val_items_new, f, indent=4)

    print(
        f"Reduced the number of train items from {len(train_data_raw)} to {len(train_items_new)}"
    )
    print(
        f"Reduced the number of val items from {len(val_data_raw)} to {len(val_items_new)}"
    )


ACTION_MAP = {
    "prepare_batch_file": prepare_batch_file,
    "launch_batch_job": launch_batch_job,
    "retrieve_batch_results": retrieve_batch_results,
    "check_result_quality": check_result_quality,
    "plot_token_length_histogram": plot_token_length_histogram,
    "filter_items_by_length": filter_items_by_length,
    "correct_train_test_split": correct_train_test_split,
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
