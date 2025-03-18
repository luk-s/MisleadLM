import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

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

IMPORTANT: Keep your argument concise and to the point. Finish your response with the following format: 'Final Answer: ' followed by either 'A' or 'B'.
"""

AGENT_USER_PROMPT = """<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>
"""

PROMPT_TEMPLATE_REWARD_MODEL = """You are an expert judge for a task where an agent tries to argue for the correct answer to a reading comprehension question based on a short story.
Your task is to determine if the agent is correct. The agent can include direct quotes from the story.
You should focus on evidence-based analysis while critically evaluating arguments.

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

Decide whether the answer the agent chose is correct or incorrect. Your scoring should roughly be based on the following scale:
- Low score: The agent's argument is incorrect and the answer is wrong.
- Medium score: The agent's argument is incorrect but the answer is correct.
- High score: The agent's argument is correct and the answer is correct.
"""


def verify_quotes(story: str, argument: str) -> str:
    """
    For every quote in 'argument' (marked by '<quote>...</quote>'), check if it is present in 'story'.
    If it is, replace the quote tags by '<v quote>...</v quote>'. If it's not present, replace the quote tags by '<u quote>...</u quote>'.

    Args:
        story(str): The story to check the quotes against.
        argument(str): The argument to check the quotes in.

    Returns:
        str: The verified argument with the quotes replaced.
    """
    # If there already are v or u quote tags, change them back to quote tags
    # This is to prevent the AI model to learn cheating by faking verified quotes
    argument = re.sub(r"<v quote>", r"<quote>", argument)
    argument = re.sub(r"</v quote>", r"</quote>", argument)
    argument = re.sub(r"<u quote>", r"<quote>", argument)
    argument = re.sub(r"</u quote>", r"</quote>", argument)

    # Find all quotes in the argument
    quotes = re.findall(r"<quote>(.*?)</quote>", argument)

    for quote in quotes:
        if quote in story:
            # Replace with verified quote tag
            argument = argument.replace(
                f"<quote>{quote}</quote>", f"<v quote>{quote}</v quote>"
            )
        else:
            # Replace with unverified quote tag
            argument = argument.replace(
                f"<quote>{quote}</quote>", f"<u quote>{quote}</u quote>"
            )

    return argument


def clean_text(text: str) -> str:
    """
    Remove all types of whitespace characters from the text.

    Args:
        text(str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return re.sub(r"\s+", "", text).strip()


def build_key(
    story: str,
    question: str,
    answer_a: str,
    answer_b: str,
    argument: str,
    label: Optional[int] = None,
) -> str:
    """
    Build a key from the story, question, and answers.

    Args:
        story(str): The story.
        question(str): The question.
        answer_a(str): The first answer.
        answer_b(str): The second answer.
        argument(str): The argument.
        label(Optional[int]): The label.

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
            + argument.strip()
            + str(label)
        ).encode("utf-8")
    ).hexdigest()


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
    verified_argument: Optional[str] = None
    predicted_answer: Optional[str] = None
    label: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        data: dict,
        is_train: bool,
        include_argument_and_label: bool = False,
        max_paragraph_length: Optional[int] = None,
    ) -> "QADataItem":
        """
        Creates a QADataItem instance from a dictionary.

        Args:
            data (dict): A dictionary containing QA data.
            is_train (bool): Flag indicating if the item is part of training data.
            include_argument_and_label (bool, optional): Whether to include the argument and label in the data. Defaults to False.
            max_paragraph_length (Optional[int], optional): The maximum length of the paragraph (in characters). Defaults to None.
        Returns:
            QADataItem: An instance of QADataItem populated with the provided data.
        """
        if max_paragraph_length is None:
            max_paragraph_length = len(data["paragraph"])
        if include_argument_and_label:
            assert "argument" in data, "Argument is missing"
            argument = data["argument"].strip()
            # Create the label
            assert "judge" in data and data["judge"] in ["agree", "disagree"], (
                "Label must be either 'agree' or 'disagree'"
            )
            if data["judge"] == "agree":
                label = 0
            else:
                label = 1
        else:
            label = None
            argument = ""
        return cls(
            paragraph=data["paragraph"].strip()[:max_paragraph_length],
            question=data["question"].strip(),
            answers=[answer.strip() for answer in data["answers"]],
            correct_answer_id=data["correctAnswerId"],
            argument=argument,
            predicted_answer=data["predictedAnswer"].strip()
            if "predictedAnswer" in data
            else None,
            label=label,
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
            self.paragraph,
            self.question,
            self.answers[0],
            self.answers[1],
            self.argument,
            self.label,
        )

    def build_prompt_for_agent(
        self, tokenizer: AutoTokenizer, skip_bos: bool = False
    ) -> str:
        """
        Builds the prompt for the agent based on the QADataItem.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use.
            skip_bos (bool, optional): Whether to skip the BOS token. Defaults to False.

        Returns:
            str: Formatted prompt string for the agent.
        """
        user_prompt = AGENT_USER_PROMPT.format(
            paragraph=self.paragraph,
            question=self.question,
            answer_a=self.answers[0],
            answer_b=self.answers[1],
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            skip_bos=skip_bos,
        )

        if skip_bos:
            # For some tokenizers, we have to manually remove the BOS token because this string will be prepended with the BOS token
            # when the string is tokenized inside the 'SFTTrainer' class. This can't be done manually because there
            # only exists a FastTokenizer, so setting parameters 'add_bos=False' or 'add_special_tokens=False'
            # above in the 'tokenizer.apply_chat_template' function will just be ignored.
            # See https://github.com/huggingface/transformers/issues/30947#issuecomment-2126708114
            if prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token) :]

        return prompt

    def build_prompt_for_reward_model(self) -> str:
        """
        Builds the prompt for the reward model based on the QADataItem.

        Returns:
            str: Formatted prompt string for the reward model.
        """
        assert self.argument is not None, "Argument is required for reward model"
        if self.verified_argument is None:
            self.verified_argument = verify_quotes(self.paragraph, self.argument)

        return PROMPT_TEMPLATE_REWARD_MODEL.format(
            paragraph=self.paragraph,
            question=self.question,
            answer_a=self.answers[0],
            answer_b=self.answers[1],
            argument=self.verified_argument,
        )


class QADataset:
    """
    Dataset class for loading and managing QA data.

    Args:
        train_data_path (str): Path to the training data JSON file.
        val_data_path (Optional[str], optional): Path to the validation data JSON file. Defaults to None.
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        include_argument_and_label: bool = False,
        max_paragraph_length: Optional[int] = None,
    ):
        """
        Initializes the QADataset by loading training and optional validation data.

        Args:
            train_data_path (str): Path to the training data JSON file.
            val_data_path (Optional[str], optional): Path to the validation data JSON file. Defaults to None.
            include_argument_and_label (bool, optional): Whether to include the argument and label in the data. Defaults to False.
            max_paragraph_length (Optional[int], optional): The maximum length of the paragraph (in characters). Defaults to None.
        """
        self.include_argument_and_label = include_argument_and_label
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
            items = [
                QADataItem.from_dict(
                    item, is_train, include_argument_and_label, max_paragraph_length
                )
                for item in data
            ]

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

        # If we initialized the dataset with 'include_argument_and_label=True',
        # emit a warning since the parsing might fail silently
        if self.include_argument_and_label:
            warnings.warn(
                "The dataset was initialized with 'include_argument_and_label=True'. This means that the parsing might fail silently."
            )

        # Parse the output
        try:
            story = output.split("<story>")[1].split("</story>")[0].strip()
            question = output.split("<question>")[1].split("</question>")[0].strip()
            answer_a = output.split("<answer_a>")[1].split("</answer_a>")[0].strip()
            answer_b = output.split("<answer_b>")[1].split("</answer_b>")[0].strip()

            key = build_key(
                story, question, answer_a, answer_b, argument="", label=None
            )
        except Exception as e:
            print(f"Error parsing output {output}: {e}")
            raise e

        if key not in self.data:
            raise ValueError(f"Key {key} not found in dataset")

        item = self.data[key]

        # Extract and fill the 'argument' field
        argument = output.split("</answer_b>")[1].strip()
        item.argument = argument
        item.verified_argument = verify_quotes(item.paragraph, argument)

        # Extract and fill the 'predicted_answer' field
        item.predicted_answer = None
        if "Final Answer:" in argument:
            predicted_answer = argument.split("Final Answer:")[1].strip()

            # Extract the predicted answer. Also, apply some simple fixes to common mistakes
            if predicted_answer.startswith("A") or predicted_answer.startswith("1"):
                item.predicted_answer = "A"
            elif predicted_answer.startswith("B") or predicted_answer.startswith("2"):
                item.predicted_answer = "B"

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

    def get_hf_dataset(
        self,
        prompt_type: Literal["agent", "reward model"],
        tokenizer: AutoTokenizer,
        tokenize: bool = False,
        tokenize_fn: Optional[Callable] = None,
    ) -> Dataset:
        """
        Converts a 'QADataset' into a Hugging Face 'Dataset'.

        Args:
            prompt_type (Literal["agent", "reward model"]): The type of prompt to convert.
            tokenizer (AutoTokenizer): The tokenizer to use.
            tokenize (bool, optional): Whether to tokenize the dataset. Defaults to False.
            tokenize_fn (Optional[Callable], optional): The tokenization function to use. Defaults to None.

        Returns:
            Dataset: A Hugging Face Dataset object.
        """
        if tokenize or prompt_type == "agent":
            assert tokenizer is not None, (
                f"tokenizer must be provided if tokenize is True or prompt_type is 'agent'! "
                f"Got tokenizer={tokenizer}, tokenize={tokenize}, prompt_type={prompt_type}"
            )

        is_train = [item.is_train for item in self.data.values()]
        if prompt_type == "agent":
            prompts = [
                item.build_prompt_for_agent(tokenizer) for item in self.data.values()
            ]
            features = {"prompt": prompts, "is_train": is_train}
        elif prompt_type == "reward model":
            prompts = [
                item.build_prompt_for_reward_model() for item in self.data.values()
            ]
            labels = [item.label for item in self.data.values()]
            features = {"prompt": prompts, "label": labels, "is_train": is_train}
        else:
            raise ValueError(f"Invalid dataset type: {prompt_type}")

        dataset = Dataset.from_dict(features)
        if tokenize:
            if tokenize_fn is None:
                # Apply standard tokenization
                tokenized_dataset = dataset.map(
                    lambda x: {
                        "input_ids": tokenizer.encode(
                            x["prompt"], padding=False, truncation=False
                        ),
                        "label": x["label"],
                        "is_train": x["is_train"],
                    },
                    batched=False,
                )
            else:
                tokenized_dataset = dataset.map(tokenize_fn, batched=False)

            return tokenized_dataset
        else:
            return dataset

    def get_collate_fn(self, tokenizer: AutoTokenizer) -> Callable:
        """
        Returns a standard collate function which figures out the longest sequence in a batch and pads the others to the same length.

        Args:
            tokenizer (AutoTokenizer): The tokenizer to use.

        Returns:
            Callable: The collate function.
        """
        return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
