import json
import time
from pathlib import Path
from typing import Dict, Iterator

import ijson  # For iterative JSON parsing
from dotenv import load_dotenv
from openai import OpenAI

CURRENT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=str(CURRENT_DIR.parents[2] / ".env"))

SYSTEM_PROMPT = """
You will be given a long paragraph of text together with a question. Please create a ~200 words shorter version of the paragraph.
Make sure that there is sufficient information in the shorter version, such that the question can still be answered.
Don't directly refer to the question in the shorter version. Lastly, write the shorter version from the same perspective as the original paragraph.
"""

USER_PROMPT = """
QUESTION: {question}

PARAGRAPH:\n
{paragraph}
"""


def load_json_iteratively(filename: str) -> Iterator[Dict]:
    """Load JSON file iteratively to handle large files."""
    with open(filename, "rb") as file:
        parser = ijson.items(file, "item")
        for item in parser:
            yield item


def get_summary(item: Dict[str, str], client: OpenAI, model_name: str) -> str:
    """Get summary from OpenAI API with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(**item)},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2**attempt)  # Exponential backoff


def process_file(data_file_name: str, data_output_file_name: str, model_name: str):
    # Initialize OpenAI client
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment variables

    # Process data and store results
    processed_data = []

    try:
        for index, item in enumerate(load_json_iteratively(data_file_name)):

            # Get summary from OpenAI
            summary = get_summary(item, client, model_name)

            # Add summary to the dictionary
            item["summary"] = summary

            # Store processed item
            with open(data_output_file_name, "a", encoding="utf-8") as f:
                json.dump(item, f, indent=2, ensure_ascii=False)
                f.write("\n")

            # Optional: Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

            print(f"Processed item: {index + 1}")

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    # Get the path to the directory of this file:
    data_file_name = CURRENT_DIR / "val.json"
    data_output_file_name = CURRENT_DIR / "val_extended.jsonl"
    model_name = "gpt-4o-mini-2024-07-18"
    process_file(data_file_name, data_output_file_name, model_name)
