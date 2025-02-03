import json
import re
from pathlib import Path
from typing import Callable, List

from datasets import load_dataset


def set_year_to_integer(line: str, **kwargs) -> str:

    # Regular expression to find "year": "XXXX" and replace it with "year": XXXX
    pattern = r'"year": "(\d{4})"'
    replacement = r'"year": \1'

    return re.sub(pattern, replacement, line)


def add_missing_keys(line: str, required_keys: List[str], **kwargs) -> str:
    # Load the JSON object
    obj = json.loads(line)

    # Add missing keys
    for key in required_keys:
        if key not in obj:
            obj[key] = None

    return json.dumps(obj)


CLEANING_FUNCTIONS: List[Callable] = [set_year_to_integer]


def clean_dataset(data_file_path: str) -> None:
    print(f"\nStarting cleaning {data_file_path}...")
    data_file_path = Path(data_file_path)

    # Read the dataset
    with open(data_file_path, "r") as file:
        lines = file.readlines()

    # Create some helper variables
    required_keys = json.loads(lines[0]).keys()

    kwargs = {
        "required_keys": required_keys,
    }

    # Clean the dataset
    for line_number, line in enumerate(lines):
        for cleaning_function in CLEANING_FUNCTIONS:
            lines[line_number] = cleaning_function(line, **kwargs)

    # Write the cleaned dataset
    # Add a '_cleaned' suffix to the file name but before the file extension
    data_file_path = data_file_path.with_name(f"{data_file_path.stem}_cleaned{data_file_path.suffix}")
    with open(str(data_file_path), "w") as file:
        file.writelines(lines)

    # Check if the dataset can be loaded
    try:
        load_dataset("json", data_files=str(data_file_path))
        print(f"Successfully cleaned {data_file_path}!")
    except Exception as e:
        print(f"Failed to clean {data_file_path}!")
        print(e)


if __name__ == "__main__":
    data_file_paths = [
        "QuALITY.v1.0.1.train",
        "QuALITY.v1.0.1.dev",
        "QuALITY.v1.0.1.test",
        "QuALITY.v1.0.1.htmlstripped.train",
        "QuALITY.v1.0.1.htmlstripped.dev",
        "QuALITY.v1.0.1.htmlstripped.test",
    ]

    for data_file_path in data_file_paths:
        clean_dataset(data_file_path)
