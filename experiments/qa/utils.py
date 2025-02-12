import hashlib
import re


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


if __name__ == "__main__":
    story = "The quick brown fox jumps over the lazy dog. The dog barks at the fox."
    argument = "<quote>brown fox jumps  over</quote>. Hence I would also like <quote> to add that <quote>dog barks at the cat.</quote> <quote>"
    print(verify_quotes(story, argument))
