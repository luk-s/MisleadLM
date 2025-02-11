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


if __name__ == "__main__":
    story = "The quick brown fox jumps over the lazy dog. The dog barks at the fox."
    argument = "<quote>brown fox jumps  over</quote>. Hence I would also like <quote> to add that <quote>dog barks at the cat.</quote> <quote>"
    print(verify_quotes(story, argument))
