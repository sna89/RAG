import tiktoken
from typing import List, Dict, Union


def count_tokens(messages: Union[List[Dict[str, str]], str], model_name: str = 'gpt-35-turbo') -> int:
    """
        Count tokens for a given model and text.

        Args:
            messages: Either a list of message dictionaries or a string
            model_name: The model name to use for token counting
        Returns:
            Number of tokens
    """

    encoding = tiktoken.encoding_for_model(model_name)
    if isinstance(messages, list):
        text = " ".join([item["content"] for item in messages])
    elif isinstance(messages, str):
        text = messages
    else:
        return 0
    num_tokens = len(encoding.encode(text))
    return num_tokens
