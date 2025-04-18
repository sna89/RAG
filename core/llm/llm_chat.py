from typing import Dict, List, Tuple, Any
from core.llm.utils import count_tokens

MAX_TOKENS = {"gpt-4o-mini": 16384}
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class LLMChat:
    """
      A class for managing chat interactions with OpenAI models.
      This class handles conversation history, token counting, and API interactions.
    """

    def __init__(
            self,
            model_name,
            llm
    ):
        """
        Initialize the LLMChat client.

        Args:
            model_name: The deployment name for the chat model
        """

        self.model_name = model_name
        self._llm = llm

        self.conversation_history = []
        self.initialize_history()

    def initialize_history(self):
        """Reset the conversation history to only include the system prompt."""

        self.conversation_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]

    def query(
            self,
            prompt: str,
            user_query: str = ""
    ) -> Tuple[Any, List[Dict[str, str]]]:
        """
        Send a query to the LLM and manage conversation history.

        Args:
            prompt: The prompt to send to the model
            user_query: Optional user message to store in history
                        (useful when prompt contains context not meant for future queries)

        Returns:
            Tuple containing (response object, updated conversation history)
        """

        self.conversation_history.append(
            {"role": "user", "content": prompt}
        )

        num_tokens = count_tokens(self.conversation_history, self.model_name)

        if num_tokens >= MAX_TOKENS[self.model_name]:
            response_message = "Exceeded number of max tokens, " \
                               "please provide shorter prompt or start a new conversation"
            self.conversation_history.pop()
            return response_message, self.conversation_history

        response = self.llm.invoke(self.conversation_history)

        self.conversation_history.pop()
        # Store only the user_query.
        # This optimization prevents storing large context windows repeatedly:
        # - The full prompt with context is used only for the current query.
        # - Only the essential user query is preserved in conversation history.

        self.conversation_history.append(
            {"role": "user", "content": str(user_query)}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response.content}
        )

        return response, self.conversation_history

    @property
    def llm(self):
        return self._llm


if __name__ == "__main__":
    pass
    # load_dotenv()
    #
    # endpoint = os.getenv("AZURE_ENDPOINT")
    # api_version = os.getenv("AZURE_API_VERSION")
    # openai_model_name = os.getenv("LLM_MODEL")
    #
    # llm = AzureLLMChat(endpoint, api_version, openai_model_name)
    # # response = llm.query(prompt="My favorite color is Green")
    # # print(response.content)
    # #
    # # # Check memory capabilities
    # # response = llm.query(prompt="What is my favorite color?")
    # # print(response.content)
