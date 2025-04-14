from typing import List, Union

from langchain_openai import OpenAIEmbeddings
from core.embedding.embedding_client import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """
        Initialize the OpenAI embedding client.

        Args:
            model_name: The name of the OpenAI embedding model to use
        """
        super().__init__()
        self._embeddings = OpenAIEmbeddings(
            model=model_name,
        )

    @property
    def embeddings(self):
        """Return the OpenAI embeddings instance."""
        return self._embeddings

    def embed_text(self, text: Union[str, List[str]]) -> List[float]:
        """
        Embed text using OpenAI.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        return self.embeddings.embed_query(text)
