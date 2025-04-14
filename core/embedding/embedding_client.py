from abc import ABC, abstractmethod
from typing import List, Any



class EmbeddingClient(ABC):
    """
    Abstract base class for embedding clients.
    Defines the common interface for different embedding model implementations.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def embeddings(self) -> Any:
        """
        Property that must be implemented by subclasses.
        Ensures all implementations have an embeddings attribute.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        pass