from langchain_community.embeddings import HuggingFaceEmbeddings


class HuggingFaceEmbeddingClient:
    """
        Service for generating text embeddings using HuggingFace models.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-l6-v2"):
        """
        Initialize the embedding service with a specific model.

        Args:
            model_name (str): Name of the HuggingFace model to use for embeddings.
        """

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_text(self, text):
        """
        Generate an embedding for a given text.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        return self.embeddings.embed_query(text)


if __name__ == "__main__":
    embedding_client = HuggingFaceEmbeddingClient()
    embedding = embedding_client.embed_text("The birth of automobile manufacturing stands as a fascinating "
                                            "transitional "
                                            "phase between craft production traditions and the emerging industrial "
                                            "paradigms of the 20th century.")
    print(embedding)
