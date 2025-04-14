from typing import List

from langchain_openai import OpenAIEmbeddings
from typing_extensions import Union


class OpenAIEmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
        )

    def embed_text(self, text: Union[str, List[str]]):
        return self.embeddings.embed_query(text)
