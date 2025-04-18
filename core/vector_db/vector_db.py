from typing import Literal
import os
from langchain_community.vectorstores import FAISS, Chroma


class VectorDB:
    def __init__(self,
                 db_type: Literal["faiss", "chroma"],
                 persist_path: str = ""):
        """
        Initialize the vector database wrapper.

        Args:
            db_type: "faiss" or "chroma" - which vector database to use
            persist_path: path to save index DB
        """
        self.db_type = db_type.lower()
        self.persist_path = persist_path
        self._vector_store = None

        if not os.path.exists(self.persist_path):
            self.persist_path = os.path.join("..", self.persist_path)

    def from_documents(self, documents, embedding):
        """
        Create a vector store from documents using the provided embedding.

        Args:
            documents: A list of documents to index
            embedding: An embedding instance to use for vectorization

        Returns:
            self (for method chaining)
        """
        if self.db_type == "faiss":
            self._create_faiss_store(documents, embedding)
        elif self.db_type == "chroma":
            self._create_chroma_store(documents, embedding)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}. Use 'faiss' or 'chroma'.")

        return self

    def _create_faiss_store(self, documents, embedding):
        """Create a FAISS vector store."""

        self._vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding,
        )

        self._vector_store.save_local(self.persist_path)

    def _create_chroma_store(self, documents, embedding):
        """Create a Chroma vector store."""

        def filter_metadata(documents):
            for doc in documents:
                for key, value in doc.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        doc.metadata[key] = value[0]
                    elif isinstance(value, list):
                        doc.metadata[key] = str(value)
            return documents

        self._vector_store = Chroma.from_documents(
            documents=filter_metadata(documents),
            embedding=embedding,
            persist_directory=os.path.join("../", self.persist_path)
        )

        self._vector_store.persist()

    @property
    def vector_store(self):
        """Return the vector_store property."""
        return self._vector_store

    def load_db(self, embedding):
        if self.db_type == "faiss":
            self._vector_store = FAISS.load_local(os.path.join("../", self.persist_path),
                                                  embedding,
                                                  allow_dangerous_deserialization=True)
        elif self.db_type == "chroma":
            self._vector_store = Chroma(
                persist_directory=self.persist_path,
                embedding_function=embedding
            )
