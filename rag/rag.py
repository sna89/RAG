import os

import openai
from langchain_openai import ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import load_config
from core.embedding.openai_embedding import OpenAIEmbeddingClient
from core.embedding.embedding_client import EmbeddingClient
from typing import List, Tuple, Optional, Any, Union

from core.llm.llm_chat import LLMChat
from core.vector_db.vector_db import VectorDB
from query.query_helper import QueryHelper
from langchain.schema import Document

from rag.utils import get_unique_union


class RAG:
    def __init__(
            self,
            query_helper: QueryHelper,
            embedding_client: EmbeddingClient = EmbeddingClient,
            vector_db: Any = None,
            top_k: int = 10,
            persist_path: str = "persist.db",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
    ):

        """
        Initialize the RAG system.

        Args:
            query_helper: query_helper for applying llm queries
            embedding_client: Embedding client for transforming text to vector representation
            vector_db: index vector_db for storing and retrieving embeddings
            top_k: Number of results to return in queries
            persist_path: Path to save/load the vector database
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """

        self.query_helper = query_helper

        # Initialize embedding client
        self.embedding_client = embedding_client

        # initialize vector db
        self.vector_db = vector_db

        # Set parameters
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

        # Resolve persist path
        self.persist_path = persist_path
        self._resolve_persist_path()

        # Load existing database if it exists
        if os.path.exists(self.persist_path):
            self._load_db()

    def _resolve_persist_path(self):
        """
        Resolve the full path for vector store persistence.

        Args:
            persist_path: Relative or absolute path for persistence

        Returns:
            Resolved absolute path
        """
        if not os.path.exists(self.persist_path):
            self.persist_path = os.path.join("\\".join(os.getcwd().split("\\")[:-1]), self.persist_path)

    def index_documents(self, path: str, override_db: bool = False) -> bool:
        """
        Index documents from a directory.

        Args:
            path: Directory path containing documents to index
            override_db: Whether to override existing database

        Returns:
            True if indexing was successful, False otherwise
        """

        if not os.path.exists(path):
            return False

        if not os.path.exists(self.persist_path) or override_db:
            doc_list = self._load_documents(path)
            if not doc_list:
                return False

            split_documents = self._apply_chunking(doc_list)
            self._create_new_db(split_documents)
            return True
        else:
            return False

    @staticmethod
    def _load_documents(path: str) -> List[Any]:
        """
        Load documents from a directory.

        Args:
            path: Directory path containing documents

        Returns:
            List of loaded documents
        """
        doc_list = []
        doc_name_list = os.listdir(path)
        for doc_name in doc_name_list:
            doc_path = os.path.join(path, doc_name)
            if os.path.isfile(doc_path):
                loader = UnstructuredLoader(doc_path)
                loaded_docs = loader.load()
                doc_list.append(loaded_docs)

        return doc_list

    def _apply_chunking(self, doc_list: List[Any]) -> List[Any]:
        """
        Split documents into chunks.

        Args:
            doc_list: List of documents to split

        Returns:
            List of document chunks
        """
        rec_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_documents = []
        summary_list = []

        for doc in doc_list:
            split_document = rec_text_splitter.split_documents(doc)
            split_documents.extend(split_document)

            # Summarize document to deal with long document questions
            doc_summary = self.query_helper.summarize_text(doc[0].page_content)
            summary_list.append(doc_summary)

            doc_summary = Document(page_content=doc_summary, metadata={"filename": doc[0].metadata["filename"]})
            split_documents.extend([doc_summary])

        # Add root general summary to deal with general questions
        root_summary = self.query_helper.summarize_text("".join(summary_list))
        doc_root_summary = Document(page_content=root_summary,
                                    metadata={"filename": doc_list[0][0].metadata["filename"]})
        split_documents.extend([doc_root_summary])
        return split_documents

    def _create_new_db(self, split_documents: List[Any]) -> None:
        """
        Create a new vector database from documents.

        Args:
            split_documents: List of document chunks to index
        """
        self.vector_db.from_documents(split_documents, self.embedding_client.embeddings)
        self.vector_store = self.vector_db.vector_store

    def _load_db(self) -> None:
        """Load the vector database from disk."""

        self.vector_db.load_db(self.embedding_client.embeddings)
        self.vector_store = self.vector_db.vector_store

    def add_documents(self, path: str) -> bool:
        """
       Add documents to an existing database.

       Args:
           path: Directory path containing documents to add

       Returns:
           True if documents were added successfully, False otherwise
        """
        if self.vector_store:
            doc_list = self._load_documents(path)
            split_documents = self._apply_chunking(doc_list)
            self.vector_store.add_documents(split_documents)
            self.vector_store.save_local(self.persist_path)
            return True
        else:
            print("Need to create index DB before adding data.")
            return False

    def query_similarity(self, query_text: Union[str, List[str]]) -> Optional[List[Tuple[Any, float]]]:
        """
        Query the vector database for similar documents.

        Args:
            query_text: The query text

        Returns:
            List of (document, similarity_score) tuples or None if database isn't ready
        """
        results = []
        if self.vector_store:

            if isinstance(query_text, str):
                results = self.vector_store.similarity_search_with_score(query_text, k=self.top_k)
                results = [result[0].page_content for result in results]

            elif isinstance(query_text, List):
                for query in query_text:
                    results.extend(self.vector_store.similarity_search_with_score(query, k=self.top_k))

                results = get_unique_union(results)
                results = [result.page_content for result in results]

        else:
            print("Error applying query similarity. Need to create index.")

        return results

    def update_top_k(self, top_k: int) -> None:
        """
        Update the number of results to return.

        Args:
            top_k: New number of results to return
        """
        if top_k > 0:
            self.top_k = top_k


def initialize_rag_components(config):
    # Initialize LLM components
    llm_config = config["core_config"]["llm_config"]
    query_llm = LLMChat(llm_config["model_name"], ChatOpenAI(**llm_config))
    query_helper = QueryHelper(query_llm)

    embedding_client = None
    embedding_config = config["core_config"]["embedding_model"]
    if embedding_config["provider"] == "openai":
        embedding_client = OpenAIEmbeddingClient(embedding_config["model_name"])

    vector_db_conf = config["core_config"]["vector_db"]
    vector_db = VectorDB(**vector_db_conf)
    vector_db.load_db(embedding_client.embeddings)
    return query_helper, embedding_client, vector_db


if __name__ == "__main__":
    config = load_config()

    openai.api_key = config["env_config"]["openai_api_key"]
    query_helper, embedding, vector_db = initialize_rag_components(config)

    rag = RAG(query_helper,
              embedding,
              vector_db,
              **config["rag_config"])

    rag.index_documents(r".\data_files", True)
    results = rag.query_similarity("When does the transformation started?")
    for result in results:
        print(result)
