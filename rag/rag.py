import os
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import load_config
from llm.embedding.openai_embedding import OpenAIEmbeddingClient
from langchain_community.vectorstores import FAISS
from typing import List, Tuple, Optional, Any, Dict

from llm.llm_helper import LLMChat
from query.query_helper import QueryHelper
from langchain.schema import Document


class RAG:
    """
   Retrieval Augmented Generation class for document indexing and similarity search.

   This class handles:
   - Document loading and chunking
   - Vector database creation and management
   - Document summarization at multiple levels
   - Similarity search for queries
   """

    def __init__(
            self,
            env_config: Dict[str, Any] = None,
            llm_config: Dict[str, Any] = None,
            embedding_model: str = "text-embedding-ada-002",
            top_k: int = 10,
            persist_path: str = "../persist.db",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
    ):

        """
        Initialize the RAG system.

        Args:
            env_config: Environment configuration for LLM
            llm_config: Configuration for LLM
            embedding_model: Model name for embeddings
            top_k: Number of results to return in queries
            persist_path: Path to save/load the vector database
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
        """

        # Set default dicts if None
        self.env_config = env_config or {}
        self.llm_config = llm_config or {}

        # Initialize embedding client
        self.embedding_model = embedding_model
        self.embedding = OpenAIEmbeddingClient(self.embedding_model)

        # Set parameters
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

        # Resolve persist path
        self.persist_path = persist_path
        self._resolve_persist_path()

        # Initialize LLM components
        self.query_llm = LLMChat(**self.llm_config)
        self.query_helper = QueryHelper(self.query_llm)

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

            doc_summary = Document(page_content=doc_summary, **{"metadata": doc[0].metadata})
            split_documents.extend([doc_summary])

        # Add root general summary to deal with general questions
        root_summary = self.query_helper.summarize_text("".join(summary_list))
        doc_root_summary = Document(page_content=root_summary, **{"metadata": doc_list[0][0].metadata})
        split_documents.extend([doc_root_summary])
        return split_documents

    def _create_new_db(self, split_documents: List[Any]) -> None:
        """
        Create a new vector database from documents.

        Args:
            split_documents: List of document chunks to index
        """
        self.vector_store = FAISS.from_documents(split_documents,
                                                 self.embedding.embeddings)
        self.vector_store.save_local(self.persist_path)

    def _load_db(self) -> None:
        """Load the vector database from disk."""

        self.vector_store = FAISS.load_local(self.persist_path,
                                             self.embedding.embeddings,
                                             allow_dangerous_deserialization=True)

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

    def query_similarity(self, query_text: str) -> Optional[List[Tuple[Any, float]]]:
        """
        Query the vector database for similar documents.

        Args:
            query_text: The query text

        Returns:
            List of (document, similarity_score) tuples or None if database isn't ready
        """
        results = None
        if self.vector_store:
            results = self.vector_store.similarity_search_with_score(query_text, k=self.top_k)
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


# Need to consider how to delete old data

if __name__ == "__main__":
    config = load_config()
    rag_config = config["rag_config"]
    env_config = config["env_config"]
    llm_config = config["llm_config"]

    rag = RAG(env_config, llm_config, **rag_config)
    rag.index_documents(r".\data_files", True)
    results = rag.query_similarity("When does the transformation started?")
    for result in results:
        print(result)
