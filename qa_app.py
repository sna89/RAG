import openai
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any, Union, Tuple, List

from core.llm.llm_chat import LLMChat
from core.rank import Ranker
from query.prompts import RAG_PROMPT
from query.query_helper import QueryHelper
from query.query_translator import QueryTranslator
from rag.rag import RAG, initialize_rag_components
from rag.utils import filter_retrieval_result
from config import load_config


class QAApp:
    """
    Main Question-Answering application that coordinates various NLP components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the QA application with the provided configuration.

        Args:
            config: Configuration dictionary with required settings
        """
        self.config = config
        self.qa_config = config.get("qa_config", {})

        # Set API key
        openai.api_key = config["env_config"].get("openai_api_key")
        if not openai.api_key:
            raise ValueError("OpenAI API key is missing in configuration")

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components for the QA system."""
        # Initialize RAG components
        self.query_helper, self.embedding_client, self.vector_db = initialize_rag_components(self.config)
        self.vector_store = self.vector_db.vector_store

        # Create LLM instances
        llm_config = self.config["core_config"].get("llm_config", {})
        if not llm_config:
            raise ValueError("LLM configuration is missing")

        model_name = llm_config.get("model_name")
        if not model_name:
            raise ValueError("Model name is missing in LLM configuration")

        openai_chat = ChatOpenAI(**llm_config)
        self.qa_llm = LLMChat(model_name, openai_chat)
        self.query_llm = LLMChat(model_name, openai_chat)

        # Initialize query helpers
        self.query_helper = QueryHelper(self.query_llm)
        self.query_translation = QueryTranslator(self.query_helper, self.vector_store)

        # Initialize RAG
        rag_config = self.config.get("rag_config", {})
        self.rag = RAG(self.query_helper, self.embedding_client, self.vector_db, **rag_config)

    def _preprocess_query(self, user_query: str, conversation_history: Optional[str]) -> str:
        """
        Preprocess the user query by fixing grammar and adding context.

        Args:
            user_query: The original user query
            conversation_history: Previous conversation context

        Returns:
            Tuple of (processed_query, query_with_context)
        """
        # Fix grammar if configured
        if self.qa_config.get("fix_grammar", False):
            user_query = self.query_helper.fix_grammar_query(user_query)

        user_query = self.query_translation.query_reformulate_with_context(user_query, conversation_history)

        # Apply query translation based on configuration
        query_translation_type = self.qa_config.get("query_translation_type", "")
        user_query = self.query_translation.apply(query_translation_type, user_query)

        return user_query

    def _check_ambiguity(self, query: str) -> bool:
        """
        Check if the query is ambiguous.

        Args:
            query: The user query

        Returns:
            True if the query is ambiguous, False otherwise
        """
        if not self.qa_config.get("ambiguous", False):
            return False

        threshold = self.qa_config.get("ambiguous_threshold")
        return self.query_helper.classify_ambiguous(query, threshold=threshold)

    def _retrieve_context(self, query_with_context: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for the query.

        Args:
            query_with_context: Query with additional context

        Returns:
            List of context documents
        """
        # Use reranking if configured for RAG fusion
        if self._should_use_reranking():
            ranker = Ranker(self.vector_store)
            context = ranker.rank(self.config, query_with_context)
        else:
            context = self.rag.query_similarity(query_with_context)

        # Filter by topic if configured
        if self.qa_config.get("filter_by_topic", False):
            topic_list = self.query_helper.classify_query_topic(query_with_context)
            context = filter_retrieval_result(topic_list, context)

        return context

    def _should_use_reranking(self) -> bool:
        """Determine if reranking should be used based on configuration."""
        return self.qa_config.get("query_translation_type") == "rag_fusion"

    def process_query(
            self,
            user_query: str,
            conversation_history: Optional[str] = None,
            eval: bool = False
    ) -> Union[Tuple[str, Optional[str]], Tuple[Any, List[Dict[str, str]]]]:
        """
        Process a user query and generate a response.

        Args:
            user_query: The user's input query
            conversation_history: Previous conversation context
            eval: Whether this is an evaluation run

        Returns:
            Tuple of (response, updated_conversation_history)
        """
        try:
            # Preprocess query
            user_query = self._preprocess_query(user_query, conversation_history)

            # Check for ambiguity
            if self._check_ambiguity(user_query):
                response = "Can you please be more specific in your question and add details" \
                            "that will help me to better understand you?"
                return response, conversation_history

            # Retrieve relevant context
            context = self._retrieve_context(user_query)

            # Generate response using RAG prompt
            rag_prompt = RAG_PROMPT.format(question=user_query, context=context)

            response, new_conversation_history = self.qa_llm.query(
                prompt=rag_prompt,
                user_query=user_query
            )

            # Reset history if in evaluation mode
            if eval:
                self.qa_llm.initialize_history()

            return response.content, new_conversation_history

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg, conversation_history

    def run_qa_dialog(self) -> None:
        """Run the interactive QA dialog in the terminal."""
        print("Welcome to RagQA Terminal!")
        print("Ask me any question or type 'exit' to quit.")

        conversation_history = None

        while True:
            try:
                user_query = input("Question: ").strip()

                if user_query.lower() == "exit":
                    print("Goodbye!")
                    break

                if not user_query:
                    print("Please enter a question.")
                    continue

                response, conversation_history = self.process_query(
                    user_query,
                    conversation_history
                )
                print(response)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")


def main():
    """Main entry point for the QA application."""
    try:
        config = load_config()
        qa_app = QAApp(config)
        qa_app.run_qa_dialog()
    except Exception as e:
        print(f"Failed to run QA application: {str(e)}")


if __name__ == "__main__":
    main()
