import openai

from llm.llm_helper import LLMChat
from query.prompts import RAG_PROMPT
from query.query_helper import QueryHelper
from rag.rag import RAG
from config import load_config
from typing import List, Optional, Tuple, Dict, Any, Union

TOPICS = ["food", "automobile", "steel", "textile"]
GENERAL_TOPIC = "general"


class QAApp:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the QA application with the provided configuration.

        Args:
            config: Configuration dictionary with required settings
        """
        self.config = config
        openai.api_key = config["env_config"]["openai_api_key"]

        self.qa_config = config.get("qa_config", {})

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components for the QA system."""

        env_config = self.config.get("env_config", {})
        llm_config = self.config.get("llm_config", {})
        rag_config = self.config.get("rag_config", {})

        self.rag = RAG(env_config, llm_config, **rag_config)

        # Create LLM instances
        self.qa_llm = LLMChat(**llm_config)
        self.query_llm = LLMChat(**llm_config)

        # Initialize query helper
        self.query_helper = QueryHelper(self.query_llm)

    def index_documents(self, path: str, override_db: bool = False) -> None:
        """
        Index documents from the given path.

        Args:
            path: Path to the documents to index
            override_db: Whether to override existing database
        """
        self.rag.index_documents(path, override_db=override_db)

    @staticmethod
    def _filter_retrieval_result(topic_list: List[str], results: List[Any]) -> List[Any]:
        """
        Filter retrieval results based on topics.

        Args:
            topic_list: List of topics to filter by
            results: Retrieval results to filter

        Returns:
            Filtered results
        """
        if GENERAL_TOPIC in topic_list:
            return results

        def get_topic(result):
            return result[0].metadata["source"].split("\\")[-1].split(".")[0].lower()

        return [result for result in results if get_topic(result) in topic_list]

    def process_query(self, user_query: str, conversation_history: Optional[str], eval=False) -> Union[
        tuple[str, Optional[str]], tuple[Any, list[dict[str, str]]]]:
        """
        Process a user query and generate a response.

        Args:
            user_query: The user's input query
            conversation_history: Previous conversation context

        Returns:
            Tuple of (response, updated_conversation_history)
        """
        # Fix grammar and spelling in the query for improved retrieval and overall performance
        user_query = self.query_helper.fix_grammar_query(user_query)
        user_query_w_summary = user_query

        # Incorporate conversation history if available
        if conversation_history:
            conversation_history = self.query_helper.summarize_conversation(conversation_history)
            user_query_w_summary = self.query_helper.combine_summary_query(conversation_history, user_query)

        # Check if query is ambiguous
        is_ambiguous = self.query_helper.classify_ambiguous(
            user_query_w_summary,
            threshold=self.qa_config.get("ambiguous_threshold")
        )

        if is_ambiguous:
            return ("Can you please be more specific in your question "
                    "and add details that will help me to better understand you?"), conversation_history

        # Process non-ambiguous query
        results = self.rag.query_similarity(user_query_w_summary)
        topic_list = self.query_helper.classify_query_topic(user_query_w_summary)
        filtered_results = self._filter_retrieval_result(topic_list, results)

        # Extract context from results
        context = [result[0].page_content for result in filtered_results]

        # Generate response using RAG prompt
        rag_prompt = RAG_PROMPT.format(question=user_query, context=context)
        response, new_conversation_history = self.qa_llm.query(
            prompt=rag_prompt,
            user_query=user_query
        )

        if eval:
            self.qa_llm.initialize_history()

        return response.content, new_conversation_history

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
