from typing import List, Any

from core.llm.llm_chat import MAX_TOKENS
from core.llm.utils import count_tokens
from query.prompts import (
    TOPIC_CLASSIFICATION_PROMPT,
    AMBIGUOUS_PROMPT,
    SUMMARIZE_CONVERSATION_PROMPT,
    GRAMMAR_FIX_PROMPT,
    LLM_AS_JUDGE_ACC_PROMPT, LLM_AS_JUDGE_REL_PROMPT, SUMMARIZE_DOCUMENT_PROMPT, SUMMARY_QUERY_COMBINE_PROMPT
)


class QueryHelper:
    """
    Helper class for query processing and classification using LLMs.
    """

    def __init__(self, llm: Any):
        """
        Initialize the QueryHelper.

        Args:
            llm: A language model client.
        """
        self._llm = llm

    @property
    def llm(self):
        return self._llm

    def _run_query(self, prompt: str) -> str:
        """
        Run a query through the LLM and reset history.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The content of the response as a string
        """
        response, _ = self._llm.query(prompt)
        self._llm.initialize_history()
        return response.content

    def classify_query_topic(self, query: str) -> List[str]:
        """
        Classify the topic of a query.

        Args:
            query: The query to classify

        Returns:
            A list of topic categories as strings
        """
        query_prompt = TOPIC_CLASSIFICATION_PROMPT.format(query=query)
        response = self._run_query(query_prompt)

        # Clean and parse the response
        response = response.replace("Category: ", "").lower()
        topic_list = [topic.strip() for topic in response.split(",")]
        return topic_list

    def classify_ambiguous(self, query: str, threshold: float = 0.8) -> bool:
        """
        Determine if a query is ambiguous.

        Args:
            query: The query to check for ambiguity
            threshold: The threshold score for considering a query ambiguous (0.0 to 1.0)

        Returns:
            True if the query is ambiguous, False otherwise
        """
        query_prompt = AMBIGUOUS_PROMPT.format(query=query)
        response = self._run_query(query_prompt)

        try:
            # Extract and parse the ambiguity score
            ambiguous_score = response.replace("Score: ", "")
            return float(ambiguous_score) >= threshold
        except ValueError:
            # Handle case where response isn't a valid score
            return False

    def summarize_conversation(self, conversation: str) -> str:
        """
        Summarize a conversation.

        Args:
            conversation: The conversation text to summarize

        Returns:
            A summary of the conversation
        """
        query_prompt = SUMMARIZE_CONVERSATION_PROMPT.format(conversation=conversation)
        return self._run_query(query_prompt)

    def combine_summary_query(self, summary: str, query: str) -> str:
        """
        Combine a summary and a query.

        Args:
            summary: The summary text
            query: The query text

        Returns:
            A combined string with summary and query
        """
        #Using the prompt might alter the user query based on the history,
        #which can lead to an undesirable effect

        query_prompt = SUMMARY_QUERY_COMBINE_PROMPT.format(conversation_history=summary, query=query)
        return self._run_query(query_prompt)

    def fix_grammar_query(self, query: str) -> str:
        """
        Fix grammar in a query.

        Args:
            query: The query to fix

        Returns:
            The grammatically corrected query
        """
        query_prompt = GRAMMAR_FIX_PROMPT.format(query=query)
        return self._run_query(query_prompt)

    def llm_as_a_judge_accuracy_query(self, gt, answer):
        """
       Evaluate the accuracy of an answer compared to ground truth.

       Args:
           ground_truth: The ground truth answer
           answer: The generated answer to evaluate

       Returns:
           A score between 0 and 1 representing accuracy
       """

        query_prompt = LLM_AS_JUDGE_ACC_PROMPT.format(ground_truth=gt,
                                                      generated_answer=answer)
        return self._run_query(query_prompt)

    def llm_as_a_judge_relevance_query(self, question, answer):
        """
        Evaluate the relevance of an answer to a question.

        Args:
            question: The original question
            answer: The answer to evaluate

        Returns:
            A score between 0 and 1 representing relevance
        """

        query_prompt = LLM_AS_JUDGE_REL_PROMPT.format(question=question,
                                                      answer=answer)
        return self._run_query(query_prompt)

    def summarize_text(self, text):
        """
        Generate a summary of a text document.

        Args:
            text: The document text to summarize

        Returns:
            A summary of the document
        """

        query_prompt = SUMMARIZE_DOCUMENT_PROMPT.format(document_text=text)
        num_tokens = count_tokens(query_prompt)

        if num_tokens >= MAX_TOKENS * 0.8:
            # splitting the text to paragraphs aims to deal with large context length
            text_paragraphs = [paragraph for paragraph in text.split("\n\n") if len(paragraph.split(" ")) > 50]
            summary = "".join(self._run_query(SUMMARIZE_DOCUMENT_PROMPT.format(document_text=p)) for p in text_paragraphs)
            query_prompt = SUMMARIZE_DOCUMENT_PROMPT.format(document_text=summary)

        return self._run_query(query_prompt)

