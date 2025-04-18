from typing import Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List

from query.prompts import MULTIQUERY_PROMPT
import re


class QueryTranslator:
    def __init__(self,
                 query_helper,
                 vector_store,
                 ):
        self.query_helper = query_helper
        self.llm = self.query_helper.llm
        self.vector_store = vector_store

    def apply(self, query_translation_type: str, user_query: str) -> Union[str, List[str]]:
        assert query_translation_type in ["base", "multi_query", "rag_fusion"], "Error in query_translation_type config"

        if query_translation_type == "base":
            return user_query
        elif query_translation_type == "multi_query":
            return self.multi_query(user_query)
        elif query_translation_type == "rag_fusion":
            return self.rag_fusion(user_query)

    def multi_query(self, query: str) -> list[str]:
        prompt_perspectives = ChatPromptTemplate.from_template(MULTIQUERY_PROMPT)

        generate_queries = (
                prompt_perspectives
                | self.llm
                | StrOutputParser()
        )

        context = generate_queries.invoke({"question": query})
        context = re.split("\n\n|\n", context)
        return context

    def rag_fusion(self, query: str) -> List[str]:
        return self.multi_query(query)

    def query_reformulate_with_context(self, query, context: Union[str, List[dict[str, str]]] = None) -> str:
        if context:
            summarized_history = self.query_helper.summarize_conversation(context)
            query = self.query_helper.combine_summary_query(summarized_history, query)

        return query
