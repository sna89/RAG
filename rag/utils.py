from typing import List, Any
from langchain.load import dumps, loads

TOPICS = ["food", "automobile", "steel", "textile"]
GENERAL_TOPIC = "general"


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc[0]) if isinstance(doc, tuple) else dumps(doc) for doc in documents]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def filter_retrieval_result(topic_list: List[str], results: List[Any]) -> List[Any]:
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
        return result[0].metadata["filename"].split(".")[0].lower()

    return [result for result in results if get_topic(result) in topic_list]


# Reciprocal Rank Fusion algorithm
