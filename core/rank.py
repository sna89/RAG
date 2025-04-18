import re


class Ranker:
    def __init__(self,
                 vector_store):

        self.vector_store = vector_store

    def rank(self, config, query_set_str):
        return self.rff_wrapper(query_set_str)

    def rff_wrapper(self, query_set_str):
        rrf_dict = {}
        for query_t in re.split(r'\n\n|\n', query_set_str):
            query_dict = {}

            context_list = self.vector_store.similarity_search_with_score(query_t, k=5)
            for context in context_list:
                query_dict[context[0].page_content] = context[1]

            rrf_dict[query_t] = query_dict
        ranked_results = self.reciprocal_rank_fusion(rrf_dict)
        return list(ranked_results.keys())

    @staticmethod
    def reciprocal_rank_fusion(search_results_dict, k=60):
        # https://github.com/Raudaschl/rag-fusion/blob/master/main.py
        fused_scores = {}

        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                fused_scores[doc] += 1 / (rank + k)

        ranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        return ranked_results
