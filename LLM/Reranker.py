from Model_manager.load_models import get_qwen_reranker


class Reranker:
    def __init__(self):
        self.model = get_qwen_reranker()

    def rerank(self, query, documents, top_n=10):
        if not documents:
            return []

        sentence_pairs = [[query, doc["title"] + doc["abstract"]] for doc in documents]
        scores = self.model.predict(sentence_pairs)

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:top_n]
