from Model_manager.load_models import get_reranker_model


class Reranker:
    def __init__(self):
        self.model = get_reranker_model()
        
    def rerank(self, query, documents, top_n=10):
        if not documents:
            return []

        # Prepare pairs: [[query, doc1], [query, doc2], ...]
        sentence_pairs = [[query, doc['abstract']+doc['title']] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(sentence_pairs)
        
        # Attach scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])

        # Sort by score descending
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_n]
