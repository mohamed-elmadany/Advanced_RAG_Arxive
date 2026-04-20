import faiss
import sqlite3
from Model_manager.load_models import get_embedding_model
from core.config import config


class Retriever:
    def __init__(self):
        if not config.VECTOR_INDEX_PATH.exists() or not config.DB_FILE.exists():
            raise FileNotFoundError(
                f"Missing FAISS index ({config.VECTOR_INDEX_PATH}) or SQLite DB ({config.DB_FILE}). "
                "Run `python ingestion/pipeline.py` first."
            )
        self.model = get_embedding_model()
        self.index = faiss.read_index(str(config.VECTOR_INDEX_PATH))
        self.conn = sqlite3.connect(str(config.DB_FILE), check_same_thread=False)
        self.cursor = self.conn.cursor()

    def retrieve(self, query, top_k=50):
        if not query or not isinstance(query, str):
            return []
        top_k = max(1, min(top_k, self.index.ntotal))
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i in range(top_k):
            faiss_id = int(indices[0][i])
            if faiss_id < 0:
                continue
            self.cursor.execute(
                "SELECT arxiv_id, title, abstract FROM papers WHERE id=?",
                (faiss_id,),
            )
            row = self.cursor.fetchone()
            if row:
                results.append({
                    "arxiv_id": row[0],
                    "title": row[1],
                    "abstract": row[2],
                    "distance": float(distances[0][i]),
                })
        return results

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    retriever = Retriever()
    query = input(">> ")
    results = retriever.retrieve(query=query, top_k=5)
    print("\nTop Results:\n\n")
    for result in results:
        print(f"ArXiv ID: {result['arxiv_id']}")
        print(f"Title: {result['title']}")
        print(f"Abstract: {result['abstract']}")
        print("-" * 50)
