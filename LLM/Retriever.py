from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
from Model_manager.load_models import get_embedding_model
from Model_manager.paths import VECTORDCONFIG_PATH, DB_FILE

class Retriever:
    def __init__(self):
        self.model = get_embedding_model()
        self.index = faiss.read_index(str(VECTORDCONFIG_PATH))
        self.conn = sqlite3.connect(str(DB_FILE))
        self.cursor = self.conn.cursor()
    def retrieve(self, query, top_k=50):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i in range(top_k):
            self.cursor.execute("SELECT arxiv_id, title , abstract FROM papers WHERE id=?", (int(indices[0][i]),))
            row = self.cursor.fetchone()
            if row:
                results.append({"arxiv_id": row[0], "title": row[1], "abstract": row[2], "distance": float(distances[0][i])})
                
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
