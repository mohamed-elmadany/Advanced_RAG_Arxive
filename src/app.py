from LLM.Bot import bot
from LLM.Retriever import Retriever
from LLM.Reranker import Reranker
class RagSystem:
    def __init__(self):
        self.llm = bot()
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.optimizer = bot()
    def run(self):
        print("Welcome to the ArXiv RAG System!")
        print("Type '/exit' to quit.")
        while True:
            query = input("\n>> ")
            if query.lower() == "/exit":
                print("Goodbye!")
                break
            optimized =self.optimizer.ask(f"""Given the user query: '{query}', 
                                          generate a more specific search query to retrieve relevant papers from ArXiv. 
                                          Focus on key terms and concepts, reply with only the optimized query without any explanation.
                                          example: user query: 'What are the latest advancements in deep learning for natural language processing?'
                                          optimized query: 'latest advancements in deep learning natural language processing , methods , applications , theories '
                                          """)
            retrieved_docs = self.retriever.retrieve(optimized, top_k=100)
            print("top 100 papers:\n")
            print("\n\n".join([f"{doc['title']} , distance: {doc['distance']}" for doc in retrieved_docs]))
            if not retrieved_docs:
                print("No relevant papers found.")
                continue
            ranked_docs = self.reranker.rerank(query, retrieved_docs,top_n=20)

            print("Top 5 papers:\n")
            print("\n\n".join([f"{i+1}. {doc['title']}, score: {doc['rerank_score']}" for i, doc in enumerate(ranked_docs)]))
            reranked_docs = self.reranker.rerank(query, retrieved_docs,top_n=5)

            print("Top 20 papers:\n")
            print("\n\n".join([f"{i+1}. {doc['title']}, score: {doc['rerank_score']}" for i, doc in enumerate(reranked_docs)]))

            ##context = "\n\n".join([f"Title: {doc['title']}\nAbstract: {doc['abstract']}" for doc in ranked_docs[:5]])
            ##prompt = f"Based on the following papers:\n{context}\n\nAnswer the question: {query}"
            ##self.llm.ask(prompt)
if __name__ == "__main__":
    system = RagSystem()
    system.run()
