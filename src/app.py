from LLM.Reranker import Reranker
from LLM.Retriever import Retriever
from ollama import chat



class RagSystem:
    def __init__(self, model: str = "qwen3.5:4b"):
        self.model = model
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.messages = []
        self.tool_map = self._build_tool_map()
 
    # ------------------------------------------------------------------ #
    #  Tool registry                                                       #
    # ------------------------------------------------------------------ #
 
    def _build_tool_map(self) -> dict:
        """Auto-build the tool map from all methods tagged as tools."""
        tools = [self.get_papers]   # add new tools here and nowhere else
        return {fn.__name__: fn for fn in tools}
 
    def _run_tool(self, call) -> str:
        name = call.function.name
        args = call.function.arguments
        print(f"\n[Tool Call] {name}({args})")
 
        if name in self.tool_map:
            return self.tool_map[name](**args)
        return f"Unknown tool: '{name}'"
 
    # ------------------------------------------------------------------ #
    #  Tool: get_papers                                                    #
    # ------------------------------------------------------------------ #
 
    def get_papers(self, query: str, number_of_papers: int = 5) -> str:
        """Search and retrieve the most relevant academic papers for a given query.
 
        Args:
            query: The search query describing the research topic.
            number_of_papers: How many top papers to return (between 1 and 20).
 
        Returns:
            A numbered list of the most relevant paper titles with relevance scores.
        """
        number_of_papers = max(1, min(number_of_papers, 20))
 
        retrieved_docs = self.retriever.retrieve(query, top_k=100)
        if not retrieved_docs:
            return "No relevant papers found for this query."
 
        print(f"\n[Retriever] {len(retrieved_docs)} candidates for: '{query}'")
 
        top20 = self.reranker.rerank(query, retrieved_docs, top_n=20)
        print("\n[Reranker] Top 20:")
        for i, doc in enumerate(top20):
            print(f"  {i+1:2}. {doc['title']}  (score: {doc['rerank_score']:.4f})")
 
        final_docs = self.reranker.rerank(query, retrieved_docs, top_n=number_of_papers)
        print(f"\n[Final] Top {number_of_papers} sent to LLM:")
        for i, doc in enumerate(final_docs):
            print(f"  {i+1}. {doc['title']}  (score: {doc['rerank_score']:.4f})")
 
        return "\n\n".join(
            f"{i+1}. {doc['title']}\n   Relevance score: {doc['rerank_score']:.4f}"
            for i, doc in enumerate(final_docs)
        )
 
    # ------------------------------------------------------------------ #
    #  Agentic loop                                                        #
    # ------------------------------------------------------------------ #
 
    def _agent_step(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
 
        while True:
            stream = chat(
                model=self.model,
                messages=self.messages,
                tools=list(self.tool_map.values()),
                stream=True,
                think=True,
            )
 
            thinking      = ""
            content       = ""
            tool_calls    = []
            done_thinking = False
 
            for chunk in stream:
                if chunk.message.thinking:
                    thinking += chunk.message.thinking
                    print(chunk.message.thinking, end="", flush=True)
 
                if chunk.message.content:
                    if not done_thinking:
                        done_thinking = True
                        print("\n -----done thinking----- \n\n ")
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
 
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)
 
            if thinking or content or tool_calls:
                self.messages.append({
                    "role":       "assistant",
                    "thinking":   thinking,
                    "content":    content,
                    "tool_calls": tool_calls,
                })
 
            if not tool_calls:
                print()
                return content
 
            for call in tool_calls:
                result = self._run_tool(call)
                self.messages.append({
                    "role":      "tool",
                    "tool_name": call.function.name,
                    "content":   result,
                })
 
    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #
 
    def chat(self, user_message: str) -> str:
        return self._agent_step(user_message)
 
    def reset(self):
        self.messages = []
        print("[System] Conversation history cleared.")
 
    def run(self):
        print("=" * 60)
        print(f"  ArXiv Agentic RAG  |  model: {self.model}")
        print("  Commands: /exit  /reset  /history")
        print("=" * 60)
 
        while True:
            try:
                query = input("\n>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
 
            if not query:
                continue
            if query.lower() == "/exit":
                print("Goodbye!")
                break
            if query.lower() == "/reset":
                self.reset()
                continue
            if query.lower() == "/history":
                for turn in self.messages:
                    role = turn["role"].upper()
                    body = turn.get("content") or str(turn.get("tool_calls", ""))
                    print(f"\n[{role}]\n{body}")
                continue
 
            self._agent_step(query)
 
 
if __name__ == "__main__":
    system = RagSystem(model="qwen3.5:4b")
    system.run()