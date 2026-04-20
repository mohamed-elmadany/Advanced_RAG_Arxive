import json
import asyncio
from pathlib import Path

import arxiv
import fitz
from ollama import AsyncClient

from LLM.Reranker import Reranker
from LLM.Retriever import Retriever
from core.config import config


class RagSystem:
    def __init__(self, model: str = None):
        self.model = model or config.LLM_MODEL_NAME
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.client = AsyncClient()
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are an AI Research Assistant."
                "your job is to help users find and understand academic papers."
                "maintain a proffessional but friendly tone. Always use the tools provided to find relevant papers and their content."
                "Use 'get_papers' to find relevant studies. If a user needs deep details,"
                "always enhance the user's query before tool calling to get more accurate results. feel free to ask the user for clarification if the query is vague or to get more details of what research they are interested in."
                "use 'get_full_paper_content' with the specific arxiv_id. "
                "always make the user decides if they want to get the full paper content."
                "other than that, try to answer the user's question based on the retrieved papers abstract, try to explain briefly the main idea of the paper and how it relates to the user's question."
                "Always cite the paper_id when answering."
            ),
        }
        self.messages = [self.system_prompt]
        self.tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict:
        tools = [self.get_papers, self.get_full_paper_content]
        return {fn.__name__: fn for fn in tools}

    async def summarize_conversation(self):
        summary_prompt = {
            "role": "system",
            "content": "Summarize the following conversation between a user and an assistant in 4-5 sentences, focusing on the user's main research question and the assistant's key findings.",
        }
        try:
            response = await self.client.chat(
                model=self.model, messages=[summary_prompt] + self.messages
            )
            summary = response.get("message", {}).get("content", "")
            if summary:
                self.messages = [
                    self.system_prompt,
                    {"role": "assistant", "content": summary},
                ]
        except Exception:
            pass

    async def _run_tool(self, call) -> str:
        name = call.function.name
        args = call.function.arguments or {}
        if name in self.tool_map:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    None, lambda: self.tool_map[name](**args)
                )
            except Exception as exc:
                return f"[tool '{name}' failed: {exc}]"
        return f"Unknown tool: '{name}'"

    def get_full_paper_content(self, arxiv_id):
        if isinstance(arxiv_id, str):
            arxiv_id = [arxiv_id]
        if not isinstance(arxiv_id, list) or not arxiv_id:
            return "No arxiv_id provided."

        ids = [str(i).strip() for i in arxiv_id if i and str(i).strip()]
        if not ids:
            return "No valid arxiv_id provided."

        papers_dir = Path(config.PAPERS_DIR)
        papers_dir.mkdir(parents=True, exist_ok=True)

        results = []
        try:
            search = arxiv.Search(id_list=ids)
            paper_iter = search.results()
        except Exception as exc:
            return f"[arxiv search failed: {exc}]"

        for pid, paper in zip(ids, paper_iter):
            try:
                clean_title = "".join(
                    c for c in paper.title if c.isalnum() or c in (".", "_")
                ).rstrip() or pid.replace("/", "_")
                filename = f"{clean_title}.pdf"
                pdf_path = papers_dir / filename

                if not pdf_path.exists():
                    pdf_path = Path(
                        paper.download_pdf(dirpath=str(papers_dir), filename=filename)
                    )

                if not pdf_path.exists():
                    results.append(f"[error fetching {pid}: PDF not saved]")
                    continue

                with fitz.open(str(pdf_path)) as doc:
                    full_text = "".join(page.get_text() for page in doc)

                if not full_text.strip():
                    results.append(f"[error fetching {pid}: empty PDF text]")
                else:
                    results.append(full_text)
            except Exception as exc:
                results.append(f"[error fetching {pid}: {exc}]")

        if not results:
            return "No content retrieved."
        return "\n\n".join(results)

    def get_papers(self, query: str, number_of_papers: int = 5) -> str:
        if not query or not isinstance(query, str):
            return "No query provided."
        number_of_papers = max(1, min(number_of_papers, 20))
        try:
            retrieved_docs = self.retriever.retrieve(query, top_k=100)
        except Exception as exc:
            return f"[retrieval failed: {exc}]"
        if not retrieved_docs:
            return "No papers found."
        try:
            final_docs = self.reranker.rerank(
                query, retrieved_docs, top_n=number_of_papers
            )
        except Exception as exc:
            return f"[rerank failed: {exc}]"
        return "\n\n".join(
            f"paper_id {d['arxiv_id']}. title: {d['title']}\n Abstract: {d['abstract']}"
            for d in final_docs
        )

    async def stream_chat(self, user_message: str, model: str | None = None):
        active_model = model or self.model
        if len(self.messages) > 3:
            await self.summarize_conversation()
        self.messages.append({"role": "user", "content": user_message})

        full_response_content = ""

        for _ in range(3):
            tool_calls_detected = []

            async for chunk in await self.client.chat(
                model=active_model,
                messages=self.messages,
                tools=list(self.tool_map.values()),
                stream=True,
            ):
                if chunk.get("message", {}).get("content"):
                    text = chunk["message"]["content"]
                    full_response_content += text
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

                if chunk.get("message", {}).get("tool_calls"):
                    tool_calls_detected.extend(chunk["message"]["tool_calls"])

            if not tool_calls_detected:
                break

            self.messages.append({
                "role": "assistant",
                "content": full_response_content,
                "tool_calls": tool_calls_detected,
            })

            for call in tool_calls_detected:
                result = await self._run_tool(call)

                yield f"data: {json.dumps({'type': 'tool', 'name': call['function']['name'], 'content': result})}\n\n"

                self.messages.append({"role": "tool", "content": result})

            full_response_content = ""

        if full_response_content:
            self.messages.append({"role": "assistant", "content": full_response_content})

    def reset(self):
        self.messages = [self.system_prompt]
