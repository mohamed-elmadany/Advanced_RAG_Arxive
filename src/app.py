from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ollama import AsyncClient
from pydantic import BaseModel

from core.config import config

from .Rag_Arxiv import RagSystem

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RagSystem()
ollama_client = AsyncClient()


class ChatInput(BaseModel):
    message: str
    model: str | None = None


@app.get("/api/models")
async def list_models():
    try:
        response = await ollama_client.list()
        models = []
        for m in response.get("models", []):
            name = m.get("model") or m.get("name")
            if name:
                models.append(name)
        return {"models": models, "default": config.LLM_MODEL_NAME}
    except Exception as exc:
        return {"models": [], "default": config.LLM_MODEL_NAME, "error": str(exc)}


@app.post("/api/chat")
async def chat_endpoint(request: ChatInput):
    return StreamingResponse(
        rag.stream_chat(request.message, model=request.model),
        media_type="text/event-stream",
    )


@app.post("/api/reset")
async def reset_chat():
    rag.reset()
    return {"status": "history cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
