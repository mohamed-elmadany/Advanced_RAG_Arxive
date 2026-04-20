# Advanced RAG over arXiv

A streaming, tool-using **Retrieval-Augmented Generation** system over the full arXiv metadata corpus. Ask the assistant about any research topic; it will retrieve the most relevant papers (FAISS + cross-encoder reranker), summarize them, and on request fetch and read the full PDF text.

The LLM runs locally through **Ollama**, the embeddings + reranker run locally through **sentence-transformers**, and the API is served by **FastAPI** with Server-Sent-Events streaming to a React frontend.

---

## Architecture

```
                                                 ┌─────────────┐
                                                 │   Ollama    │
                                                 │  (qwen3.5)  │
                                                 └──────▲──────┘
                                                        │
   ┌────────┐    HTTP     ┌──────────┐    async    ┌────┴────────┐
   │  React │ ──SSE────► │ FastAPI  │ ─────────► │  RagSystem   │
   │ Front  │ ◄────────── │ /api/chat│ ◄───────── │ (tool loop)  │
   └────────┘             └──────────┘            └────┬─────────┘
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                     ┌──────────┐ ┌──────────┐ ┌────────────┐
                                     │Retriever │ │Reranker  │ │ get_full_  │
                                     │ FAISS +  │ │ Qwen3-   │ │ paper_     │
                                     │ SQLite   │ │ Reranker │ │ content    │
                                     └──────────┘ └──────────┘ └─────┬──────┘
                                                                     │
                                                              arxiv API + PyMuPDF
```

Two tools are exposed to the LLM:
- `get_papers(query, number_of_papers)` — semantic search over the arXiv corpus, then cross-encoder rerank.
- `get_full_paper_content(arxiv_id)` — downloads the PDF from arXiv and extracts full text.

The active LLM is **selectable at runtime** from any model installed in your local Ollama — see [Model selection](#model-selection) below.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([https://ollama.com](https://ollama.com)). Pull the chat model:
  ```bash
  ollama pull qwen3.5:4b
  ```
- A **Kaggle** account + API token (for the dataset download).
- **~12 GB free disk**: ~5 GB for the snapshot, ~3 GB for the FAISS index + SQLite DB, ~2 GB for models, headroom for downloaded PDFs.
- A CUDA GPU is optional but strongly recommended for ingestion — CPU works, just slower.

---

## Setup

```bash
git clone <this-repo>
cd Advanced_RAG_Arxive

pip install -r requirments.txt
```

### 1. Configure Kaggle credentials

Either set environment variables:
```bash
set KAGGLE_USERNAME=<your_username>
set KAGGLE_KEY=<your_api_key>
```

Or place `kaggle.json` (downloaded from `https://www.kaggle.com/settings` → *Create New API Token*) at:
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json`

### 2. Download the arXiv snapshot

```bash
python scripts/download_dataset.py
```
Downloads `Cornell-University/arxiv` (~5 GB) and copies `arxiv-metadata-oai-snapshot.json` into `Data/`.

### 3. Download the embedding + reranker models

```bash
python Model_manager/download_models.py
```
Downloads `bge-small-en-v1.5` and `Qwen/Qwen3-Reranker-0.6B` into `Models/`.

### 4. Build the FAISS index + SQLite metadata DB

```bash
python ingestion/pipeline.py
```
Trains an IVFFlat index on 50k samples then ingests every paper. Outputs:
- `Storage/arxiv_full_ivf.faiss`
- `Storage/arxiv_metadata.db`

### 5. Start the API

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

---

## Project layout

```
Advanced_RAG_Arxive/
├── core/
│   └── config.py              # single source of truth for paths + hyperparameters
├── ingestion/
│   └── pipeline.py            # builds FAISS index + SQLite DB from the arXiv snapshot
├── Model_manager/
│   ├── download_models.py     # downloads embedding + reranker models locally
│   └── load_models.py         # lazy singletons with auto-download fallback
├── LLM/
│   ├── Retriever.py           # FAISS + SQLite semantic search
│   ├── Reranker.py            # Qwen3-Reranker cross-encoder
│   └── Bot.py                 # legacy simple chat (unused)
├── src/
│   ├── app.py                 # FastAPI app: /api/chat (SSE) + /api/reset
│   └── Rag_Arxiv.py           # RagSystem: async tool-use loop over Ollama
├── scripts/
│   └── download_dataset.py    # Kaggle dataset fetch via kagglehub
├── frontend/                  # React UI
├── Data/                      # arXiv snapshot lands here
├── Models/                    # local model weights land here
├── Storage/                   # FAISS index + SQLite DB land here
└── papers/                    # downloaded PDFs cached here
```

---

## Model selection

You can switch the chat LLM on the fly without restarting the server.

**Pull any models you want available** (Ollama caches them locally):
```bash
ollama pull qwen3.5:4b
ollama pull llama3.1:8b
ollama pull mistral
```

**From the UI:** the header has a `MODEL` dropdown that lists every model installed in Ollama. The default selected on load is `LLM_MODEL_NAME` from [core/config.py](core/config.py) (or the first installed model if the default isn't pulled). Switching mid-conversation takes effect on the next message you send.

**From the API:** every `POST /api/chat` accepts an optional `model` field. Omit it to use the server default.
```bash
curl -N -X POST localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"explain RLHF in 2 sentences","model":"llama3.1:8b"}'
```

**Listing installed models:**
```bash
curl localhost:8000/api/models
# {"models":["qwen3.5:4b","llama3.1:8b","mistral:latest"],"default":"qwen3.5:4b"}
```

Note: only models that support **tool use / function calling** will work end-to-end with this agent. If you select a model that ignores tool calls, retrieval and PDF fetch won't be triggered — the model will just answer from its prior.

---

## Configuration

All paths and hyperparameters live in [core/config.py](core/config.py). Override-able fields include:

| Field | Default | What it controls |
|-------|---------|------------------|
| `LLM_MODEL_NAME` | `qwen3.5:4b` | Ollama model name used by the agent |
| `EMBEDDING_MODEL_NAME` | `bge-small-en-v1.5` | sentence-transformers embedding model |
| `QWEN_RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | cross-encoder rerank model |
| `EMBEDDING_DIM` | `384` | must match the embedding model output dim |
| `IVF_NLIST` | `2048` | FAISS IVF cluster count |
| `IVF_TRAIN_SAMPLES` | `50_000` | samples used to train the IVF index |
| `INGEST_BATCH_SIZE` | `64` | embedding batch size during ingestion |

---

## Production notes

- **CORS** is wide open (`allow_origins=["*"]`) in [src/app.py](src/app.py) for development. Lock it down to your frontend origin before exposing the API publicly.
- The FastAPI app instantiates a single `RagSystem` at process start, which means **conversation state is shared across all requests**. For multi-user deployments, scope `RagSystem` per session (e.g. via a session-id header and a dict of systems).
- The PDF tool writes to `papers/`. That directory is gitignored but can grow unbounded — set up periodic cleanup if you run long-lived.

---

## Troubleshooting

**`FileNotFoundError: Dataset not found at Data/arxiv-metadata-oai-snapshot.json`**
Run `python scripts/download_dataset.py` first.

**`FileNotFoundError: Missing FAISS index ... Run python ingestion/pipeline.py first`**
You skipped the ingestion step. Run it.

**`RuntimeError: No CUDA GPUs are available` during ingestion**
Already handled — `autocast` is now CPU-safe. If you still hit this, make sure you pulled the latest [ingestion/pipeline.py](ingestion/pipeline.py).

**Ollama connection refused**
Make sure `ollama serve` is running and the model in `LLM_MODEL_NAME` has been pulled (`ollama pull qwen3.5:4b`).

**Embedding/reranker model not found**
Run `python Model_manager/download_models.py`. The loaders also auto-download on first miss but explicit is safer for the first run.

**Bad arXiv ID crashes the chat**
It no longer should — `get_full_paper_content` now returns a per-paper `[error fetching <id>: ...]` string and the agent continues. If you still see a crash, capture the ID and open an issue.
