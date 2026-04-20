import json
import sqlite3
from contextlib import nullcontext

import faiss
import torch
from tqdm import tqdm

from core.config import config
from Model_manager.load_models import get_embedding_model


def stream_arxiv(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.RAW_DATA_PATH}. "
            "Run `python scripts/download_dataset.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    model = get_embedding_model()

    conn = sqlite3.connect(str(config.DB_FILE))
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS papers
                      (id INTEGER PRIMARY KEY, arxiv_id TEXT, title TEXT, abstract TEXT)"""
    )
    conn.commit()

    quantizer = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(
        quantizer, config.EMBEDDING_DIM, config.IVF_NLIST, faiss.METRIC_L2
    )

    print(f"Phase 1: Training on {config.IVF_TRAIN_SAMPLES} samples...")
    train_texts = []
    for data in stream_arxiv(config.RAW_DATA_PATH):
        title = data.get("title")
        abstract = data.get("abstract")
        if not title or not abstract:
            continue
        train_texts.append(f"{title}. {abstract}")
        if len(train_texts) >= config.IVF_TRAIN_SAMPLES:
            break

    if len(train_texts) < config.IVF_NLIST:
        raise RuntimeError(
            f"Only {len(train_texts)} valid training samples found; "
            f"need at least IVF_NLIST={config.IVF_NLIST}."
        )

    train_embeddings = model.encode(
        train_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )
    index.train(train_embeddings)
    del train_texts, train_embeddings

    print("Phase 2: Ingesting vectors and metadata...")
    batch_texts = []
    batch_meta = []
    current_id = 0

    total_lines = count_lines(config.RAW_DATA_PATH)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda") if device == "cuda" else nullcontext()
    )

    def flush(texts, meta):
        with autocast_ctx:
            embeddings = model.encode(texts, convert_to_numpy=True)
        index.add(embeddings)
        cursor.executemany("INSERT INTO papers VALUES (?, ?, ?, ?)", meta)
        conn.commit()

    for data in tqdm(stream_arxiv(config.RAW_DATA_PATH), total=total_lines):
        title = data.get("title")
        abstract = data.get("abstract")
        arxiv_id = data.get("id")
        if not title or not abstract or not arxiv_id:
            continue
        batch_texts.append(f"{title}. {abstract}")
        batch_meta.append((current_id, arxiv_id, title, abstract))
        current_id += 1

        if len(batch_texts) == config.INGEST_BATCH_SIZE:
            flush(batch_texts, batch_meta)
            batch_texts, batch_meta = [], []

    if batch_texts:
        flush(batch_texts, batch_meta)

    faiss.write_index(index, str(config.VECTOR_INDEX_PATH))
    conn.close()
    print(f"Done! Index ({config.VECTOR_INDEX_PATH}) and DB ({config.DB_FILE}) ready.")


if __name__ == "__main__":
    main()
