from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "Models"
STORAGE_DIR = BASE_DIR / "storage"
EMBEDDING_MODEL_PATH = MODELS_DIR / "embeddings" / "bge-small-en-v1.5"
RERANKER_MODEL_PATH = MODELS_DIR / "reranker" / "bge-reranker-v2-m3"
VECTORDCONFIG_PATH = STORAGE_DIR / "arxiv_full_ivf.faiss"
DB_FILE = STORAGE_DIR / "arxiv_metadata.db"

