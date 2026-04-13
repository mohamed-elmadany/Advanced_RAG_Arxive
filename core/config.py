from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    # Paths
    DATA_DIR: Path = BASE_DIR / "data" / "raw"
    MODELS_DIR: Path = BASE_DIR / "models"
    STORAGE_DIR: Path = BASE_DIR / "storage"

    # Models
    RAW_DATA_PATH: Path = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
    EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "embeddings" / "bge-small-en-v1.5"
    RERANKER_MODEL_PATH: Path = MODELS_DIR / "reranker" / "bge-reranker-v2-m3"
    LLM_MODEL_NAME: str ="gemma3:4b"

config = Config()