from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    DATA_DIR: Path = field(default_factory=lambda: BASE_DIR / "Data")
    MODELS_DIR: Path = field(default_factory=lambda: BASE_DIR / "Models")
    STORAGE_DIR: Path = field(default_factory=lambda: BASE_DIR / "Storage")
    PAPERS_DIR: Path = field(default_factory=lambda: BASE_DIR / "papers")

    LLM_MODEL_NAME: str = "qwen3.5:4b"
    EMBEDDING_MODEL_NAME: str = "bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "bge-reranker-v2-m3"
    QWEN_RERANKER_MODEL_NAME: str = "Qwen/Qwen3-Reranker-0.6B"

    EMBEDDING_DIM: int = 384
    INGEST_BATCH_SIZE: int = 64
    IVF_NLIST: int = 2048
    IVF_TRAIN_SAMPLES: int = 50_000

    RAW_DATA_PATH: Path = field(init=False)
    EMBEDDING_MODEL_PATH: Path = field(init=False)
    RERANKER_MODEL_PATH: Path = field(init=False)
    QWEN_RERANKER_MODEL_PATH: Path = field(init=False)
    VECTOR_INDEX_PATH: Path = field(init=False)
    DB_FILE: Path = field(init=False)

    def __post_init__(self):
        self.RAW_DATA_PATH = self.DATA_DIR / "arxiv-metadata-oai-snapshot.json"
        self.EMBEDDING_MODEL_PATH = self.MODELS_DIR / "embeddings" / self.EMBEDDING_MODEL_NAME
        self.RERANKER_MODEL_PATH = self.MODELS_DIR / "reranker" / self.RERANKER_MODEL_NAME
        self.QWEN_RERANKER_MODEL_PATH = self.MODELS_DIR / "reranker" / self.QWEN_RERANKER_MODEL_NAME
        self.VECTOR_INDEX_PATH = self.STORAGE_DIR / "arxiv_full_ivf.faiss"
        self.DB_FILE = self.STORAGE_DIR / "arxiv_metadata.db"

        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
