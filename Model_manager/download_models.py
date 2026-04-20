from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import config


def download_embedding_model():
    print("Downloading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    config.EMBEDDING_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(config.EMBEDDING_MODEL_PATH))
    print("Saved to:", config.EMBEDDING_MODEL_PATH)


def download_reranker_model():
    print("Downloading reranker model...")
    tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL_NAME)

    config.RERANKER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(config.RERANKER_MODEL_PATH)
    model.save_pretrained(config.RERANKER_MODEL_PATH)

    print("Saved to:", config.RERANKER_MODEL_PATH)


def download_qwen_reranker_model():
    print("Downloading qwen reranker...")
    model = CrossEncoder(model_name_or_path=config.QWEN_RERANKER_MODEL_NAME, revision="refs/pr/24")
    config.QWEN_RERANKER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(config.QWEN_RERANKER_MODEL_PATH))
    print("Saved to:", config.QWEN_RERANKER_MODEL_PATH)


if __name__ == "__main__":
    download_embedding_model()
    download_qwen_reranker_model()
