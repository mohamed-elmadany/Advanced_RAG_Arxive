from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .paths import EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH


def download_embedding_model():
    print("Downloading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    EMBEDDING_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(EMBEDDING_MODEL_PATH))
    print("Saved to:", EMBEDDING_MODEL_PATH)


def download_reranker_model():
    print("Downloading reranker model...")
    model_name = "BAAI/bge-reranker-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    RERANKER_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(RERANKER_MODEL_PATH)
    model.save_pretrained(RERANKER_MODEL_PATH)

    print("Saved to:", RERANKER_MODEL_PATH)


if __name__ == "__main__":
    download_embedding_model()
    # optional
    download_reranker_model()
