from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from .paths import EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model = None
_reranker_model = None

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
    
    return _embedding_model

def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder(str(RERANKER_MODEL_PATH)).to(device)

    return _reranker_model

__all__ = ["get_embedding_model", "get_reranker_model"]
