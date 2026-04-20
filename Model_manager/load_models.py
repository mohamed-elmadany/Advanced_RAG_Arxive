from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from .download_models import download_embedding_model,download_reranker_model, download_qwen_reranker_model
from core.config import config
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model = None
_reranker_model = None
_qwen_reranker_model=None

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(str(config.EMBEDDING_MODEL_PATH)).to(device)
        except Exception as e:
            print(F"\nmodel:{config.EMBEDDING_MODEL_NAME} not found..")
            print(F"\ndownloading model:{config.EMBEDDING_MODEL_NAME} .. ")
            download_embedding_model()
            _embedding_model = SentenceTransformer(str(config.EMBEDDING_MODEL_PATH)).to(device)

    return _embedding_model

def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        try:
            _reranker_model = CrossEncoder(str(config.RERANKER_MODEL_PATH)).to(device)
        except Exception as e:
            print(F"\nmodel:{config.RERANKER_MODEL_NAME} not found..")
            print(F"\ndownloading model:{config.RERANKER_MODEL_NAME} ")
            download_reranker_model()
            _reranker_model = CrossEncoder(str(config.RERANKER_MODEL_PATH)).to(device)
            

    return _reranker_model
def get_qwen_reranker():
    global _qwen_reranker_model
    if _qwen_reranker_model is None:
        try:
            _qwen_reranker_model = CrossEncoder(str(config.QWEN_RERANKER_MODEL_PATH)).to(device)
        except Exception as e:
            print(F"\nmodel:{config.QWEN_RERANKER_MODEL_NAME} not found..")
            print(F"\ndownloading model:{config.QWEN_RERANKER_MODEL_NAME} ")
            download_qwen_reranker_model()
            _qwen_reranker_model = CrossEncoder(str(config.QWEN_RERANKER_MODEL_PATH)).to(device)
    return _qwen_reranker_model
            
            

__all__ = ["get_embedding_model", "get_reranker_model","get_qwen_reranker"]
