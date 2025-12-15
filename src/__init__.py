"""
RAG Analysis Project - Source Package
Evaluating Retrieval-Augmented Generation for Hallucination Reduction
"""

__version__ = "1.0.0"
__author__ = "RAG Analysis Team"

from .data_loader import HotpotQALoader, chunk_text
from .rag_pipeline import VectorStore, LLMGenerator, RAGPipeline
from .evaluator import Evaluator, exact_match, token_f1
from .utils import setup_logging, set_random_seed, load_config, save_config, get_device

__all__ = [
    "HotpotQALoader",
    "chunk_text",
    "VectorStore",
    "LLMGenerator",
    "RAGPipeline",
    "Evaluator",
    "exact_match",
    "token_f1",
    "setup_logging",
    "set_random_seed",
    "load_config",
    "save_config",
    "get_device",
]
