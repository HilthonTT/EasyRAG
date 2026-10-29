"""rag: a simple, modular Retrieval-Augmented Generation toolkit."""

from easyrag.config import Config
from easyrag.factory import build_pipeline
from easyrag.pipelines import BasePipeline, HyDEPipeline, IterativeRAGPipeline, NaiveRAGPipeline
from easyrag.schema import Chunk, Document, RAGResponse, RetrievedChunk

__all__ = [
    "BasePipeline",
    "Chunk",
    "Config",
    "Document",
    "HyDEPipeline",
    "IterativeRAGPipeline",
    "NaiveRAGPipeline",
    "RAGResponse",
    "RetrievedChunk",
    "build_pipeline",
]
__version__ = "0.1.0"
