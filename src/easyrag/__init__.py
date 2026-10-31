"""rag: a simple, modular Retrieval-Augmented Generation toolkit."""

from easyrag.config import Config
from easyrag.factory import build_pipeline
from easyrag.pipelines import (
    BasePipeline,
    HyDEPipeline,
    IRCOTPipeline,
    IterativeRAGPipeline,
    NaiveRAGPipeline,
    SelfAskPipeline,
)
from easyrag.schema import Chunk, Document, RAGResponse, RetrievedChunk

__all__ = [
    "BasePipeline",
    "Chunk",
    "Config",
    "Document",
    "HyDEPipeline",
    "IRCOTPipeline",
    "IterativeRAGPipeline",
    "NaiveRAGPipeline",
    "RAGResponse",
    "RetrievedChunk",
    "SelfAskPipeline",
    "build_pipeline",
]
__version__ = "0.1.0"
