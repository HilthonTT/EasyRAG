"""rag: a simple, modular Retrieval-Augmented Generation toolkit."""

from easyrag.config import Config
from easyrag.conversation import Conversation
from easyrag.factory import build_pipeline
from easyrag.pipelines import (
    AgentPipeline,
    BasePipeline,
    HyDEPipeline,
    IRCOTPipeline,
    IterativeRAGPipeline,
    MultiQueryPipeline,
    NaiveRAGPipeline,
    SelfAskPipeline,
    SelfRAGPipeline,
    StepBackPipeline,
)
from easyrag.schema import Chunk, Document, RAGResponse, RetrievedChunk

__all__ = [
    "AgentPipeline",
    "BasePipeline",
    "Chunk",
    "Config",
    "Conversation",
    "Document",
    "HyDEPipeline",
    "IRCOTPipeline",
    "IterativeRAGPipeline",
    "MultiQueryPipeline",
    "NaiveRAGPipeline",
    "RAGResponse",
    "RetrievedChunk",
    "SelfAskPipeline",
    "SelfRAGPipeline",
    "StepBackPipeline",
    "build_pipeline",
]
__version__ = "0.1.0"
