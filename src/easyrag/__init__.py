"""EasyRAG: a simple, modular Retrieval-Augmented Generation toolkit."""

from easyrag.config import Config
from easyrag.pipelines import NaiveRAGPipeline
from easyrag.schema import Chunk, Document, RetrievedChunk

__all__ = ["Chunk", "Config", "Document", "NaiveRAGPipeline", "RetrievedChunk"]
__version__ = "0.1.0"
