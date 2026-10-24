"""Pipelines: wire components together into an end-to-end RAG flow."""

from easyrag.pipelines.base import BasePipeline
from easyrag.pipelines.naive import NaiveRAGPipeline

__all__ = ["BasePipeline", "NaiveRAGPipeline"]
