"""Pipelines: wire components together into an end-to-end RAG flow."""

from easyrag.pipelines.base import BasePipeline
from easyrag.pipelines.naive import NaiveRAGPipeline

PIPELINES: dict[str, type[BasePipeline]] = {}

__all__ = ["PIPELINES", "BasePipeline", "NaiveRAGPipeline"]
