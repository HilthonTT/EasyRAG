"""Pipelines: wire components together into an end-to-end RAG flow."""

from easyrag.pipelines.base import BasePipeline
from easyrag.pipelines.hyde import HyDEPipeline
from easyrag.pipelines.ircot import IRCOTPipeline
from easyrag.pipelines.iterative import IterativeRAGPipeline
from easyrag.pipelines.multi_query import MultiQueryPipeline
from easyrag.pipelines.naive import NaiveRAGPipeline
from easyrag.pipelines.self_ask import SelfAskPipeline

PIPELINES: dict[str, type[BasePipeline]] = {
    "naive": NaiveRAGPipeline,
    "hyde": HyDEPipeline,
    "multi_query": MultiQueryPipeline,
    "iterative": IterativeRAGPipeline,
    "self_ask": SelfAskPipeline,
    "ircot": IRCOTPipeline,
}

__all__ = [
    "PIPELINES",
    "BasePipeline",
    "NaiveRAGPipeline",
    "IterativeRAGPipeline",
    "HyDEPipeline",
    "MultiQueryPipeline",
    "SelfAskPipeline",
    "IRCOTPipeline",
]
