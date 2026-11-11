"""Pipelines: wire components together into an end-to-end RAG flow."""

from easyrag.pipelines.agent import AgentPipeline
from easyrag.pipelines.base import BasePipeline
from easyrag.pipelines.hyde import HyDEPipeline
from easyrag.pipelines.ircot import IRCOTPipeline
from easyrag.pipelines.iterative import IterativeRAGPipeline
from easyrag.pipelines.multi_query import MultiQueryPipeline
from easyrag.pipelines.naive import NaiveRAGPipeline
from easyrag.pipelines.self_ask import SelfAskPipeline
from easyrag.pipelines.self_rag import SelfRAGPipeline
from easyrag.pipelines.step_back import StepBackPipeline

PIPELINES: dict[str, type[BasePipeline]] = {
    "naive": NaiveRAGPipeline,
    "hyde": HyDEPipeline,
    "multi_query": MultiQueryPipeline,
    "step_back": StepBackPipeline,
    "iterative": IterativeRAGPipeline,
    "self_ask": SelfAskPipeline,
    "ircot": IRCOTPipeline,
    "self_rag": SelfRAGPipeline,
    "agent": AgentPipeline,
}

__all__ = [
    "PIPELINES",
    "AgentPipeline",
    "BasePipeline",
    "NaiveRAGPipeline",
    "IterativeRAGPipeline",
    "HyDEPipeline",
    "MultiQueryPipeline",
    "SelfAskPipeline",
    "SelfRAGPipeline",
    "StepBackPipeline",
    "IRCOTPipeline",
]
