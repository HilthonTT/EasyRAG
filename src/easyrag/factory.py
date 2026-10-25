from __future__ import annotations

from typing import Any

from easyrag.config import (
    Config,
    EmbeddingConfig,
    GeneratorConfig,
    RerankerConfig,
    RetrieverConfig,
    SplitterConfig,
)
from easyrag.embeddings import BaseEmbeddings, HashingEmbeddings
from easyrag.generators import BaseGenerator, EchoGenerator
from easyrag.rerankers import BaseReranker
from easyrag.retrievers import BaseRetriever, BM25Retriever, HybridRetriever, VectorRetriever
from easyrag.splitters import BaseSplitter, CharacterSplitter, RecursiveCharacterSplitter


def build_splitter(cfg: SplitterConfig) -> BaseSplitter:
    if cfg.method == "recursive":
        return RecursiveCharacterSplitter(cfg.chunk_size, cfg.chunk_overlap)
    if cfg.method == "character":
        return CharacterSplitter(cfg.chunk_size, cfg.chunk_overlap)
    raise ValueError(f"Unknown splitter.method: {cfg.method!r}")


def build_embeddings(cfg: EmbeddingConfig) -> BaseEmbeddings:
    if cfg.provider in ("hash", "fake"):
        return HashingEmbeddings()
    if cfg.provider in ("sentence_transformer", "huggingface"):
        from easyrag.embeddings.sentence_transformer import (
            DEFAULT_MODEL,
            SentenceTransformerEmbeddings,
        )

        return SentenceTransformerEmbeddings(cfg.model or DEFAULT_MODEL, batch_size=cfg.batch_size)
    if cfg.provider == "openai":
        from easyrag.embeddings.openai import DEFAULT_MODEL as OPENAI_MODEL
        from easyrag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(cfg.model or OPENAI_MODEL)

    raise ValueError(f"Unknown embedding.provider: {cfg.provider!r}")


def build_retriever(cfg: RetrieverConfig, embeddings: BaseEmbeddings) -> BaseRetriever:
    if cfg.method == "vector":
        return VectorRetriever(embeddings, top_k=cfg.top_k)
    if cfg.method == "bm25":
        return BM25Retriever(top_k=cfg.top_k)
    if cfg.method == "hybrid":
        return HybridRetriever(
            [BM25Retriever(), VectorRetriever(embeddings)], top_k=cfg.top_k, rrf_k=cfg.rrf_k
        )
    raise ValueError(f"Unknown retriever.method: {cfg.method!r}")


def build_reranker(cfg: RerankerConfig) -> BaseReranker | None:
    if cfg.provider in ("", "none"):
        return None
    if cfg.provider == "cross_encoder":
        from easyrag.rerankers.cross_encoder import DEFAULT_MODEL, CrossEncoderReranker

        return CrossEncoderReranker(cfg.model or DEFAULT_MODEL)
    raise ValueError(f"Unknown reranker.provider: {cfg.provider!r}")


def build_generator(cfg: GeneratorConfig) -> BaseGenerator:
    if cfg.provider == "echo":
        return EchoGenerator()
    if cfg.provider == "anthropic":
        from easyrag.generators.anthropic import DEFAULT_MODEL, AnthropicGenerator

        return AnthropicGenerator(cfg.model or DEFAULT_MODEL, cfg.effort, cfg.max_tokens)
    if cfg.provider == "openai":
        from easyrag.generators.openai import DEFAULT_MODEL as OPENAI_MODEL
        from easyrag.generators.openai import OpenAIGenerator

        return OpenAIGenerator(cfg.model or OPENAI_MODEL, cfg.max_tokens, cfg.base_url or None)
    if cfg.provider == "ollama":
        from easyrag.generators.ollama import DEFAULT_BASE_URL, OllamaGenerator
        from easyrag.generators.ollama import DEFAULT_MODEL as OLLAMA_MODEL

        return OllamaGenerator(cfg.model or OLLAMA_MODEL, cfg.base_url or DEFAULT_BASE_URL)
    raise ValueError(f"Unknown generator.provider: {cfg.provider!r}")


def build_components(config: Config) -> dict[str, Any]:
    """Keyword arguments for any `BasePipeline` subclass constructor."""
    return {
        "retriever": build_retriever(config.retriever, build_embeddings(config.embedding)),
        "generator": build_generator(config.generator),
        "splitter": build_splitter(config.splitter),
        "reranker": build_reranker(config.reranker),
        "top_k": config.retriever.top_k,
        "fetch_k": config.reranker.fetch_k,
    }
