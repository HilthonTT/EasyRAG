from pathlib import Path

import pytest

from easyrag import Config, Document, HyDEPipeline, NaiveRAGPipeline
from easyrag.rerankers import BaseReranker
from easyrag.schema import RetrievedChunk


class _RecordingReranker(BaseReranker):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return results[:top_k]


def test_naive_pipeline_end_to_end(sample_documents: list[Document]) -> None:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"top_k": 2}}))
    assert pipeline.index(sample_documents) == 3

    response = pipeline.run("What is the capital of France?")

    assert len(response.contexts) == 2
    assert "Paris" in response.contexts[0].chunk.content
    # EchoGenerator returns the prompt, so the retrieved context must be in it.
    assert "Paris is the capital of France." in response.answer


def test_hyde_reranks_against_the_original_question(sample_documents: list[Document]) -> None:
    reranker = _RecordingReranker()
    pipeline = HyDEPipeline.from_config(Config(), reranker=reranker)
    pipeline.index(sample_documents)

    pipeline.run("What is the capital of France?")

    # Retrieval searches with the hypothetical passage; the reranker must see only the question.
    assert reranker.queries == ["What is the capital of France?"]


def test_load_refuses_a_non_empty_pipeline(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    pipeline = NaiveRAGPipeline.from_config(Config())
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)

    with pytest.raises(RuntimeError, match="already holds 3 chunks"):
        pipeline.load(tmp_path)

    fresh = NaiveRAGPipeline.from_config(Config())
    assert fresh.load(tmp_path) == 3
    assert [c.id for c in fresh.chunks] == [c.id for c in pipeline.chunks]


def test_reindexing_the_same_documents_adds_no_duplicates() -> None:
    config = Config()
    config.retriever.method = "vector"
    pipeline = NaiveRAGPipeline.from_config(config)
    docs = [
        Document(content="Paris is the capital of France."),
        Document(content="Berlin is the capital of Germany."),
    ]
    assert pipeline.index(docs) == 2
    assert pipeline.index(docs[:1]) == 0
    assert len(pipeline.chunks) == 2
    ids = [r.chunk.id for r in pipeline.retrieve("capital of France")]
    assert len(ids) == len(set(ids)) == 2
