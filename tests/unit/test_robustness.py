import io
import json
import threading
import time
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cache import CachedEmbeddings, CachedGenerator
from easyrag.embeddings.base import BaseEmbeddings
from easyrag.evaluation import EvalExample, LLMJudge, evaluate
from easyrag.factory import build_components, build_judge
from easyrag.generators import Usage
from easyrag.generators.base import BaseGenerator, track_usage
from easyrag.generators.ollama import OllamaGenerator

# --- Ollama retries ---------------------------------------------------------------------------


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x/api/chat", code, "error", headers, io.BytesIO())


def _reply(stream: bool) -> io.BytesIO:
    final = {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 1, "eval_count": 1}
    return io.BytesIO(json.dumps(final).encode() + (b"\n" if stream else b""))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr("easyrag.generators.ollama.time.sleep", waited.append)
    return waited


def _fail_then_succeed(monkeypatch: pytest.MonkeyPatch, failures: list[Exception]) -> list[int]:
    attempts: list[int] = []

    def urlopen(request: Any, timeout: float) -> io.BytesIO:
        attempts.append(1)
        if failures:
            raise failures.pop(0)
        return _reply(json.loads(request.data)["stream"])

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return attempts


@pytest.mark.parametrize(
    "error",
    [_http_error(503), _http_error(429), urllib.error.URLError("refused"), TimeoutError()],
)
def test_ollama_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], error: Exception
) -> None:
    attempts = _fail_then_succeed(monkeypatch, [error, error])
    gen = OllamaGenerator(max_retries=2)
    assert gen.generate("hi") == "ok"
    assert len(attempts) == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0] > 0  # exponential backoff
    assert gen.usage == Usage(1, 1, 1)  # failed attempts are not calls


def test_ollama_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = _fail_then_succeed(monkeypatch, [_http_error(503)] * 5)
    with pytest.raises(urllib.error.HTTPError):
        OllamaGenerator(max_retries=1).generate("hi")
    assert len(attempts) == 2


def test_ollama_does_not_retry_client_errors(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = _fail_then_succeed(monkeypatch, [_http_error(404)])  # e.g. model not pulled
    with pytest.raises(urllib.error.HTTPError):
        OllamaGenerator().generate("hi")
    assert len(attempts) == 1 and not sleeps


def test_ollama_honours_retry_after(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    _fail_then_succeed(monkeypatch, [_http_error(429, retry_after="7")])
    OllamaGenerator().generate("hi")
    assert sleeps == [7.0]


def test_ollama_stream_retries_before_the_first_piece(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = _fail_then_succeed(monkeypatch, [_http_error(502)])
    assert "".join(OllamaGenerator().stream("hi")) == "ok"
    assert len(attempts) == 2


# --- caches -----------------------------------------------------------------------------------


class CountingEmbeddings(BaseEmbeddings):
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded.append(text)
        return [float(len(text)), -1.0]  # differs from the document vector on purpose


class CountingGenerator(BaseGenerator):
    def __init__(self, reply: str = "answer", delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return f"counting:{self.reply}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        time.sleep(self.delay)
        self.prompts.append(prompt)
        self._record(10, 2)
        return self.reply


def test_cached_embeddings_embed_each_text_once(tmp_path: Path) -> None:
    inner = CountingEmbeddings()
    cached = CachedEmbeddings(inner, tmp_path / "e.sqlite")

    first = cached.embed_documents(["a", "bb", "a"])
    assert first == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert inner.embedded == ["a", "bb"]  # the duplicate is embedded once
    assert cached.embed_documents(["bb", "ccc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert inner.embedded == ["a", "bb", "ccc"]
    assert (cached.hits, cached.misses) == (2, 3)


def test_cached_embeddings_keep_queries_apart_and_persist(tmp_path: Path) -> None:
    cached = CachedEmbeddings(CountingEmbeddings(), tmp_path / "e.sqlite")
    cached.embed_documents(["a"])
    assert cached.embed_query("a") == [1.0, -1.0]  # not the cached document vector

    inner = CountingEmbeddings()
    reopened = CachedEmbeddings(inner, tmp_path / "e.sqlite")
    assert reopened.embed_query("a") == [1.0, -1.0]
    assert reopened.embed_documents(["a"]) == [[1.0, 1.0]]
    assert inner.embedded == []
    assert reopened.name == inner.name  # saved indexes stay compatible


def test_cached_generator_replays_answers_without_counting_calls(tmp_path: Path) -> None:
    inner = CountingGenerator()
    cached = CachedGenerator(inner, tmp_path / "llm.sqlite")

    assert cached.generate("q") == "answer"
    assert cached.generate("q") == "answer"
    assert "".join(cached.stream("q")) == "answer"
    assert inner.prompts == ["q"]
    assert cached.usage == Usage(1, 10, 2)

    cached.generate("q", system="be brief")  # a different system prompt is a different call
    other_model = CachedGenerator(CountingGenerator("other"), tmp_path / "llm.sqlite")
    assert other_model.generate("q") == "other"
    assert len(inner.prompts) == 2


def test_cached_generator_does_not_store_an_unfinished_stream(tmp_path: Path) -> None:
    class Streaming(CountingGenerator):
        def stream(self, prompt: str, system: str | None = None) -> Any:
            yield "par"
            yield "tial"

    inner = Streaming()
    cached = CachedGenerator(inner, tmp_path / "llm.sqlite")
    pieces = cached.stream("q")
    assert next(pieces) == "par"
    pieces.close()  # e.g. Ctrl+C in chat
    assert len(cached.cache) == 0
    assert "".join(cached.stream("q")) == "partial"
    assert cached.generate("q") == "partial"


def test_config_turns_the_caches_on(tmp_path: Path) -> None:
    raw = {"cache": {"dir": str(tmp_path), "embeddings": True, "llm": True}}
    components = build_components(Config.from_dict(raw))
    assert isinstance(components["generator"], CachedGenerator)
    assert isinstance(components["retriever"].retrievers[1].embeddings, CachedEmbeddings)
    assert isinstance(build_judge(Config.from_dict(raw)).generator, CachedGenerator)
    assert not isinstance(build_components(Config())["generator"], CachedGenerator)


def test_cached_pipeline_saves_and_loads_like_an_uncached_one(
    tmp_path: Path, sample_documents: list[Document]
) -> None:
    cached = Config.from_dict({"cache": {"dir": str(tmp_path / "cache"), "embeddings": True}})
    pipeline = NaiveRAGPipeline.from_config(cached)
    pipeline.index(sample_documents)
    pipeline.save(tmp_path / "index")

    plain = NaiveRAGPipeline.from_config(Config())
    plain.load(tmp_path / "index")
    query = "capital of France"
    assert [r.chunk.id for r in plain.retrieve(query)] == [
        r.chunk.id for r in pipeline.retrieve(query)
    ]


# --- parallel runs ----------------------------------------------------------------------------


def test_track_usage_counts_only_its_own_thread() -> None:
    gen = CountingGenerator()
    barrier = threading.Barrier(2)
    seen: list[Usage] = []

    def work(calls: int) -> None:
        with track_usage() as usage:
            barrier.wait()
            for _ in range(calls):
                gen.generate("q")
        seen.append(usage)

    threads = [threading.Thread(target=work, args=(n,)) for n in (1, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(u.calls for u in seen) == [1, 3]
    assert gen.usage == Usage(4, 40, 8)


def _pipeline(sample_documents: list[Document], generator: BaseGenerator) -> NaiveRAGPipeline:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"top_k": 1}}))
    pipeline.generator = generator
    pipeline.index(sample_documents)
    return pipeline


def test_parallel_evaluate_matches_sequential(sample_documents: list[Document]) -> None:
    # The judge shares the generator, and slow calls make the questions overlap.
    generator = CountingGenerator('{"score": 1, "reason": "ok"}', delay=0.01)
    pipeline = _pipeline(sample_documents, generator)
    questions = ["capital of France?", "capital of Germany?", "What is Python?"] * 3
    examples = [EvalExample(q, answers=["Paris"], sources=["france.txt"]) for q in questions]

    sequential = evaluate(pipeline, examples, judge=LLMJudge(generator))
    parallel = evaluate(pipeline, examples, judge=LLMJudge(generator), workers=4)

    assert parallel.to_dict() == sequential.to_dict()
    assert [r.example.question for r in parallel.results] == questions
    assert all(r.usage == Usage(1, 10, 2) for r in parallel.results)
    assert all(r.judge_usage == Usage(2, 20, 4) for r in parallel.results)


def test_batch_keeps_question_order(sample_documents: list[Document]) -> None:
    pipeline = _pipeline(sample_documents, CountingGenerator(delay=0.01))
    questions = [f"question {i}" for i in range(8)]
    assert [r.question for r in pipeline.batch(questions, workers=4)] == questions


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_must_be_positive(sample_documents: list[Document], workers: int) -> None:
    pipeline = _pipeline(sample_documents, CountingGenerator())
    with pytest.raises(ValueError, match="workers"):
        pipeline.batch(["q"], workers=workers)
    with pytest.raises(ValueError, match="workers"):
        evaluate(pipeline, [EvalExample("q")], workers=workers)
