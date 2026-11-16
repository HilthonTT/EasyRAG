import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.evaluation import EvalExample, LLMJudge, evaluate
from easyrag.generators import EchoGenerator, Usage
from easyrag.generators.anthropic import AnthropicGenerator
from easyrag.generators.base import BaseGenerator
from easyrag.generators.ollama import OllamaGenerator
from easyrag.generators.openai import OpenAIGenerator


class CountingGenerator(BaseGenerator):
    """Records 10 input and 2 output tokens per call."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str, system: str | None = None) -> str:
        self._record(10, 2)
        return self.reply


def test_usage_arithmetic() -> None:
    assert Usage(2, 30, 4) - Usage(1, 10, 1) == Usage(1, 20, 3)
    assert Usage(1, 10, 1) + Usage(1, 20, 3) == Usage(2, 30, 4)


def test_record_counts_calls_and_treats_missing_tokens_as_zero() -> None:
    gen = EchoGenerator()
    before = replace(gen.usage)
    gen.generate("hi")
    list(gen.stream("hi"))  # the default stream goes through generate
    gen._record(None, None)
    assert gen.usage - before == Usage(calls=3)
    assert before == Usage()  # the snapshot is not changed by later calls


def test_anthropic_records_usage_including_cached_input() -> None:
    usage = SimpleNamespace(
        input_tokens=5, output_tokens=7, cache_creation_input_tokens=3, cache_read_input_tokens=None
    )
    response = SimpleNamespace(
        usage=usage, stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")]
    )
    gen = AnthropicGenerator.__new__(AnthropicGenerator)  # skip the SDK import and client
    gen.model, gen.effort, gen.max_tokens = "m", "low", 10
    gen.client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    )
    assert gen.generate("hi") == "ok"
    assert gen.usage == Usage(1, 8, 7)


def _openai(create: Any) -> OpenAIGenerator:
    gen = OpenAIGenerator.__new__(OpenAIGenerator)
    gen.model, gen.max_tokens = "m", 10
    gen.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return gen


def test_openai_records_usage_for_generate_and_stream() -> None:
    usage = SimpleNamespace(prompt_tokens=4, completion_tokens=6)

    def create(**kw: Any) -> Any:
        if not kw.get("stream"):
            message = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
        assert kw["stream_options"] == {"include_usage": True}
        delta = SimpleNamespace(delta=SimpleNamespace(content="ok"))
        return iter(
            [SimpleNamespace(choices=[delta], usage=None), SimpleNamespace(choices=[], usage=usage)]
        )

    gen = _openai(create)
    assert gen.generate("hi") == "ok"
    assert "".join(gen.stream("hi")) == "ok"
    assert gen.usage == Usage(2, 8, 12)


def test_ollama_records_usage_for_generate_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    final = {"message": {"content": "ok"}, "done": True, "prompt_eval_count": 3, "eval_count": 2}

    def urlopen(request: Any, timeout: float) -> io.BytesIO:
        if json.loads(request.data)["stream"]:
            lines = [{"message": {"content": "o"}, "done": False}, final | {"message": {}}]
            return io.BytesIO(b"\n".join(json.dumps(line).encode() for line in lines))
        return io.BytesIO(json.dumps(final).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    gen = OllamaGenerator()
    assert gen.generate("hi") == "ok"
    assert "".join(gen.stream("hi")) == "o"
    assert gen.usage == Usage(2, 6, 4)


def _pipeline(sample_documents: list[Document], generator: BaseGenerator) -> NaiveRAGPipeline:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"top_k": 1}}))
    pipeline.generator = generator
    pipeline.index(sample_documents)
    return pipeline


def test_evaluate_reports_usage_per_question_and_judge_separately(
    sample_documents: list[Document],
) -> None:
    # The judge shares the pipeline's generator, so only snapshots keep the two apart
    generator = CountingGenerator('{"score": 1, "reason": "ok"}')
    pipeline = _pipeline(sample_documents, generator)
    examples = [EvalExample("capital of France?", answers=["Paris"])] * 2

    report = evaluate(pipeline, examples, judge=LLMJudge(generator))

    assert [r.usage for r in report.results] == [Usage(1, 10, 2)] * 2
    assert [r.judge_usage for r in report.results] == [Usage(2, 20, 4)] * 2  # correct + faithful
    assert report.usage == Usage(2, 20, 4)
    assert report.judge_usage == Usage(4, 40, 8)
    data = report.to_dict()
    assert data["usage"]["judge"] == {"calls": 4, "input_tokens": 40, "output_tokens": 8}
    assert data["results"][0]["usage"]["calls"] == 1


def test_retrieval_only_eval_uses_no_llm(sample_documents: list[Document]) -> None:
    pipeline = _pipeline(sample_documents, CountingGenerator("x"))
    report = evaluate(pipeline, [EvalExample("capital?", sources=["france.txt"])], generate=False)
    assert report.usage == Usage()


def test_cli_eval_prints_usage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "france.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"question": "Capital?", "answers": ["Paris"]}), "utf-8")

    main(["eval", str(dataset), "--docs", str(docs)])
    assert "generation: 1 LLM calls" in capsys.readouterr().out

    main(["eval", str(dataset), "--docs", str(docs), "--retrieval-only"])
    assert "LLM calls" not in capsys.readouterr().out
