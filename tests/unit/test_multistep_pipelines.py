import pytest

from easyrag import Config, Document, IRCOTPipeline, IterativeRAGPipeline, SelfAskPipeline
from easyrag.factory import build_pipeline
from easyrag.generators import BaseGenerator
from easyrag.pipelines import PIPELINES
from easyrag.pipelines.base import merge_contexts
from easyrag.schema import Chunk, RetrievedChunk


class _ScriptedGenerator(BaseGenerator):
    """Returns the queued outputs in order and records every prompt."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


@pytest.mark.parametrize("pipeline_type", sorted(PIPELINES))
def test_every_registered_pipeline_runs(
    pipeline_type: str, sample_documents: list[Document]
) -> None:
    pipeline = build_pipeline(Config.from_dict({"pipeline": {"type": pipeline_type}}))
    pipeline.index(sample_documents)

    response = pipeline.run("What is the capital of France?")

    assert response.answer
    assert any("Paris" in c.chunk.content for c in response.contexts)


def test_self_ask_answers_follow_ups_from_retrieval(sample_documents: list[Document]) -> None:
    generator = _ScriptedGenerator(
        [
            # The model runs on and invents an intermediate answer; it must be discarded.
            "Yes.\nFollow up: What is the capital of Germany?\nIntermediate answer: Munich",
            "Berlin [1]",
            "So the final answer is: Berlin",
        ]
    )
    pipeline = SelfAskPipeline.from_config(Config(), generator=generator)
    pipeline.index(sample_documents)

    response = pipeline.run("Which city is the capital of the country where Germans live?")

    assert response.answer == "Berlin"
    assert response.metadata["steps"] == [
        {"follow_up": "What is the capital of Germany?", "answer": "Berlin [1]"}
    ]
    assert any("Berlin" in c.chunk.content for c in response.contexts)
    assert "Intermediate answer: Berlin [1]" in generator.prompts[-1]
    assert "Munich" not in generator.prompts[-1]


def test_self_ask_falls_back_to_a_final_answer(sample_documents: list[Document]) -> None:
    generator = _ScriptedGenerator(
        ["Follow up: What is the capital of France?", "Paris", "Paris it is"]
    )
    pipeline = SelfAskPipeline.from_config(Config(), generator=generator, max_iterations=1)
    pipeline.index(sample_documents)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris it is"
    assert "Follow up: What is the capital of France?" in generator.prompts[-1]


def test_self_ask_parsers() -> None:
    assert SelfAskPipeline._parse_follow_up("Yes.\nFollow up: Who is X?\n") == "Who is X?"
    assert SelfAskPipeline._parse_follow_up("No follow up needed.") is None
    assert SelfAskPipeline._parse_final_answer("So the final answer is: 42") == "42"
    assert SelfAskPipeline._parse_final_answer("Follow up: Who?") is None


def test_ircot_searches_with_each_thought(sample_documents: list[Document]) -> None:
    generator = _ScriptedGenerator(
        ["Python is a programming language. It is popular.", "So the answer is: a language"]
    )
    pipeline = IRCOTPipeline.from_config(
        Config.from_dict({"retriever": {"top_k": 1}}), generator=generator
    )
    pipeline.index(sample_documents)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "a language"
    # Only the first sentence is kept, and searching with it pulled in the Python passage.
    assert response.metadata["thoughts"][0] == "Python is a programming language."
    sources = [c.chunk.source for c in response.contexts]
    assert "france.txt" in sources and "python.txt" in sources


def test_ircot_generates_a_final_answer_without_the_marker(
    sample_documents: list[Document],
) -> None:
    generator = _ScriptedGenerator(["Paris is in France.", "It is the capital.", "Paris"])
    pipeline = IRCOTPipeline.from_config(Config(), generator=generator, max_iterations=2)
    pipeline.index(sample_documents)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris"
    assert response.metadata["thoughts"] == ["Paris is in France.", "It is the capital."]
    assert "Reasoning so far:\nParis is in France.\nIt is the capital." in generator.prompts[-1]


def test_ircot_extract_answer() -> None:
    assert IRCOTPipeline._extract_answer("So the answer is: Paris") == "Paris"
    assert IRCOTPipeline._extract_answer("so the answer is:  Paris.\n") == "Paris."
    assert IRCOTPipeline._extract_answer("So the answer is unclear.") is None


def test_iterative_records_each_round(sample_documents: list[Document]) -> None:
    pipeline = IterativeRAGPipeline.from_config(Config(), max_iterations=2)
    pipeline.index(sample_documents)

    response = pipeline.run("What is the capital of France?")

    assert len(response.metadata["iterations"]) == 2


def test_merge_contexts_dedupes_and_keeps_best_score() -> None:
    a, b = Chunk("a", "d1"), Chunk("b", "d2")
    merged = merge_contexts(
        [RetrievedChunk(a, 0.2), RetrievedChunk(b, 0.5)], [RetrievedChunk(a, 0.9)]
    )
    assert [(r.chunk.id, r.score) for r in merged] == [(a.id, 0.9), (b.id, 0.5)]
