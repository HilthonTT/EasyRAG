from pathlib import Path

from easyrag.evaluation import load_examples
from easyrag.evaluation.synthetic import draft_examples, write_examples
from easyrag.generators.base import BaseGenerator
from easyrag.schema import Chunk

PASSAGE = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. " * 4


class QueueGenerator(BaseGenerator):
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.replies.pop(0)


def _chunk(source: str, text: str = PASSAGE) -> Chunk:
    return Chunk(content=text, document_id="d", metadata={"source": source})


def test_keeps_answers_found_in_passage_and_drops_the_rest(tmp_path: Path) -> None:
    generator = QueueGenerator(
        [
            '{"question": "When was the Eiffel Tower finished?", "answer": "1889"}',
            '{"question": "Who designed it?", "answer": "Gustave Eiffel"}',  # not in passage
            '{"skip": true}',
            "not json",
        ]
    )
    chunks = [_chunk("docs/paris.txt") for _ in range(4)]
    examples = draft_examples(chunks, generator, per_source=4)

    assert len(examples) == 1
    assert examples[0].answers == ["1889"]
    assert examples[0].sources == ["paris.txt"]

    path = tmp_path / "eval.jsonl"
    write_examples(examples, path)
    assert load_examples(path)[0].question == "When was the Eiffel Tower finished?"


def test_respects_per_source_and_min_chars() -> None:
    reply = '{"question": "Q%d?", "answer": "1889"}'
    generator = QueueGenerator([reply % i for i in range(10)])
    chunks = [_chunk("a.txt") for _ in range(5)] + [_chunk("b.txt", "too short")]
    examples = draft_examples(chunks, generator, per_source=2)
    assert [e.sources for e in examples] == [["a.txt"], ["a.txt"]]
