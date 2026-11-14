"""Multi-turn chat: follow-ups are rewritten into standalone questions before searching."""

from pathlib import Path

import pytest

from easyrag import Config, Conversation, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.generators import BaseGenerator

DOCS = [
    Document("Nimbus is a weather app led by Ada Okafor.", {"source": "nimbus.md"}),
    Document("Skua is a sailing app led by Ben Ito.", {"source": "skua.md"}),
]


class _ScriptedGenerator(BaseGenerator):
    """Returns the queued outputs in order and records every (system, prompt) pair."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str | None, str]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((system, prompt))
        return self.outputs.pop(0)


def _conversation(generator: BaseGenerator, **kwargs: int) -> Conversation:
    config = Config.from_dict({"retriever": {"top_k": 1}})
    pipeline = NaiveRAGPipeline.from_config(config, generator=generator)
    pipeline.index(DOCS)
    return Conversation(pipeline, **kwargs)


def test_first_question_is_searched_as_is() -> None:
    generator = _ScriptedGenerator(["Nimbus is a weather app [1]."])
    chat = _conversation(generator)

    response = chat.run("What is Nimbus?")

    assert response.metadata["standalone_question"] == "What is Nimbus?"
    assert len(generator.calls) == 1  # no rewrite call on the first turn
    assert chat.history[0].answer == "Nimbus is a weather app [1]."


def test_follow_up_is_rewritten_with_the_history() -> None:
    generator = _ScriptedGenerator(
        [
            "Nimbus is a weather app [1].",
            "Standalone question: Who leads Nimbus?",
            "Ada Okafor [1].",
        ]
    )
    chat = _conversation(generator)
    chat.run("What is Nimbus?")

    response = chat.run("Who leads it?")

    system, prompt = generator.calls[1]
    assert prompt == "Who leads it?"
    assert system is not None
    assert "User: What is Nimbus?\nAssistant: Nimbus is a weather app [1]." in system
    assert response.question == "Who leads it?"
    assert response.metadata["standalone_question"] == "Who leads Nimbus?"
    assert response.metadata["searched"] is True
    assert response.contexts[0].chunk.source == "nimbus.md"
    assert "Who leads Nimbus?" in generator.calls[2][1]  # the answer prompt


def test_history_is_limited_to_max_turns_and_reset_clears_it() -> None:
    generator = _ScriptedGenerator(["a1", "q2", "a2", "q3", "a3", "q4"])
    chat = _conversation(generator, max_turns=1)
    chat.run("first")
    chat.run("second")
    chat.run("third")

    assert chat.standalone_question("fourth") == "q4"
    system = generator.calls[-1][0] or ""
    assert "q3" in system and "q2" not in system

    chat.reset()
    assert chat.standalone_question("fresh") == "fresh"


def test_zero_max_turns_disables_rewriting() -> None:
    generator = _ScriptedGenerator(["a1", "a2"])
    chat = _conversation(generator, max_turns=0)
    chat.run("What is Nimbus?")

    assert chat.run("Who leads it?").metadata["standalone_question"] == "Who leads it?"
    assert len(generator.calls) == 2


def test_empty_rewrite_falls_back_to_the_question() -> None:
    generator = _ScriptedGenerator(["a1", "  \n", "a2"])
    chat = _conversation(generator)
    chat.run("What is Nimbus?")

    assert chat.run("Who leads it?").metadata["standalone_question"] == "Who leads it?"


@pytest.mark.parametrize("marker", ["NO_SEARCH", "`NO_SEARCH`", "No search.", "no-search"])
def test_no_search_follow_up_is_answered_from_the_conversation(marker: str) -> None:
    long_answer = "Nimbus is a weather app [1]. " + "It shows forecasts. " * 50
    generator = _ScriptedGenerator([long_answer, marker, "Nimbus is a weather app."])
    chat = _conversation(generator)
    chat.run("What is Nimbus?")

    response = chat.run("Make that shorter")

    system, prompt = generator.calls[2]
    assert prompt == "Make that shorter"
    assert system is not None and long_answer.strip() in system  # the last answer is not cut
    assert response.contexts == []
    assert response.metadata == {"standalone_question": None, "searched": False}
    assert chat.history[-1].standalone is None
    assert "User: Make that shorter\n" in chat._format(chat.history)


def test_unfinished_answer_is_not_kept() -> None:
    chat = _conversation(_ScriptedGenerator(["a1"]))
    _, _, answer = chat.stream("What is Nimbus?")
    assert chat.history == []
    list(answer)
    assert [t.answer for t in chat.history] == ["a1"]


def test_cli_chat_keeps_history_with_echo_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "nimbus.md").write_text(DOCS[0].content, encoding="utf-8")
    inputs = iter(["What is Nimbus?", "/new", "Who leads it?", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main(["chat", "--docs", str(docs)])

    out = capsys.readouterr().out
    assert "Started a new conversation." in out
    assert "searching for" not in out  # the echo generator returns follow-ups unchanged
