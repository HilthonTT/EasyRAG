"""Step-back, Self-RAG and agent pipelines, driven by scripted LLM replies."""

from easyrag import AgentPipeline, Config, Document, SelfRAGPipeline, StepBackPipeline
from easyrag.generators import BaseGenerator


class _ScriptedGenerator(BaseGenerator):
    """Returns the queued outputs in order and records every (system, prompt) pair."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str | None, str]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((system, prompt))
        return self.outputs.pop(0)


DOCS = [
    Document("Part-time staff accrue vacation pro rata to hours worked.", {"source": "leave.md"}),
    Document("Full-time employees get 25 vacation days a year.", {"source": "leave2.md"}),
    Document("Paris is the capital of France.", {"source": "france.txt"}),
    Document("Berlin is the capital of Germany.", {"source": "germany.txt"}),
]


def _build(cls: type, generator: BaseGenerator, top_k: int = 2, **kwargs: object) -> object:
    config = Config.from_dict({"retriever": {"top_k": top_k}})
    pipeline = cls.from_config(config, generator=generator, **kwargs)
    pipeline.index(DOCS)
    return pipeline


# --- step-back -----------------------------------------------------------------------------


def test_step_back_searches_the_question_and_the_step_back_question() -> None:
    generator = _ScriptedGenerator(
        ["Step-back question: How is vacation calculated for part-time staff?", "answer"]
    )
    pipeline = _build(StepBackPipeline, generator, top_k=3)

    response = pipeline.run("How many vacation days do I get on three days a week?")

    assert response.metadata["queries"] == [
        "How many vacation days do I get on three days a week?",
        "How is vacation calculated for part-time staff?",
    ]
    assert any(c.chunk.source == "leave.md" for c in response.contexts)
    assert generator.calls[0][1] == "How many vacation days do I get on three days a week?"


def test_step_back_ignores_num_queries() -> None:
    pipeline = StepBackPipeline.from_config(Config(), num_queries=5)
    assert pipeline.num_queries == 1


# --- Self-RAG ------------------------------------------------------------------------------


def test_self_rag_answers_directly_when_no_retrieval_is_needed() -> None:
    generator = _ScriptedGenerator(["Retrieve: no", "Hello!"])
    pipeline = _build(SelfRAGPipeline, generator)

    response = pipeline.run("Hi there")

    assert response.answer == "Hello!"
    assert response.contexts == []
    assert response.metadata == {"retrieved": False, "attempts": []}


def test_self_rag_drops_irrelevant_passages() -> None:
    generator = _ScriptedGenerator(["Retrieve: yes", "Relevant: 1", "Paris [1]", "Supported: yes"])
    pipeline = _build(SelfRAGPipeline, generator)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris [1]"
    assert [c.chunk.source for c in response.contexts] == ["france.txt"]
    assert response.metadata["attempts"] == [
        {
            "query": "What is the capital of France?",
            "retrieved": 2,
            "relevant": 1,
            "supported": True,
        }
    ]


def test_self_rag_rewrites_the_query_when_nothing_is_relevant() -> None:
    generator = _ScriptedGenerator(
        [
            "Retrieve: yes",
            "Relevant: none",
            "capital city of Germany",  # the rewrite
            "Relevant: 1",
            "Berlin [1]",
            "Supported: yes",
        ]
    )
    pipeline = _build(SelfRAGPipeline, generator)

    response = pipeline.run("Where does the German government sit?")

    assert response.answer == "Berlin [1]"
    attempts = response.metadata["attempts"]
    assert [a["query"] for a in attempts] == [
        "Where does the German government sit?",
        "capital city of Germany",
    ]
    rewrite_system = generator.calls[2][0] or ""
    assert "- Where does the German government sit?" in rewrite_system  # tried queries shown


def test_self_rag_retries_unsupported_answers_and_keeps_the_last() -> None:
    generator = _ScriptedGenerator(
        ["Retrieve: yes", "Relevant: 1", "Lyon [1]", "Supported: no", "France capital"]
        + ["Relevant: 1", "Still Lyon [1]", "Supported: no"]
    )
    pipeline = _build(SelfRAGPipeline, generator, max_iterations=2)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Still Lyon [1]"
    assert [a["supported"] for a in response.metadata["attempts"]] == [False, False]


def test_self_rag_unparseable_checks_fall_back_to_plain_rag() -> None:
    generator = _ScriptedGenerator(["hmm", "no idea", "Paris", "looks fine"])
    pipeline = _build(SelfRAGPipeline, generator)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris"
    assert len(response.contexts) == 2  # nothing filtered


def test_self_rag_answers_from_candidates_when_nothing_is_ever_relevant() -> None:
    generator = _ScriptedGenerator(["Retrieve: yes", "Relevant: none", "I don't know."])
    pipeline = _build(SelfRAGPipeline, generator, max_iterations=1)

    response = pipeline.run("What is the airspeed of a swallow?")

    assert response.answer == "I don't know."
    assert len(response.contexts) == 2


def test_self_rag_parse_relevant() -> None:
    parse = SelfRAGPipeline._parse_relevant
    assert parse("Relevant: 1, 3", 3) == {1, 3}
    assert parse("relevant: [2]", 3) == {2}
    assert parse("Relevant: none", 3) == set()
    assert parse("Relevant: 7", 3) is None  # out of range: unparseable, keep everything
    assert parse("passage one", 3) is None


# --- agent ---------------------------------------------------------------------------------


def test_agent_searches_then_finishes_with_matching_citation_numbers() -> None:
    generator = _ScriptedGenerator(
        [
            "Thought: I need France's capital.\nAction: search[capital of France]",
            "Thought: Now Germany.\nAction: search[capital of Germany]\nObservation: invented",
            "Thought: I know both.\nAction: finish[Paris [1] and Berlin [2]]",
        ]
    )
    pipeline = _build(AgentPipeline, generator, top_k=1)

    response = pipeline.run("What are the capitals of France and Germany?")

    assert response.answer == "Paris [1] and Berlin [2]"
    # Numbered in the order first found, so the citations match the contexts.
    assert [c.chunk.source for c in response.contexts] == ["france.txt", "germany.txt"]
    steps = response.metadata["steps"]
    assert [s["action"] for s in steps] == ["search", "search", "finish"]
    assert steps[0]["query"] == "capital of France"
    assert steps[1]["passages"] == [2]
    # The invented observation was cut; the real one (with passage text) was appended.
    last_prompt = generator.calls[-1][1]
    assert "invented" not in last_prompt
    assert "Observation:\n[2] (source: germany.txt)\nBerlin is the capital" in last_prompt


def test_agent_observation_refers_to_passages_shown_before() -> None:
    pipeline = _build(AgentPipeline, _ScriptedGenerator([]), top_k=2)
    results = pipeline.retrieve("capital of France")
    contexts = results[:1]

    numbers, observation = AgentPipeline._observe(contexts, results)

    assert numbers == [1, 2]
    assert observation.startswith("[1] (shown above)\n\n[2] (source: ")
    assert len(contexts) == 2
    assert AgentPipeline._observe(contexts, []) == ([], "No passages found.")


def test_agent_answers_from_what_it_found_after_max_iterations() -> None:
    generator = _ScriptedGenerator(
        ["Action: search[France]", "Action: search[Paris]", "Action: search[more]", "Paris [1]"]
    )
    pipeline = _build(AgentPipeline, generator, max_iterations=2)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris [1]"
    assert len(response.metadata["steps"]) == 2
    assert "Searched: France\nSearched: Paris" in generator.calls[-1][1]


def test_agent_without_an_action_falls_back_to_plain_retrieval() -> None:
    generator = _ScriptedGenerator(["I think it's Paris.", "Paris [1]"])
    pipeline = _build(AgentPipeline, generator)

    response = pipeline.run("What is the capital of France?")

    assert response.answer == "Paris [1]"
    assert response.contexts[0].chunk.source == "france.txt"


def test_agent_parse_action() -> None:
    parse = AgentPipeline._parse_action
    assert parse("Thought: x\nAction: search[a b]\nmore") == ("search", "a b")
    assert parse("Action: finish[Paris [1]]") == ("finish", "Paris [1]")
    assert parse("action: Finish[multi\nline]") == ("finish", "multi\nline")
    assert parse("Action: lookup[x]") is None
    assert parse("no action") is None
