from __future__ import annotations

import re
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    SELF_RAG_RELEVANCE_PROMPT,
    SELF_RAG_RELEVANCE_SYSTEM_PROMPT,
    SELF_RAG_RETRIEVE_SYSTEM_PROMPT,
    SELF_RAG_SUPPORT_PROMPT,
    SELF_RAG_SUPPORT_SYSTEM_PROMPT,
    build_rag_prompt,
    format_context,
)
from easyrag.schema import RAGResponse, RetrievedChunk

_RETRIEVE = re.compile(r"Retrieve:\s*(yes|no)\b", re.IGNORECASE)
_RELEVANT = re.compile(r"Relevant:\s*(.*)", re.IGNORECASE)
_SUPPORTED = re.compile(r"Supported:\s*(yes|no|partial)", re.IGNORECASE)


class SelfRAGPipeline(BasePipeline):
    """Self-RAG with prompted reflection (after Asai et al., 2024; LangGraph's self-RAG recipe).

    The original Self-RAG trains a model to emit reflection tokens; here an ordinary LLM is asked
    the same questions:

    1. Retrieve? Messages that need no documents are answered directly.
    2. Which retrieved passages are relevant? Irrelevant ones are dropped before generation.
    3. Is the answer supported by those passages?

    When no passage is relevant or the answer is unsupported, the question is rewritten into a
    new search query and tried again, up to `max_iterations` searches. A reply the checks cannot
    parse counts as "retrieve", "all relevant" and "supported", i.e. plain RAG.
    """

    retrieval_needs_generation = True

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        if not self.needs_retrieval(question):
            direct = self.generator.generate(question, system=DIRECT_ANSWER_SYSTEM_PROMPT)
            return RAGResponse(
                question=question, answer=direct, metadata={"retrieved": False, "attempts": []}
            )

        query = question
        queries: list[str] = []
        attempts: list[dict[str, Any]] = []
        answer: str | None = None
        contexts: list[RetrievedChunk] = []
        candidates: list[RetrievedChunk] = []

        for i in range(self.max_iterations):
            queries.append(query)
            candidates = self.retrieve(query, question)
            relevant = self.filter_relevant(question, candidates)
            attempt: dict[str, Any] = {
                "query": query,
                "retrieved": len(candidates),
                "relevant": len(relevant),
            }
            attempts.append(attempt)
            if relevant:
                contexts = relevant
                answer = self._generate(question, relevant)
                attempt["supported"] = self.is_supported(answer, relevant)
                if attempt["supported"]:
                    break
            if i + 1 < self.max_iterations:
                query = self.rewrite_query(question, queries)

        if answer is None:  # nothing was ever relevant: let the model say it doesn't know
            contexts = candidates
            answer = self._generate(question, contexts)

        return RAGResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            metadata={"retrieved": True, "attempts": attempts},
        )

    def _generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        prompt = build_rag_prompt(question, contexts, self.prompt_template)
        return self.generator.generate(prompt, system=self.system_prompt)

    def needs_retrieval(self, question: str) -> bool:
        raw = self.generator.generate(question, system=SELF_RAG_RETRIEVE_SYSTEM_PROMPT)
        match = _RETRIEVE.search(raw)
        return match is None or match.group(1).lower() == "yes"

    def filter_relevant(
        self, question: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """The candidates the LLM marks as relevant, in their original order."""
        if not candidates:
            return []
        raw = self.generator.generate(
            SELF_RAG_RELEVANCE_PROMPT.format(context=format_context(candidates), question=question),
            system=SELF_RAG_RELEVANCE_SYSTEM_PROMPT,
        )
        keep = self._parse_relevant(raw, len(candidates))
        return candidates if keep is None else [c for i, c in enumerate(candidates, 1) if i in keep]

    @staticmethod
    def _parse_relevant(raw: str, n: int) -> set[int] | None:
        """Passage numbers from "Relevant: 1, 3"; empty for "none"; None if unparseable."""
        match = _RELEVANT.search(raw)
        if match is None:
            return None
        text = match.group(1)
        numbers = {int(x) for x in re.findall(r"\d+", text)}
        if not numbers:
            return set() if "none" in text.lower() else None
        valid = {i for i in numbers if 1 <= i <= n}
        return valid or None

    def is_supported(self, answer: str, contexts: list[RetrievedChunk]) -> bool:
        raw = self.generator.generate(
            SELF_RAG_SUPPORT_PROMPT.format(context=format_context(contexts), answer=answer),
            system=SELF_RAG_SUPPORT_SYSTEM_PROMPT,
        )
        match = _SUPPORTED.search(raw)
        return match is None or match.group(1).lower() == "yes"

    def rewrite_query(self, question: str, tried: list[str]) -> str:
        """A new search query for `question`, different from the ones already `tried`."""
        system = (
            QUERY_REWRITE_SYSTEM_PROMPT
            + "\n\nQueries already tried:\n"
            + "\n".join(f"- {q}" for q in tried)
        )
        rewritten = " ".join(self.generator.generate(question, system=system).split())
        return rewritten or question
