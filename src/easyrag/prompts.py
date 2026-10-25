"""Prompt templates."""

from __future__ import annotations
from pathlib import Path

from easyrag.schema import RetrievedChunk

SYSTEM_PROMPT = """You answer questions using only the numbered context passages you are given.
Cite the passages you rely on with their numbers in square brackets, e.g. [1] or [2][3].
If the context does not contain the answer, say that you don't know rather than guessing."""

DEFAULT_RAG_PROMPT = """Context:
{context}

Question: {question}
Answer:"""

HYDE_PROMPT = """Write a short passage that answers the question below, as it might appear in a
reference document. It is used only for search, so be specific and factual in style.

Question: {question}
Passage:"""

ITERATIVE_QUERY_TEMPLATE = "{question}\n{previous_answer}"


def format_context(results: list[RetrievedChunk]) -> str:
    parts = []

    for i, r in enumerate(results, start=1):
        source = r.chunk.source
        header = f"[{i}] (source: {Path(source).name})" if source else f"[{i}]"
        parts.append(f"{header}\n{r.chunk.content}")

    return "\n\n".join(parts)


def build_rag_prompt(
    question: str, results: list[RetrievedChunk], template: str = DEFAULT_RAG_PROMPT
) -> str:
    return template.format(context=format_context(results), question=question)
