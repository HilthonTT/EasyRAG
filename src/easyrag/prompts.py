"""Prompt templates."""

from __future__ import annotations

from easyrag.schema import RetrievedChunk

DEFAULT_RAG_PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def format_context(results: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {r.chunk.content}" for i, r in enumerate(results, start=1))


def build_rag_prompt(
    question: str, results: list[RetrievedChunk], template: str = DEFAULT_RAG_PROMPT
) -> str:
    return template.format(context=format_context(results), question=question)
