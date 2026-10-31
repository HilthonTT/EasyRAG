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

# Self-Ask (Press et al., 2023). The few-shot examples live in the system prompt so the user
# prompt is only the scratchpad: a generator that echoes its prompt cannot trip the parsers.
SELF_ASK_SYSTEM_PROMPT = """Answer multi-hop questions by breaking them into simpler follow-up
questions. Write exactly one next step, then stop:
- "Follow up: <question>" if you need another fact, or
- "So the final answer is: <answer>" once you can answer.
Never write the "Intermediate answer:" lines yourself; they are filled in from a search engine.

Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No"""

SELF_ASK_SCRATCHPAD = "Question: {question}\nAre follow up questions needed here:"

# IRCoT (Trivedi et al., 2023): one reasoning sentence per step, each used as the next query.
IRCOT_SYSTEM_PROMPT = """You answer questions by reasoning step by step over the numbered
context passages. Write only the next single sentence of reasoning, building on the thoughts
so far. When you can answer, write "So the answer is: <answer>" instead."""

IRCOT_PROMPT = """Context:
{context}

Question: {question}
Thoughts so far:
{thoughts}
Next thought:"""

# Final answer for the multi-step pipelines when the model never produced one itself.
REASONING_ANSWER_PROMPT = """Context:
{context}

Question: {question}
Reasoning so far:
{reasoning}
Answer:"""


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
