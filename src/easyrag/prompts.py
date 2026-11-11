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

# Multi-query (LangChain MultiQueryRetriever; RAG-Fusion). The instructions live in the system
# prompt so the user prompt is only the question: a generator that echoes its prompt then yields
# just the original question back, which is de-duplicated away.
MULTI_QUERY_SYSTEM_PROMPT = """You help a search engine find documents. Given a question, write
{n} different search queries that ask for the same information, using different wording,
synonyms or angles. Write one query per line, with no numbering and no other text."""

# Step-back prompting (Zheng et al., 2023). As with multi-query, the user prompt is only the
# question, so a generator that echoes its prompt adds no second query.
STEP_BACK_SYSTEM_PROMPT = """You help a search engine find background knowledge. Given a
question, write one more generic "step-back" question about the concept, rule or entity behind
it, which is easier to answer and whose answer helps answer the original. Write only the
step-back question.

Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956?
Step-back question: Which positions has Knox Cunningham held in his career?

Question: How many vacation days does a part-time employee working three days a week get?
Step-back question: How is vacation time calculated for part-time employees?"""

# Self-RAG, prompted (after Asai et al., 2024, which trains the model to emit these judgements as
# reflection tokens). Each check puts its instructions in the system prompt; an unparseable reply
# falls back to the plain RAG behaviour.
SELF_RAG_RETRIEVE_SYSTEM_PROMPT = """Decide whether answering the user's message needs facts
from a document collection (policies, products, people, numbers, events), or can be answered
without any (greetings, arithmetic, general reasoning, rewording). Reply with only
"Retrieve: yes" or "Retrieve: no"."""

SELF_RAG_RELEVANCE_SYSTEM_PROMPT = """You are given a question and numbered passages. List the
passages that contain information useful for answering the question. Reply with one line only:
"Relevant: 1, 3" (the numbers), or "Relevant: none"."""

SELF_RAG_RELEVANCE_PROMPT = """Passages:
{context}

Question: {question}"""

SELF_RAG_SUPPORT_SYSTEM_PROMPT = """You check an answer against the numbered passages it was
written from. Reply with one line only: "Supported: yes" if every factual claim in the answer is
backed by the passages, or "Supported: no" if any claim is missing from or contradicts them."""

SELF_RAG_SUPPORT_PROMPT = """Passages:
{context}

Answer: {answer}"""

QUERY_REWRITE_SYSTEM_PROMPT = """The search for the user's question found nothing useful.
Rewrite it as one better search query: name the specific things to look up and use the words a
reference document would use. Write only the query."""

DIRECT_ANSWER_SYSTEM_PROMPT = """Answer the user's message directly and concisely."""

# Agent-style RAG: a ReAct loop (Yao et al., 2023) with a search tool. The instructions live in
# the system prompt so the user prompt is only the trajectory.
AGENT_SYSTEM_PROMPT = """You answer questions by searching a document collection. Work in steps.
In each step write one "Thought:" line, then exactly one action line, then stop:
- "Action: search[<query>]" to search; the passages found are shown to you as an Observation,
  numbered so you can cite them.
- "Action: finish[<answer>]" once you can answer. Cite the passages you used with their numbers
  in square brackets, e.g. [1] or [2][3]. If the passages do not contain the answer, say so.
Search for one fact at a time and rephrase the query if a search finds nothing useful. Never
write an "Observation:" yourself; it comes from the search engine."""

AGENT_SCRATCHPAD = "Question: {question}"

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
