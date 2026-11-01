"""Draft an evaluation set from your own documents with an LLM.

The drafts are a starting point: review every line, fix or delete weak questions, and add
paraphrased, multi-hop and unanswerable questions by hand before you trust the numbers.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from easyrag.evaluation.evaluator import EvalExample
from easyrag.evaluation.metrics import normalize_answer
from easyrag.generators.base import BaseGenerator
from easyrag.schema import Chunk

QUESTION_PROMPT = """Passage from the file "{source}":
\"\"\"
{passage}
\"\"\"

Write one question that a user of these documents might realistically ask and that this
passage answers. Rules:
- The question must make sense on its own, without seeing the passage (name the subject;
  never write "according to the passage" or "in this text").
- Use your own words, not the passage's exact phrasing.
- The answer must be short (a name, number, date or short phrase) and copied word for word
  from the passage.
- If the passage has no clear fact worth asking about, reply with {{"skip": true}}.

Reply with JSON only: {{"question": "...", "answer": "..."}}"""


def _parse(raw: str) -> tuple[str, str] | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("skip"):
        return None
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    return (question, answer) if question and answer else None


def draft_examples(
    chunks: list[Chunk],
    generator: BaseGenerator,
    per_source: int = 3,
    min_chars: int = 200,
    seed: int = 0,
) -> list[EvalExample]:
    """Ask the generator for one question per sampled chunk, up to `per_source` per file.

    Drafts whose answer does not appear in the passage are dropped: they are usually
    hallucinated or paraphrased, and string metrics could never match them."""
    rng = random.Random(seed)
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if len(chunk.content.strip()) >= min_chars:
            by_source[chunk.source].append(chunk)

    examples: list[EvalExample] = []
    seen: set[str] = set()
    for source in sorted(by_source):
        candidates = by_source[source][:]
        rng.shuffle(candidates)
        kept = 0
        for chunk in candidates:
            if kept >= per_source:
                break
            prompt = QUESTION_PROMPT.format(source=Path(source).name, passage=chunk.content)
            parsed = _parse(generator.generate(prompt, system="You write evaluation questions."))
            if parsed is None:
                continue
            question, answer = parsed
            if normalize_answer(answer) not in normalize_answer(chunk.content):
                continue
            if normalize_answer(question) in seen:
                continue
            seen.add(normalize_answer(question))
            examples.append(EvalExample(question, answers=[answer], sources=[Path(source).name]))
            kept += 1
    return examples


def write_examples(examples: list[EvalExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            row = {"question": ex.question, "answers": ex.answers, "sources": ex.sources}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
