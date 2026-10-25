"""Evaluate a pipeline on a JSONL dataset.

Each line: {"question": "...", "answers": ["..."], "context": "..."}

Usage: uv run python scripts/evaluate.py path/to/dataset.jsonl [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
from statistics import mean

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.evaluation import exact_match, token_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    config = Config.load(args.config)
    pipeline = NaiveRAGPipeline.from_config(config)
    pipeline.index([Document(content=row["context"]) for row in rows if row.get("context")])

    em, f1 = [], []
    for row in rows:
        answer = pipeline.run(row["question"]).answer
        em.append(exact_match(answer, row["answers"]))
        f1.append(token_f1(answer, row["answers"]))

    print(f"Samples: {len(rows)}  EM: {mean(em):.3f}  F1: {mean(f1):.3f}")


if __name__ == "__main__":
    main()
