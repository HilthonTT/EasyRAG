"""Evaluate a pipeline on a JSONL dataset.

    uv run python scripts/evaluate.py examples/data/eval.jsonl --docs examples/data
    uv run python scripts/evaluate.py data/my_eval.jsonl --retrieval-only

Equivalent to `rag eval ...`; kept as a script for experiments you want to customise.
"""

from __future__ import annotations

import sys

from easyrag.cli import main

if __name__ == "__main__":
    main(["eval", *sys.argv[1:]])
