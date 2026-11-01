"""Draft an evaluation set from your documents with the configured LLM.

    uv run python scripts/make_eval_set.py data/docs --output data/eval_draft.jsonl
    uv run python scripts/make_eval_set.py data/docs --per-source 5 --config configs/my.yaml

Uses `judge.provider` if set, otherwise `generator.provider` (it must be a real LLM, not echo).
Review the draft by hand, then save it as data/eval.jsonl and run `easyrag eval data/eval.jsonl`.
"""

from __future__ import annotations

import argparse

from easyrag.config import Config
from easyrag.evaluation.synthetic import draft_examples, write_examples
from easyrag.factory import build_generator, build_splitter
from easyrag.loaders.directory import DirectoryLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("docs", help="Directory or file: .txt .md .rst .pdf .html")
    parser.add_argument("--output", default="data/eval_draft.jsonl")
    parser.add_argument("--glob", default="**/*", help="File pattern inside the directory")
    parser.add_argument("--per-source", type=int, default=3, help="Questions per file")
    parser.add_argument("--min-chars", type=int, default=200, help="Skip shorter chunks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", help="YAML config file (default: configs/default.yaml)")
    args = parser.parse_args()

    config = Config.load(args.config)
    gen_cfg = config.judge if config.judge.provider else config.generator
    if gen_cfg.provider == "echo":
        parser.error("set generator.provider (or judge.provider) to a real LLM, not echo")
    generator = build_generator(gen_cfg)

    documents = DirectoryLoader(args.docs, glob=args.glob).load()
    if not documents:
        parser.error(f"No supported documents found in {args.docs}")
    chunks = build_splitter(config.splitter).split_documents(documents)

    examples = draft_examples(chunks, generator, args.per_source, args.min_chars, args.seed)
    write_examples(examples, args.output)
    print(f"Drafted {len(examples)} questions from {len(documents)} documents -> {args.output}")
    print("Review every line before using it as data/eval.jsonl.")


if __name__ == "__main__":
    main()
