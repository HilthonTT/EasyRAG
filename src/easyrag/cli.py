"""Command-line entry point: `easyrag ask <docs_dir> "question"`."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from easyrag.config import Config
from easyrag.factory import build_pipeline
from easyrag.loaders.directory import DirectoryLoader
from easyrag.pipelines import PIPELINES
from easyrag.pipelines.base import BasePipeline
from easyrag.schema import RetrievedChunk


def main(argv: list[str] | None = None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="easyrag", description="Retrieval-Augmented Generation CLI"
    )
    parser.add_argument("--config", help="YAML config file (default: configs/default.yaml)")
    parser.add_argument(
        "--index-dir", help="Where the index is saved (default: pipeline.index_dir)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Index a folder (or file) of documents and save it")
    index.add_argument("docs", help="Directory or file: .txt .md .rst .pdf .html")
    index.add_argument("--glob", default="**/*", help="File pattern inside the directory")

    ask = sub.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    _add_query_options(ask)

    chat = sub.add_parser("chat", help="Ask questions interactively")
    _add_query_options(chat)

    ev = sub.add_parser("eval", help="Evaluate on a JSONL set of questions")
    ev.add_argument("dataset", help='JSONL: {"question", "answers": [...], "sources": [...]}')
    ev.add_argument("--docs", help="Index these documents instead of loading the saved index")
    ev.add_argument("--retrieval-only", action="store_true", help="Skip generation (no LLM)")
    ev.add_argument("--output", help="Write per-question results to this JSON file")

    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        pipeline = build_pipeline(config)
    except (ValueError, ImportError, OSError) as e:  # bad config value, missing extra, no file
        parser.error(str(e))
    index_dir = Path(args.index_dir or config.pipeline.index_dir)

    if args.command == "index":
        n_docs, n_chunks = _index(pipeline, args.docs, args.glob)
        pipeline.save(index_dir)
        print(f"Indexed {n_docs} documents into {n_chunks} chunks -> {index_dir}")
        return

    _prepare(pipeline, getattr(args, "docs", None), index_dir)

    if args.command == "ask":
        _answer(pipeline, args.question, args.show_context)
    elif args.command == "chat":
        print("Ask a question (empty line or Ctrl+C to quit).")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            _answer(pipeline, question, args.show_context)
    elif args.command == "eval":
        from easyrag.evaluation import evaluate, load_examples

        report = evaluate(pipeline, load_examples(args.dataset), generate=not args.retrieval_only)
        for name, value in report.summary.items():
            print(f"{name:>16}: {value:.3f}")
        print(f"{'questions':>16}: {len(report.results)}")
        if args.output:
            Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"Per-question results -> {args.output}")


def _add_query_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--docs", help="Index these documents on the fly instead of the saved index")
    p.add_argument("--show-context", action="store_true", help="Print the retrieved passages")


def _index(pipeline: BasePipeline, docs: str, glob: str = "**/*") -> tuple[int, int]:
    documents = DirectoryLoader(docs, glob=glob).load()
    if not documents:
        raise SystemExit(f"No supported documents found in {docs}")
    return len(documents), pipeline.index(documents)


def _prepare(pipeline: BasePipeline, docs: str | None, index_dir: Path) -> None:
    if docs:
        n_docs, n_chunks = _index(pipeline, docs)
        print(f"Indexed {n_docs} documents into {n_chunks} chunks.", file=sys.stderr)
    else:
        try:
            pipeline.load(index_dir)
        except FileNotFoundError as e:
            raise SystemExit(f"{e}\nOr pass --docs <folder> to index on the fly.") from None


def _answer(pipeline: BasePipeline, question: str, show_context: bool) -> None:
    contexts, answer = pipeline.stream(question)
    if show_context:
        _print_contexts(contexts)
    for piece in answer:
        print(piece, end="", flush=True)
    print()
    if contexts:  # numbered like the [n] citations in the prompt and answer
        print("\nSources: " + ", ".join(f"[{i}] {_cite(c)}" for i, c in enumerate(contexts, 1)))


def _cite(result: RetrievedChunk) -> str:
    name = Path(result.chunk.source).name or "(unknown)"
    page = result.chunk.metadata.get("page")
    return f"{name} p.{page}" if page else name


def _print_contexts(contexts: list[RetrievedChunk]) -> None:
    for i, r in enumerate(contexts, start=1):
        snippet = " ".join(r.chunk.content.split())[:300]
        print(f"[{i}] {_cite(r)} (score {r.score:.3f})\n    {snippet}\n")


if __name__ == "__main__":
    main()
