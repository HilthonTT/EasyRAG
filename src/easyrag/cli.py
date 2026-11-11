"""Command-line entry point: `easyrag index <docs>`, then `easyrag ask "question"`."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from easyrag.config import Config
from easyrag.evaluation.citations import invalid_citations
from easyrag.factory import build_judge, build_pipeline
from easyrag.loaders.directory import DirectoryLoader
from easyrag.pipelines.base import BasePipeline
from easyrag.schema import Document, RetrievedChunk


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
    index.add_argument(
        "--update",
        action="store_true",
        help="Update the saved index instead of replacing it: re-index changed files, add new "
        "ones and drop files deleted from DOCS",
    )

    remove = sub.add_parser("remove", help="Remove source files from the saved index")
    remove.add_argument("sources", nargs="+", help="Source file paths, as they were indexed")

    ask = sub.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    _add_query_options(ask)

    chat = sub.add_parser("chat", help="Ask questions interactively")
    _add_query_options(chat)

    ev = sub.add_parser("eval", help="Evaluate on a JSONL set of questions")
    ev.add_argument("dataset", help='JSONL: {"question", "answers": [...], "sources": [...]}')
    ev.add_argument("--docs", help="Index these documents instead of loading the saved index")
    ev.add_argument("--retrieval-only", action="store_true", help="Skip generation (no LLM)")
    ev.add_argument(
        "--judge", action="store_true", help="Also grade correctness and faithfulness with an LLM"
    )
    ev.add_argument("--output", help="Write per-question results to this JSON file")

    args = parser.parse_args(argv)
    if getattr(args, "judge", False) and args.retrieval_only:
        parser.error("--judge needs generated answers; drop --retrieval-only")
    try:
        config = Config.load(args.config)
        pipeline = build_pipeline(config)
        judge = build_judge(config) if getattr(args, "judge", False) else None
    except (ValueError, ImportError, OSError) as e:  # bad config value, missing extra, no file
        parser.error(str(e))
    index_dir = Path(args.index_dir or config.pipeline.index_dir)

    if args.command == "index":
        if args.update:
            _update(pipeline, args.docs, args.glob, index_dir)
        else:
            n_docs, n_chunks = _index(pipeline, args.docs, args.glob)
            print(f"Indexed {n_docs} documents into {n_chunks} chunks -> {index_dir}")
        pipeline.save(index_dir)
        return
    if args.command == "remove":
        _prepare(pipeline, None, index_dir)
        removed = pipeline.remove(args.sources)
        if not removed:
            raise SystemExit(f"No indexed chunks come from: {', '.join(args.sources)}")
        pipeline.save(index_dir)
        print(f"Removed {removed} chunks; {len(pipeline.chunks)} left -> {index_dir}")
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
            try:
                _answer(pipeline, question, args.show_context)
            except KeyboardInterrupt:  # stop this answer, keep the session
                print("\n[interrupted]")
    elif args.command == "eval":
        from easyrag.evaluation import evaluate, load_examples

        try:
            report = evaluate(
                pipeline, load_examples(args.dataset), generate=not args.retrieval_only, judge=judge
            )
        except ValueError as e:
            parser.error(str(e))
        for name, value in report.summary.items():
            print(f"{name:>16}: {value:.3f}")
        print(f"{'questions':>16}: {len(report.results)}")
        if args.output:
            Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"Per-question results -> {args.output}")


def _add_query_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--docs", help="Index these documents on the fly instead of the saved index")
    p.add_argument("--show-context", action="store_true", help="Print the retrieved passages")


def _load(docs: str, glob: str = "**/*") -> list[Document]:
    try:
        documents = DirectoryLoader(docs, glob=glob).load()
    except (FileNotFoundError, ImportError) as e:  # bad path, or a .pdf without the pdf extra
        raise SystemExit(str(e)) from None
    if not documents:
        raise SystemExit(f"No supported documents found in {docs}")
    return documents


def _index(pipeline: BasePipeline, docs: str, glob: str = "**/*") -> tuple[int, int]:
    documents = _load(docs, glob)
    return len(documents), pipeline.index(documents)


def _update(pipeline: BasePipeline, docs: str, glob: str, index_dir: Path) -> None:
    """Refresh the saved index (if any) with the current contents of `docs`."""
    if (index_dir / "manifest.json").exists():
        pipeline.load(index_dir)
    documents = _load(docs, glob)
    added, removed = pipeline.update(documents)
    removed += pipeline.remove_missing(docs)
    print(
        f"Updated from {len(documents)} documents: {added} chunks added, {removed} removed, "
        f"{len(pipeline.chunks)} total -> {index_dir}"
    )


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
    pieces = []
    for piece in answer:
        print(piece, end="", flush=True)
        pieces.append(piece)
    print()
    bad = invalid_citations("".join(pieces), len(contexts))
    if bad:
        cited = ", ".join(f"[{n}]" for n in bad)
        print(f"\nWarning: the answer cites {cited}, but only {len(contexts)} passages were given.")
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
