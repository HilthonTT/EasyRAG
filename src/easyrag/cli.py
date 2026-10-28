"""Command-line entry point: `easyrag ask <docs_dir> "question"`."""

from __future__ import annotations

import argparse

from easyrag.config import Config
from easyrag.loaders.directory import DirectoryLoader
from easyrag.pipelines import PIPELINES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="easyrag", description="EasyRAG command line")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Index a folder of text files and ask a question")
    ask.add_argument("docs_dir", help="Directory containing documents")
    ask.add_argument("question", help="Question to ask")
    ask.add_argument("--config", help="Path to a YAML config file")
    ask.add_argument("--glob", default="**/*.txt", help="File pattern to load")

    args = parser.parse_args(argv)

    if args.command == "ask":
        config = Config.load(args.config)
        if config.pipeline.type not in PIPELINES:
            parser.error(f"Unknown pipeline.type: {config.pipeline.type!r}")
        pipeline = PIPELINES[config.pipeline.type].from_config(config)
        n_chunks = pipeline.index(DirectoryLoader(args.docs_dir, glob=args.glob).load())
        print(f"Indexed {n_chunks} chunks.\n")
        print(pipeline.run(args.question).answer)


if __name__ == "__main__":
    main()
