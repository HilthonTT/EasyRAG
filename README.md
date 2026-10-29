# EasyRAG

A simple, modular toolkit for Retrieval-Augmented Generation (RAG) research in Python.

Every stage of the pipeline (loading, splitting, embedding, storing, retrieving and generating)
is a small swappable component behind an abstract base class, so you can try new methods
without rewriting everything else.

## Installation

```bash
git clone https://github.com/HilthonTT/EasyRAG.git
cd EasyRAG
uv sync
```

Optional extras: `uv sync --extra openai`, `--extra huggingface`, `--extra pdf`, or `--extra all`.

## Quickstart

```python
from easyrag import Document, NaiveRAGPipeline

pipeline = NaiveRAGPipeline.from_config()
pipeline.index([Document(content="Paris is the capital of France.")])

response = pipeline.run("What is the capital of France?")
print(response.answer)
```

From the command line:

```bash
uv run easyrag index examples/data      # build and save the index
uv run easyrag ask "What is RAG?"       # or: easyrag ask --docs examples/data "What is RAG?"
uv run easyrag chat                     # interactive
```

See [docs/architecture.md](docs/architecture.md) for how the pieces fit together.

## Configuration

Settings live in [configs/default.yaml](configs/default.yaml). Any value can be overridden with
an environment variable `EASYRAG_<SECTION>_<KEY>`, for example `EASYRAG_RETRIEVER_TOP_K=8`.
API keys go in `.env` (copy `.env.example`).

## Development

```bash
uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy src         # type-check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
