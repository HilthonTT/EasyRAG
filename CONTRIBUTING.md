# Contributing

## Setup

```bash
uv sync                     # creates .venv with dev dependencies
uv run pre-commit install   # optional: lint on every commit
```

## Common commands

| Task           | Command                       |
| -------------- | ----------------------------- |
| Run tests      | `uv run pytest`               |
| Lint           | `uv run ruff check .`         |
| Format         | `uv run ruff format .`        |
| Type-check     | `uv run mypy src`             |

## Adding a component

See [docs/architecture.md](docs/architecture.md#adding-a-component).
