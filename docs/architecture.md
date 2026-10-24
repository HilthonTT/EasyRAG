# Architecture

EasyRAG is split into small components, each defined by an abstract base class in its
own subpackage. A pipeline wires them together.

```
Documents ─► Loader ─► Splitter ─► Embeddings ─► VectorStore
                                                     │
Question ─────────────► Retriever ◄──────────────────┘
                            │
                            ▼
                   Prompt ─► Generator ─► Answer
```

| Package        | Base class        | Built-in implementations         |
| -------------- | ----------------- | -------------------------------- |
| `loaders`      | `BaseLoader`      | `TextLoader`, `DirectoryLoader`  |
| `splitters`    | `BaseSplitter`    | `CharacterSplitter`              |
| `embeddings`   | `BaseEmbeddings`  | `FakeEmbeddings`                 |
| `vectorstores` | `BaseVectorStore` | `InMemoryVectorStore`            |
| `retrievers`   | `BaseRetriever`   | `VectorRetriever`                |
| `generators`   | `BaseGenerator`   | `EchoGenerator`                  |
| `pipelines`    | `BasePipeline`    | `NaiveRAGPipeline`               |
| `evaluation`   | —                 | `exact_match`, `token_f1`, `hit_rate` |

## Adding a component

1. Create a module in the matching subpackage, e.g. `src/easyrag/embeddings/openai.py`.
2. Subclass the base class and implement its abstract methods.
3. Import optional third-party packages *inside* the class, so the core install stays light,
   and add the dependency to an extra in `pyproject.toml`.
4. Export it from the subpackage `__init__.py` and add tests under `tests/unit/`.
