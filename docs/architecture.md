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

| Package        | Base class        | Built-in implementations |
| -------------- | ----------------- | ------------------------ |
| `loaders`      | `BaseLoader`      | `TextLoader`, `PDFLoader`, `HTMLLoader`, `DirectoryLoader` |
| `splitters`    | `BaseSplitter`    | `RecursiveCharacterSplitter`, `CharacterSplitter` |
| `embeddings`   | `BaseEmbeddings`  | `HashingEmbeddings`, `SentenceTransformerEmbeddings`, `OpenAIEmbeddings`, `FakeEmbeddings` (tests) |
| `vectorstores` | `BaseVectorStore` | `InMemoryVectorStore` (saved as NumPy `.npz`) |
| `retrievers`   | `BaseRetriever`   | `VectorRetriever`, `BM25Retriever`, `HybridRetriever` (reciprocal rank fusion) |
| `rerankers`    | `BaseReranker`    | `CrossEncoderReranker` |
| `generators`   | `BaseGenerator`   | `EchoGenerator`, `AnthropicGenerator`, `OpenAIGenerator`, `OllamaGenerator` |
| `pipelines`    | `BasePipeline`    | `NaiveRAGPipeline`, `HyDEPipeline`, `MultiQueryPipeline`, `IterativeRAGPipeline`, `SelfAskPipeline`, `IRCOTPipeline` |
| `evaluation`   | —                 | `evaluate`; metrics `exact_match`, `contains_answer`, `token_f1`, `hit_rate`, `recall_at_k`, `reciprocal_rank` |

`factory.py` builds every component from the config (`configs/default.yaml`), and
`build_pipeline` picks the pipeline class by `pipeline.type`. If a reranker is configured,
`BasePipeline.retrieve` fetches `reranker.fetch_k` candidates and reranks them down to `top_k`.
`BasePipeline.save` / `load` write the chunks and vectors to `pipeline.index_dir`, and re-embed
when the saved embedding model differs from the configured one.

## Pipelines

| `pipeline.type` | Class                  | How it combines retrieval and generation |
| --------------- | ---------------------- | ---------------------------------------- |
| `naive`         | `NaiveRAGPipeline`     | Retrieve once, then generate (streams) |
| `hyde`          | `HyDEPipeline`         | Search with an LLM-written hypothetical answer |
| `multi_query`   | `MultiQueryPipeline`   | Search with the question and `num_queries` LLM rewrites, fused by rank |
| `iterative`     | `IterativeRAGPipeline` | Re-retrieve with the question plus the previous answer, `max_iterations` rounds |
| `self_ask`      | `SelfAskPipeline`      | The model asks follow-up questions, each answered from retrieval |
| `ircot`         | `IRCOTPipeline`        | One reasoning sentence per step, each used as the next search query |

## Adding a component

1. Create a module in the matching subpackage, e.g. `src/easyrag/embeddings/openai.py`.
2. Subclass the base class and implement its abstract methods.
3. Import optional third-party packages *inside* the class, so the core install stays light,
   and add the dependency to an extra in `pyproject.toml`.
4. Export it from the subpackage `__init__.py` and add tests under `tests/unit/`.
