# Roadmap

Steps for turning EasyRAG from a scaffold into a working RAG system, in order.
Measure after each change: an improvement only counts if the evaluation numbers go up.

## 0. Baseline

- [x] Project scaffold (components, config, CLI, tests, CI)
- [ ] Commit the scaffold

## 1. Real models

The built-in `FakeEmbeddings` and `EchoGenerator` are placeholders. Replace them first.

### Embeddings

- [ ] Install the extra: `uv sync --extra huggingface`
- [ ] Implement `SentenceTransformerEmbeddings` in `src/easyrag/embeddings/sentence_transformer.py`
  - Normalize vectors (`normalize_embeddings=True`) so cosine scores are consistent
  - Return plain lists (`.tolist()`), not numpy arrays
  - Support a query prefix if the model needs one (check the model card: BGE, E5, ...)
- [ ] Export it from `src/easyrag/embeddings/__init__.py`
- [ ] Check it finds results by meaning, where `FakeEmbeddings` only matches exact words
- [ ] Integration test in `tests/integration/test_sentence_transformer.py`

References:
- LangChain `HuggingFaceEmbeddings`: [`libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py)
- FlashRAG `STEncoder`: [`flashrag/retriever/encoder.py`](https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/retriever/encoder.py)
- [sbert.net quickstart](https://sbert.net/docs/quickstart.html) and [pretrained models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)

### Generator

- [ ] Add an LLM generator in `src/easyrag/generators/` (OpenAI, Anthropic or Ollama),
      implementing `generate(prompt) -> str`
- [ ] Wire `embedding.provider` and `generator.provider` in `NaiveRAGPipeline.from_config`

## 2. Evaluation set

- [ ] Write 20–50 questions about your own documents, with the correct answer and source file,
      as JSONL in `data/`
- [ ] Run `uv run python scripts/evaluate.py data/<file>.jsonl` and record the baseline
- [ ] Measure retrieval and generation separately:
  - Retrieval: did the right chunk come back? (`hit_rate`)
  - Generation: given the right chunk, was the answer correct? (EM / F1, later an LLM judge)

Most RAG failures are retrieval failures.

## 3. Better retrieval (one change at a time)

- [ ] **Chunking:** split on paragraphs or sentences instead of fixed character counts; try sizes
- [ ] **Hybrid search:** add a BM25 retriever (`rank-bm25`) and merge with vector results
- [ ] **Reranking:** retrieve top 20, rescore with a cross-encoder (e.g. `bge-reranker`), keep top 4
      (new `rerankers/` package)
- [ ] **Query rewriting:** rephrase the question or search with a hypothetical answer (HyDE)

## 4. Practical features

- [ ] **Persistence:** FAISS, Chroma or Qdrant vector store, so documents aren't re-embedded every run
- [ ] **More loaders:** PDF (`--extra pdf`), HTML
- [ ] **Citations:** prompt the model to cite the `[1]`, `[2]` chunk numbers

## 5. Advanced pipelines

- [ ] Multi-step / iterative retrieval
- [ ] Self-RAG (the model decides when to retrieve)
- [ ] Agent-style RAG

Each is a new class in `src/easyrag/pipelines/`. FlashRAG's
[`pipeline/`](https://github.com/RUC-NLPIR/FlashRAG/tree/main/flashrag/pipeline) folder is a good reference.
