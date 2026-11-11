# Roadmap

Steps for turning EasyRAG from a scaffold into a working RAG system, in order.
Measure after each change: an improvement only counts if the evaluation numbers go up.

## 0. Baseline

- [x] Project scaffold (components, config, CLI, tests, CI)
- [x] Commit the scaffold

## 1. Real models

### Embeddings

- [x] `SentenceTransformerEmbeddings` in `src/easyrag/embeddings/sentence_transformer.py`
      (`uv sync --extra huggingface`): normalized vectors, plain lists, BGE/E5 query prefixes
- [x] `OpenAIEmbeddings`; `HashingEmbeddings` as the no-download default
- [x] Integration test in `tests/integration/test_sentence_transformer.py`

References:
- LangChain `HuggingFaceEmbeddings`: [`libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py`](https://github.com/langchain-ai/langchain/blob/master/libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py)
- FlashRAG `STEncoder`: [`flashrag/retriever/encoder.py`](https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/retriever/encoder.py)
- [sbert.net quickstart](https://sbert.net/docs/quickstart.html) and [pretrained models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)

### Generator

- [x] Anthropic, OpenAI (and OpenAI-compatible) and Ollama generators, with streaming
- [x] `embedding.provider` and `generator.provider` wired through `factory.py`

## 2. Evaluation set

- [x] Example set: 56 questions over a fictional 20-document corpus in `examples/eval/`
- [x] Retrieval baseline recorded in [`examples/eval/README.md`](../examples/eval/README.md)
- [ ] Answer baseline with a real generator (`contains_answer`, `f1`, `--judge`)
- [ ] Write 20–50 questions about your own documents, with the correct answer and source file,
      as JSONL in `data/` (`scripts/make_eval_set.py` drafts them)
- [x] Retrieval and generation are measured separately:
  - Retrieval: did the right chunk come back? (`hit_rate`, `mrr`; `--retrieval-only` skips the LLM)
  - Generation: was the answer correct? (`exact_match`, `contains_answer`, `f1`)
- [x] LLM judge for answer correctness and faithfulness to the retrieved passages

Most RAG failures are retrieval failures.

## 3. Better retrieval (one change at a time)

- [x] **Chunking:** `RecursiveCharacterSplitter` splits on paragraphs, lines, then sentences
- [x] **Hybrid search:** `BM25Retriever` merged with vector results by reciprocal rank fusion
- [x] **Reranking:** retrieve `fetch_k`, rescore with `CrossEncoderReranker`, keep `top_k`
- [x] **Query rewriting:** HyDE (`pipeline.type: hyde`)
- [x] **Multi-query:** `pipeline.type: multi_query` searches with LLM rewrites, fused by RRF
- [ ] Step-back questions

## 4. Practical features

- [x] **Persistence:** `easyrag index` saves chunks and vectors; `ask` / `chat` load them
- [x] **Vector database:** `FaissVectorStore` (`vectorstore.provider: faiss`, `--extra faiss`),
      exact `flat` or approximate `hnsw` index for indexes beyond ~100k chunks
- [x] **Incremental indexing:** `easyrag index --update` re-indexes changed files, adds new ones and
      drops deleted ones; `easyrag remove` drops files (`BasePipeline.update` / `remove`)
- [x] **More loaders:** PDF (`--extra pdf`), HTML
- [x] **Citations:** the model cites `[1]`, `[2]` passages and the CLI lists the sources
- [ ] Check that cited numbers point at passages that support the claim

## 5. Advanced pipelines

- [x] Multi-step / iterative retrieval (`iterative`, ITER-RETGEN)
- [x] Question decomposition (`self_ask`) and interleaved chain-of-thought retrieval (`ircot`)
- [ ] Self-RAG (the model decides when to retrieve)
- [ ] Agent-style RAG

Each is a new class in `src/easyrag/pipelines/`, registered in `PIPELINES`. FlashRAG's
[`pipeline/`](https://github.com/RUC-NLPIR/FlashRAG/tree/main/flashrag/pipeline) folder is a good reference.
