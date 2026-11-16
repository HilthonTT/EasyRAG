# Example evaluation set

A small benchmark for comparing EasyRAG settings: 20 documents about **Tessaly Marine
Instruments**, a made-up company (handbook policies, products, engineering practices), and 56
questions in [`questions.jsonl`](questions.jsonl).

The company is fictional, so no model knows the answers from its training data: a correct
answer has to come from retrieval. The documents deliberately repeat similar facts (several
"six months", "14", "one hour" and "30 minutes" rules in different files), so a retriever that
only matches numbers or surface words picks the wrong file. Three questions need two documents
(for example, "In what year did the office that handles returns from the Americas open?").

## Running it

```bash
# Retrieval only: no LLM needed, takes seconds
uv run easyrag eval examples/eval/questions.jsonl --docs examples/eval/docs --retrieval-only

# Full answers (set generator.provider first), optionally graded by an LLM judge
uv run easyrag eval examples/eval/questions.jsonl --docs examples/eval/docs --judge --output results.json
```

Change one setting at a time with environment variables, e.g. `EASYRAG_RETRIEVER_METHOD=bm25` or
`EASYRAG_RETRIEVER_TOP_K=1`. Answers are short strings and `contains_answer` checks whether any
of them appears in the generated answer, so each question lists several phrasings ("10 days",
"ten days").

With generation, `eval` also prints how many LLM calls and tokens the answers (and the judge)
used, and `--output` stores them per question under `usage` and `judge_usage`. Record them next
to answer scores: multi-step pipelines can cost several times more than `naive`.

`tests/unit/test_eval_set.py` checks that every source names exactly one document and that an
accepted answer appears in it. Run it after editing the set.

## Baseline

Retrieval only, default config (hashing embeddings, chunk size 500, no reranker), 2026-11-10:

| Retriever | hit@1 | hit@2 | hit@4 | MRR@4 |
| --------- | ----- | ----- | ----- | ----- |
| `bm25`    | 0.911 | 0.964 | 0.982 | 0.943 |
| `vector`  | 0.786 | 0.875 | 0.964 | 0.856 |
| `hybrid`  | 0.929 | 0.982 | 1.000 | 0.952 |

With 20 documents hit@4 is close to the ceiling, so compare retrievers on **hit@1**
(`EASYRAG_RETRIEVER_TOP_K=1`) and MRR. The `vector` row uses the offline hashing embeddings,
which only match exact words.

### Real embeddings and reranking

Same set, 2026-11-15. `st` = `embedding.provider: sentence_transformer` (default model
`BAAI/bge-small-en-v1.5`); `rerank` = `reranker.provider: cross_encoder` (default model
`BAAI/bge-reranker-base`, `fetch_k: 20`). Needs `uv sync --extra huggingface`.

| Embeddings | Retriever | Reranker | hit@1 | hit@2 | hit@4 | MRR@4 |
| ---------- | --------- | -------- | ----- | ----- | ----- | ----- |
| hash       | `hybrid`  | –        | 0.929 | 0.982 | 1.000 | 0.952 |
| st         | `vector`  | –        | 0.893 | 0.964 | 0.964 | 0.929 |
| st         | `hybrid`  | –        | 0.893 | 0.964 | 1.000 | 0.949 |
| hash       | `hybrid`  | rerank   | **0.982** | **1.000** | 1.000 | **0.991** |
| st         | `hybrid`  | rerank   | **0.982** | 0.982 | 1.000 | 0.988 |

What this shows:

- **The reranker is the biggest win.** It lifts hit@1 from 0.929 to 0.982 with either embedding.
  Only one question still misses at top 1 ("Where should a customer in Brazil send a faulty
  instrument?"). It needs the warranty document but gets a product page that mentions
  instruments.
- **Real embeddings help `vector` a lot** (hit@1 0.786 → 0.893) **but not `hybrid`**
  (0.929 → 0.893). On this corpus the wording of the questions overlaps strongly with the
  documents, so BM25 already finds most answers. The dense model adds confusions between
  similar handbook pages (expenses vs travel, leave vs learning, IT equipment vs remote work).
  Real embeddings should matter more on corpora where users phrase things differently from
  the documents.
- Recommended retrieval settings for this set: `hybrid` + `cross_encoder` reranker. The
  embedding model makes little difference once you rerank.

Answer quality (`contains_answer`, `f1`, judge scores) has no baseline yet. It needs a real
generator, so record one the first time you run with `generator.provider` set.
