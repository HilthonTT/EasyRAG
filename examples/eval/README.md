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
which only match exact words. Real embeddings (`embedding.provider: sentence_transformer`)
should improve it most on the reworded questions.

Answer quality (`contains_answer`, `f1`, judge scores) has no baseline yet. It needs a real
generator, so record one the first time you run with `generator.provider` set.
