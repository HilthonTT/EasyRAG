from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.evaluation import EvalExample, LLMJudge, evaluate
from easyrag.evaluation.evaluator import source_matches
from easyrag.generators.base import BaseGenerator
from easyrag.pipelines import HyDEPipeline


class StubGenerator(BaseGenerator):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.reply


def _pipeline(cls: type[NaiveRAGPipeline], docs: list[Document]) -> NaiveRAGPipeline:
    pipeline = cls.from_config(Config.from_dict({"retriever": {"top_k": 1}}))
    pipeline.index(docs)
    return pipeline


def test_summary_averages_each_metric_over_examples_that_have_it(
    sample_documents: list[Document],
) -> None:
    pipeline = _pipeline(NaiveRAGPipeline, sample_documents)
    examples = [
        EvalExample("What is the capital of France?"),  # no sources or answers: no scores
        EvalExample("What is the capital of France?", sources=["france.txt"]),
        EvalExample("What is the capital of Germany?", sources=["python.txt"]),
    ]
    summary = evaluate(pipeline, examples, generate=False).summary
    assert summary["hit_rate"] == 0.5


def test_retrieval_only_eval_uses_pipeline_search_query(sample_documents: list[Document]) -> None:
    class RewritingPipeline(HyDEPipeline):
        def search_query(self, question: str) -> str:
            return "Python programming language"

    pipeline = _pipeline(RewritingPipeline, sample_documents)
    report = evaluate(pipeline, [EvalExample("capital of France?", sources=["python.txt"])], False)
    assert report.summary["hit_rate"] == 1.0


def test_judge_parses_score_and_reason() -> None:
    judge = LLMJudge(StubGenerator('Sure: {"score": 1, "reason": "matches"}'))
    assert judge.correctness("q", "a", ["a"]) == (1.0, "matches")


def test_judge_garbage_reply_scores_zero() -> None:
    score, reason = LLMJudge(StubGenerator("no json here")).faithfulness("a", "ctx")
    assert score == 0.0
    assert "unparseable" in reason


def test_evaluate_with_judge(sample_documents: list[Document]) -> None:
    pipeline = _pipeline(NaiveRAGPipeline, sample_documents)
    judge = LLMJudge(StubGenerator('{"score": 1, "reason": "ok"}'))
    examples = [EvalExample("capital of France?", answers=["Paris"]), EvalExample("unanswerable?")]
    report = evaluate(pipeline, examples, judge=judge)
    assert report.summary["judge_correct"] == 1.0
    assert all("faithfulness" in r.scores for r in report.results)
    assert report.results[0].judge_reasons["faithfulness"] == "ok"


def test_source_matches_trailing_path_parts() -> None:
    assert source_matches("data/docs/a/intro.md", "intro.md")
    assert source_matches(r"data\docs\a\intro.md", "a/intro.md")
    assert not source_matches("data/docs/a/intro.md", "b/intro.md")
    assert not source_matches("data/docs/myintro.md", "intro.md")
    assert not source_matches("intro.md", "")


def test_same_file_name_in_different_folders_is_not_a_hit() -> None:
    docs = [
        Document(content="Paris is the capital of France.", metadata={"source": "eu/capitals.txt"}),
        Document(
            content="Tokyo is the capital of Japan.", metadata={"source": "asia/capitals.txt"}
        ),
    ]
    pipeline = _pipeline(NaiveRAGPipeline, docs)
    examples = [
        EvalExample("What is the capital of France?", sources=["asia/capitals.txt"]),
        EvalExample("What is the capital of France?", sources=["eu/capitals.txt"]),
    ]
    report = evaluate(pipeline, examples, generate=False)
    assert [r.scores["hit_rate"] for r in report.results] == [0.0, 1.0]
    assert report.results[1].retrieved_sources == ["eu/capitals.txt"]
