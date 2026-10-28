from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.evaluation import EvalExample, evaluate
from easyrag.pipelines import HyDEPipeline


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
