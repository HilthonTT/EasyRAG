from easyrag import Config, Document, NaiveRAGPipeline


def test_naive_pipeline_end_to_end(sample_documents: list[Document]) -> None:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"top_k": 2}}))
    assert pipeline.index(sample_documents) == 3

    response = pipeline.run("What is the capital of France?")

    assert len(response.contexts) == 2
    assert "Paris" in response.contexts[0].chunk.content
    # EchoGenerator returns the prompt, so the retrieved context must be in it.
    assert "Paris is the capital of France." in response.answer
