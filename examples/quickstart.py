"""Minimal end-to-end example. Run with: uv run python examples/quickstart.py"""

from easyrag import Document, NaiveRAGPipeline

documents = [
    Document(content="EasyRAG is a modular toolkit for Retrieval-Augmented Generation."),
    Document(content="A RAG pipeline retrieves relevant chunks and passes them to an LLM."),
    Document(content="The in-memory vector store uses cosine similarity."),
]

pipeline = NaiveRAGPipeline.from_config()
pipeline.index(documents)

response = pipeline.run("What does a RAG pipeline do?")
for result in response.contexts:
    print(f"{result.score:.3f}  {result.chunk.content}")

print("\n--- Prompt sent to the generator ---\n")
print(response.answer)
