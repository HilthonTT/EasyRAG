import pytest

from easyrag.schema import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(content="Paris is the capital of France.", metadata={"source": "france.txt"}),
        Document(content="Berlin is the capital of Germany.", metadata={"source": "germany.txt"}),
        Document(content="Python is a programming language.", metadata={"source": "python.txt"}),
    ]
