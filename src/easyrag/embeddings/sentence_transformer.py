from easyrag.embeddings.base import BaseEmbeddings


class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str = "",
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run: uv sync --extra hugging face"
            ) from e
        self.model = SentenceTransformer(model)
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def embed_documents(self, texts):
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.embed_documents([self.query_instruction + text])[0]
