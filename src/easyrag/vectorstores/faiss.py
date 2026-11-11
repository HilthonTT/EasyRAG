from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore, normalize_rows, read_vectors

INDEX_TYPES = ("flat", "hnsw")


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "vectorstore.provider faiss needs the faiss extra: uv sync --extra faiss"
        ) from e
    return faiss


class FaissVectorStore(BaseVectorStore):
    """Cosine similarity search with FAISS (`uv sync --extra faiss`).

    `index_type="flat"` is exact like `InMemoryVectorStore`, only faster. `"hnsw"` builds an
    approximate graph index (Malkov & Yashunin, 2018) that stays fast at millions of chunks;
    raise `ef_search` for better recall at some speed cost. Vectors are normalized, so inner
    product equals cosine similarity.
    """

    def __init__(self, index_type: str = "flat", hnsw_m: int = 32, ef_search: int = 64) -> None:
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")
        self._faiss = _import_faiss()
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.chunks: list[Chunk] = []
        self._index: Any = None

    def __len__(self) -> int:
        return len(self.chunks)

    def _new_index(self, dim: int) -> Any:
        faiss = self._faiss
        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = self.ef_search
        return index

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if chunks:
            self._add_matrix(chunks, np.asarray(embeddings, dtype=np.float32))

    def _add_matrix(self, chunks: list[Chunk], matrix: Any) -> None:
        if matrix.ndim != 2 or (self._index is not None and matrix.shape[1] != self._index.d):
            raise ValueError(f"embedding shape {matrix.shape} does not match the stored vectors")
        if self._index is None:
            self._index = self._new_index(matrix.shape[1])
        self._index.add(np.ascontiguousarray(normalize_rows(matrix), dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query = normalize_rows(np.asarray(query_embedding, dtype=np.float32)[None, :])
        k = min(top_k, len(self.chunks))
        if self.index_type == "hnsw":
            self._index.hnsw.efSearch = max(self.ef_search, k)
        scores, positions = self._index.search(np.ascontiguousarray(query), k)
        return [
            RetrievedChunk(self.chunks[int(i)], float(s))
            for s, i in zip(scores[0], positions[0], strict=True)
            if i >= 0  # FAISS pads with -1 when it finds fewer than k
        ]

    def delete(self, chunk_ids: Collection[str]) -> int:
        ids = set(chunk_ids)
        removed = [i for i, c in enumerate(self.chunks) if c.id in ids]
        if not removed:
            return 0
        keep = [i for i, c in enumerate(self.chunks) if c.id not in ids]
        if self.index_type == "flat":  # a flat index compacts in place, keeping the order
            self._index.remove_ids(np.asarray(removed, dtype=np.int64))
        else:  # HNSW graphs cannot drop nodes: rebuild from the kept vectors
            vectors = self._index.reconstruct_n(0, self._index.ntotal)[keep]
            self._index = self._new_index(self._index.d)
            if keep:
                self._index.add(np.ascontiguousarray(vectors))
        self.chunks = [self.chunks[i] for i in keep]
        return len(removed)

    def save(self, path: str | Path) -> None:
        """Write chunk ids and the serialized FAISS index to `path` (an `.npz` file)."""
        ids = np.array([c.id for c in self.chunks], dtype=str)
        blob = (
            self._faiss.serialize_index(self._index)
            if self._index is not None
            else np.zeros(0, dtype=np.uint8)
        )
        with open(path, "wb") as f:
            np.savez(f, ids=ids, faiss_index=blob)

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        """Add the vectors saved at `path`; `chunks` maps saved ids to their chunks.

        A saved index of the same type is used as is (no HNSW rebuild); anything else, including
        an `InMemoryVectorStore` file, is re-added from its vectors.
        """
        if not self.chunks and Path(path).suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "faiss_index" in data and data["faiss_index"].size:
                    index = self._faiss.deserialize_index(data["faiss_index"])
                    if self._is_own_type(index):
                        self._index = index
                        self.chunks = [chunks[i] for i in data["ids"].tolist()]
                        return
        ids, vectors = read_vectors(path)
        if ids:
            self._add_matrix([chunks[i] for i in ids], vectors)

    def _is_own_type(self, index: Any) -> bool:
        if self.index_type == "flat":
            return isinstance(index, self._faiss.IndexFlatIP)
        return isinstance(index, self._faiss.IndexHNSWFlat) and index.hnsw.nb_neighbors(1) == (
            self.hnsw_m
        )


def deserialize_vectors(blob: Any) -> Any:
    """The vectors stored in a serialized FAISS index, as a float32 matrix."""
    if blob.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    index = _import_faiss().deserialize_index(blob)
    return index.reconstruct_n(0, index.ntotal)
