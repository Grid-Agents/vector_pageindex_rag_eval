from __future__ import annotations

from typing import Any

import numpy as np

from .text_splitters import make_chunks
from .types import Chunk, Document, RetrievedSpan, RetrievalOutput


class VectorRAG:
    name = "vector"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedder: Any = None
        self.reranker: Any = None

    def build(self, documents: list[Document]) -> None:
        self.chunks = []
        for document in documents:
            self.chunks.extend(
                make_chunks(
                    document,
                    strategy=self.cfg.get("chunk_strategy", "hierarchical"),
                    chunk_size=int(self.cfg.get("chunk_size", 1200)),
                    chunk_overlap=int(self.cfg.get("chunk_overlap", 120)),
                )
            )
        if not self.chunks:
            raise RuntimeError("No chunks built for vector RAG.")

        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .`.") from exc

        self.embedder = SentenceTransformer(self.cfg.get("embedding_model"))
        texts = [self._chunk_embedding_text(chunk) for chunk in self.chunks]
        self.embeddings = np.asarray(
            self.embedder.encode(
                texts,
                batch_size=int(self.cfg.get("batch_size", 32)),
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

        reranker_cfg = self.cfg.get("reranker", {})
        if reranker_cfg.get("enabled", True):
            self.reranker = CrossEncoder(reranker_cfg.get("model"))

    def query(self, query: str) -> RetrievalOutput:
        if self.embeddings is None or self.embedder is None:
            raise RuntimeError("Call build() before query().")

        query_text = f"{self.cfg.get('query_instruction', '')}{query}"
        query_embedding = np.asarray(
            self.embedder.encode([query_text], normalize_embeddings=True),
            dtype=np.float32,
        )[0]
        scores = self.embeddings @ query_embedding
        top_k = min(int(self.cfg.get("top_k", 20)), len(self.chunks))
        candidate_indices = np.argsort(-scores)[:top_k].tolist()

        if self.reranker is not None and candidate_indices:
            pairs = [
                (query, self._chunk_embedding_text(self.chunks[i]))
                for i in candidate_indices
            ]
            rerank_scores = np.asarray(self.reranker.predict(pairs), dtype=np.float32)
            order = np.argsort(-rerank_scores)
            rerank_top_k = min(
                int(self.cfg.get("reranker", {}).get("top_k", 5)),
                len(candidate_indices),
            )
            selected = [
                (candidate_indices[int(i)], float(rerank_scores[int(i)]))
                for i in order[:rerank_top_k]
            ]
        else:
            selected = [(i, float(scores[i])) for i in candidate_indices]

        spans = [
            RetrievedSpan(
                document_id=self.chunks[idx].document_id,
                start_char=self.chunks[idx].start_char,
                end_char=self.chunks[idx].end_char,
                text=self.chunks[idx].text,
                score=score,
                metadata={
                    "chunk_title": self.chunks[idx].title,
                    "chunk_level": self.chunks[idx].level,
                    "retriever": "vector",
                },
            )
            for idx, score in selected
        ]
        return RetrievalOutput(spans=spans)

    def _chunk_embedding_text(self, chunk: Chunk) -> str:
        if chunk.title:
            return f"{chunk.title}\n{chunk.text}"
        return chunk.text
