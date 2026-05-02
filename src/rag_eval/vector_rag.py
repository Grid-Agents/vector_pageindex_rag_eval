from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .text_splitters import make_chunks
from .types import Chunk, Document, RetrievedSpan, RetrievalOutput


_CACHE_VERSION = 1


class VectorRAG:
    name = "vector"

    def __init__(self, cfg: dict[str, Any], *, cache_dir: Path | None = None):
        self.cfg = cfg
        self.cache_dir = cache_dir or Path(cfg.get("cache_dir", ".cache/vector"))
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedder: Any = None
        self.reranker: Any = None

    def build(self, documents: list[Document]) -> None:
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .`.") from exc

        cache_key = self._cache_key(documents)
        cache_loaded = self._load_cached_index(cache_key, documents)
        if not cache_loaded:
            self.chunks = self._build_chunks(documents)
            if not self.chunks:
                raise RuntimeError("No chunks built for vector RAG.")

        # The embedder is still needed for query embeddings even when document
        # embeddings are restored from cache.
        self.embedder = SentenceTransformer(self.cfg.get("embedding_model"))
        if not cache_loaded:
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
            self._write_cached_index(cache_key, documents)

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

    def _build_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                make_chunks(
                    document,
                    strategy=self.cfg.get("chunk_strategy", "hierarchical"),
                    chunk_size=int(self.cfg.get("chunk_size", 1200)),
                    chunk_overlap=int(self.cfg.get("chunk_overlap", 120)),
                )
            )
        return chunks

    def _cache_key(self, documents: list[Document]) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "index_config": self._index_config(),
            "documents": [
                {
                    "document_id": doc.document_id,
                    "length": len(doc.text),
                    "sha1": hashlib.sha1(doc.text.encode("utf-8")).hexdigest(),
                }
                for doc in documents
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _index_config(self) -> dict[str, Any]:
        return {
            "chunk_strategy": self.cfg.get("chunk_strategy", "hierarchical"),
            "chunk_size": int(self.cfg.get("chunk_size", 1200)),
            "chunk_overlap": int(self.cfg.get("chunk_overlap", 120)),
            "embedding_model": self.cfg.get("embedding_model"),
            "embedding_text": "title_newline_text_v1",
            "normalize_embeddings": True,
        }

    def _cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{cache_key}.json",
            self.cache_dir / f"{cache_key}.npz",
        )

    def _load_cached_index(self, cache_key: str, documents: list[Document]) -> bool:
        if self.cfg.get("force_reindex", False):
            return False

        metadata_path, embeddings_path = self._cache_paths(cache_key)
        if not metadata_path.exists() or not embeddings_path.exists():
            return False

        try:
            docs_by_id = {doc.document_id: doc for doc in documents}
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            chunks = []
            for item in metadata["chunks"]:
                document_id = str(item["document_id"])
                start_char = int(item["start_char"])
                end_char = int(item["end_char"])
                doc = docs_by_id[document_id]
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        start_char=start_char,
                        end_char=end_char,
                        text=doc.text[start_char:end_char],
                        title=str(item.get("title", "")),
                        level=int(item.get("level", 0)),
                    )
                )
            with np.load(embeddings_path) as data:
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        except Exception:  # noqa: BLE001
            return False

        if len(chunks) == 0 or embeddings.shape[0] != len(chunks):
            return False

        self.chunks = chunks
        self.embeddings = embeddings
        return True

    def _write_cached_index(self, cache_key: str, documents: list[Document]) -> None:
        if self.embeddings is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path, embeddings_path = self._cache_paths(cache_key)
        metadata = {
            "version": _CACHE_VERSION,
            "index_config": self._index_config(),
            "documents": [
                {
                    "document_id": doc.document_id,
                    "length": len(doc.text),
                    "sha1": hashlib.sha1(doc.text.encode("utf-8")).hexdigest(),
                }
                for doc in documents
            ],
            "chunks": [
                {
                    "document_id": chunk.document_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "title": chunk.title,
                    "level": chunk.level,
                }
                for chunk in self.chunks
            ],
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        np.savez_compressed(embeddings_path, embeddings=self.embeddings)
