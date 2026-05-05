from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .text_splitters import make_chunks
from .types import Chunk, Document, RetrievedSpan, RetrievalOutput


_CACHE_VERSION = 2
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


class SentenceTransformerEmbedder:
    provider = "sentence_transformers"

    def __init__(self, model_name: str, *, query_instruction: str = ""):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .`.") from exc

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
        desc: str | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        query_texts = [f"{self.query_instruction}{text}" for text in texts]
        return np.asarray(
            self.model.encode(query_texts, normalize_embeddings=True),
            dtype=np.float32,
        )


class VoyageEmbedder:
    provider = "voyage"

    def __init__(
        self,
        model_name: str,
        *,
        api_key_env: str = "VOYAGE_API_KEY",
        output_dimension: int | None = None,
        query_instruction: str = "",
        truncation: bool = True,
    ):
        try:
            import voyageai
        except ImportError as exc:
            raise RuntimeError(
                "Install Voyage support with `uv add voyageai` and set VOYAGE_API_KEY."
            ) from exc

        api_key = os.getenv(api_key_env)
        self.client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        self.model_name = model_name
        self.output_dimension = output_dimension
        self.query_instruction = query_instruction
        self.truncation = truncation

    def encode_documents(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
        desc: str | None = None,
    ) -> np.ndarray:
        return self._embed(
            texts,
            input_type="document",
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            desc=desc or "Voyage document embeddings",
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        query_texts = [f"{self.query_instruction}{text}" for text in texts]
        return self._embed(
            query_texts,
            input_type="query",
            batch_size=len(query_texts) or 1,
            show_progress_bar=False,
            desc="Voyage query embeddings",
        )

    def _embed(
        self,
        texts: list[str],
        *,
        input_type: str,
        batch_size: int,
        show_progress_bar: bool,
        desc: str,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings: list[list[float]] = []
        effective_batch_size = max(1, min(int(batch_size), 1000))
        ranges = range(0, len(texts), effective_batch_size)
        iterator = tqdm(
            ranges,
            desc=desc,
            unit="batch",
            disable=not show_progress_bar or len(texts) <= effective_batch_size,
        )
        for start in iterator:
            batch = texts[start : start + effective_batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "input_type": input_type,
                "truncation": self.truncation,
            }
            if self.output_dimension is not None:
                kwargs["output_dimension"] = self.output_dimension
            result = self.client.embed(batch, **kwargs)
            embeddings.extend(result.embeddings)

        return _normalize_rows(np.asarray(embeddings, dtype=np.float32))


class SentenceTransformerReranker:
    provider = "sentence_transformers"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .`.") from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rank(
        self, query: str, documents: list[str], *, top_k: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[int(idx)])) for idx in order]


class VoyageReranker:
    provider = "voyage"

    def __init__(
        self,
        model_name: str,
        *,
        api_key_env: str = "VOYAGE_API_KEY",
        truncation: bool = True,
    ):
        try:
            import voyageai
        except ImportError as exc:
            raise RuntimeError(
                "Install Voyage support with `uv add voyageai` and set VOYAGE_API_KEY."
            ) from exc

        api_key = os.getenv(api_key_env)
        self.client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()
        self.model_name = model_name
        self.truncation = truncation

    def rank(
        self, query: str, documents: list[str], *, top_k: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        result = self.client.rerank(
            query,
            documents,
            model=self.model_name,
            top_k=min(top_k, len(documents)),
            truncation=self.truncation,
        )
        ranked: list[tuple[int, float]] = []
        for item in getattr(result, "results", []):
            index = int(getattr(item, "index"))
            score = float(
                getattr(item, "relevance_score", getattr(item, "score", 0.0))
            )
            ranked.append((index, score))
        return ranked


class BM25Index:
    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}

        dfs: Counter[str] = Counter()
        for doc_idx, text in enumerate(texts):
            counts = Counter(_tokenize(text))
            self.doc_lengths.append(sum(counts.values()))
            for term, count in counts.items():
                dfs[term] += 1
                self.postings.setdefault(term, []).append((doc_idx, count))

        self.document_count = len(texts)
        self.avgdl = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self.idf = {
            term: math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.document_count, dtype=np.float32)
        if self.document_count == 0 or self.avgdl <= 0:
            return scores

        for term, query_tf in Counter(_tokenize(query)).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, term_tf in postings:
                length = self.doc_lengths[doc_idx] or 1
                denominator = term_tf + self.k1 * (
                    1.0 - self.b + self.b * length / self.avgdl
                )
                scores[doc_idx] += (
                    idf * query_tf * (term_tf * (self.k1 + 1.0)) / denominator
                )
        return scores


class VectorRAG:
    name = "vector"

    def __init__(self, cfg: dict[str, Any], *, cache_dir: Path | None = None):
        self.cfg = cfg
        self.cache_dir = cache_dir or Path(cfg.get("cache_dir", ".cache/vector"))
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedder: Any = None
        self.reranker: Any = None
        self.bm25: BM25Index | None = None

    def build(self, documents: list[Document]) -> None:
        cache_key = self._cache_key(documents)
        cache_loaded = self._load_cached_index(cache_key, documents)

        # The embedder is needed for semantic chunking on cache misses and for
        # query embeddings even when document embeddings are restored from cache.
        self.embedder = self._make_embedder()
        if not cache_loaded:
            self.chunks = self._build_chunks(documents)
            if not self.chunks:
                raise RuntimeError("No chunks built for vector RAG.")
            texts = [self._chunk_embedding_text(chunk) for chunk in self.chunks]
            self.embeddings = self.embedder.encode_documents(
                texts,
                batch_size=int(self.cfg.get("batch_size", 32)),
                show_progress_bar=True,
                desc="Vector embeddings",
            )
            self._write_cached_index(cache_key, documents)

        if self._search_strategy() == "hybrid":
            self.bm25 = BM25Index(
                [self._chunk_embedding_text(chunk) for chunk in self.chunks],
                k1=float(self.cfg.get("hybrid", {}).get("bm25_k1", 1.5)),
                b=float(self.cfg.get("hybrid", {}).get("bm25_b", 0.75)),
            )

        reranker_cfg = self.cfg.get("reranker", {})
        if reranker_cfg.get("enabled", True):
            self.reranker = self._make_reranker(reranker_cfg)

    def query(self, query: str) -> RetrievalOutput:
        if self.embeddings is None or self.embedder is None:
            raise RuntimeError("Call build() before query().")

        query_embedding = np.asarray(
            self.embedder.encode_queries([query]),
            dtype=np.float32,
        )[0]
        candidate_indices, retrieval_scores, retrieval_metadata = (
            self._retrieve_candidates(query, query_embedding)
        )

        if self.reranker is not None and candidate_indices:
            candidate_texts = [
                self._chunk_embedding_text(self.chunks[i]) for i in candidate_indices
            ]
            rerank_top_k = min(
                int(self.cfg.get("reranker", {}).get("top_k", 5)),
                len(candidate_indices),
            )
            selected = [
                (candidate_indices[local_idx], score)
                for local_idx, score in self.reranker.rank(
                    query, candidate_texts, top_k=rerank_top_k
                )
            ]
        else:
            selected = [(i, float(retrieval_scores[i])) for i in candidate_indices]

        search_strategy = self._search_strategy()
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
                    "search_strategy": search_strategy,
                    **retrieval_metadata.get(idx, {}),
                },
            )
            for idx, score in selected
        ]
        return RetrievalOutput(spans=spans, metadata={"search_strategy": search_strategy})

    def _retrieve_candidates(
        self, query: str, query_embedding: np.ndarray
    ) -> tuple[list[int], np.ndarray, dict[int, dict[str, float]]]:
        if self.embeddings is None:
            raise RuntimeError("Call build() before query().")

        vector_scores = self.embeddings @ query_embedding
        top_k = min(int(self.cfg.get("top_k", 20)), len(self.chunks))
        strategy = self._search_strategy()
        metadata: dict[int, dict[str, float]] = {}

        if strategy == "vector":
            retrieval_scores = vector_scores
            candidate_indices = np.argsort(-retrieval_scores)[:top_k].tolist()
            for idx in candidate_indices:
                metadata[idx] = {"vector_score": float(vector_scores[idx])}
            return candidate_indices, retrieval_scores, metadata

        if self.bm25 is None:
            self.bm25 = BM25Index(
                [self._chunk_embedding_text(chunk) for chunk in self.chunks],
                k1=float(self.cfg.get("hybrid", {}).get("bm25_k1", 1.5)),
                b=float(self.cfg.get("hybrid", {}).get("bm25_b", 0.75)),
            )

        bm25_scores = self.bm25.score(query)
        hybrid_cfg = self.cfg.get("hybrid", {})
        vector_weight = float(hybrid_cfg.get("vector_weight", 0.65))
        bm25_weight = float(hybrid_cfg.get("bm25_weight", 0.35))
        retrieval_scores = (
            vector_weight * _minmax_normalize(vector_scores)
            + bm25_weight * _minmax_normalize(bm25_scores)
        )
        candidate_indices = np.argsort(-retrieval_scores)[:top_k].tolist()
        for idx in candidate_indices:
            metadata[idx] = {
                "vector_score": float(vector_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "hybrid_score": float(retrieval_scores[idx]),
            }
        return candidate_indices, retrieval_scores, metadata

    def _chunk_embedding_text(self, chunk: Chunk) -> str:
        if chunk.title:
            return f"{chunk.title}\n{chunk.text}"
        return chunk.text

    def _build_chunks(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        semantic_cfg = self.cfg.get("semantic_chunking", {})
        semantic_embedding_fn = None
        if self.cfg.get("chunk_strategy", "hierarchical") == "semantic":
            if self.embedder is None:
                raise RuntimeError("Semantic chunking requires an embedder.")

            def semantic_embedding_fn(texts: list[str]) -> np.ndarray:
                return self.embedder.encode_documents(
                    texts,
                    batch_size=int(self.cfg.get("batch_size", 32)),
                    show_progress_bar=False,
                    desc="Semantic chunk embeddings",
                )

        for document in tqdm(
            documents,
            desc="Vector chunking",
            unit="doc",
            disable=len(documents) <= 1,
        ):
            chunks.extend(
                make_chunks(
                    document,
                    strategy=self.cfg.get("chunk_strategy", "hierarchical"),
                    chunk_size=int(self.cfg.get("chunk_size", 1200)),
                    chunk_overlap=int(self.cfg.get("chunk_overlap", 120)),
                    semantic_embedding_fn=semantic_embedding_fn,
                    semantic_break_percentile=float(
                        semantic_cfg.get("break_percentile", 82.0)
                    ),
                    semantic_window_size=int(semantic_cfg.get("window_size", 3)),
                    semantic_min_chunk_size=semantic_cfg.get("min_chunk_size"),
                )
            )
        return chunks

    def _make_embedder(self) -> Any:
        provider = _normalize_provider(
            self.cfg.get("embedding_provider", "sentence_transformers")
        )
        model = str(self.cfg.get("embedding_model"))
        query_instruction = str(self.cfg.get("query_instruction", ""))
        if provider == "sentence_transformers":
            return SentenceTransformerEmbedder(
                model, query_instruction=query_instruction
            )
        if provider == "voyage":
            output_dimension = self.cfg.get("embedding_output_dimension")
            return VoyageEmbedder(
                model,
                api_key_env=str(self.cfg.get("voyage_api_key_env", "VOYAGE_API_KEY")),
                output_dimension=(
                    int(output_dimension) if output_dimension is not None else None
                ),
                query_instruction=query_instruction,
                truncation=bool(self.cfg.get("embedding_truncation", True)),
            )
        raise ValueError(f"Unknown embedding provider: {provider}")

    def _make_reranker(self, reranker_cfg: dict[str, Any]) -> Any:
        provider = _normalize_provider(
            reranker_cfg.get("provider", "sentence_transformers")
        )
        model = str(reranker_cfg.get("model"))
        if provider == "sentence_transformers":
            return SentenceTransformerReranker(model)
        if provider == "voyage":
            return VoyageReranker(
                model,
                api_key_env=str(
                    reranker_cfg.get(
                        "voyage_api_key_env",
                        self.cfg.get("voyage_api_key_env", "VOYAGE_API_KEY"),
                    )
                ),
                truncation=bool(reranker_cfg.get("truncation", True)),
            )
        raise ValueError(f"Unknown reranker provider: {provider}")

    def _search_strategy(self) -> str:
        strategy = str(self.cfg.get("search_strategy", "vector")).lower()
        if strategy in {"vector", "semantic", "semantic_vector", "dense"}:
            return "vector"
        if strategy in {"hybrid", "bm25_vector", "bm25+vector", "bm25+semantic"}:
            return "hybrid"
        raise ValueError(f"Unknown vector search strategy: {strategy}")

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
        semantic_cfg = self.cfg.get("semantic_chunking", {})
        chunk_strategy = self.cfg.get("chunk_strategy", "hierarchical")
        return {
            "chunk_strategy": chunk_strategy,
            "chunk_size": int(self.cfg.get("chunk_size", 1200)),
            "chunk_overlap": int(self.cfg.get("chunk_overlap", 120)),
            "semantic_chunking": (
                {
                    "break_percentile": float(
                        semantic_cfg.get("break_percentile", 82.0)
                    ),
                    "window_size": int(semantic_cfg.get("window_size", 3)),
                    "min_chunk_size": semantic_cfg.get("min_chunk_size"),
                }
                if chunk_strategy == "semantic"
                else None
            ),
            "embedding_provider": _normalize_provider(
                self.cfg.get("embedding_provider", "sentence_transformers")
            ),
            "embedding_model": self.cfg.get("embedding_model"),
            "embedding_output_dimension": self.cfg.get("embedding_output_dimension"),
            "embedding_truncation": bool(self.cfg.get("embedding_truncation", True)),
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


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score - min_score <= 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


def _normalize_provider(provider: Any) -> str:
    normalized = str(provider).lower().replace("-", "_")
    aliases = {
        "sentence_transformer": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
        "sbert": "sentence_transformers",
        "voyage": "voyage",
        "voyageai": "voyage",
        "voyage_ai": "voyage",
    }
    return aliases.get(normalized, normalized)
