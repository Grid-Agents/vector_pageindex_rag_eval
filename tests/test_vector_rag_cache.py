import sys
from types import SimpleNamespace

import numpy as np

from rag_eval.types import Document
from rag_eval.vector_rag import VectorRAG


class FakeSentenceTransformer:
    encode_calls = 0

    def __init__(self, _model: str):
        pass

    def encode(self, texts, **_kwargs):
        FakeSentenceTransformer.encode_calls += 1
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeCrossEncoder:
    def __init__(self, _model: str):
        pass


def test_vector_rag_reuses_cached_embeddings(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer,
            CrossEncoder=FakeCrossEncoder,
        ),
    )
    cfg = {
        "cache_dir": str(tmp_path),
        "chunk_strategy": "fixed",
        "chunk_size": 10,
        "chunk_overlap": 0,
        "embedding_model": "fake-embedder",
        "reranker": {"enabled": False},
    }
    documents = [Document("doc.txt", "0123456789abcdefghij")]

    FakeSentenceTransformer.encode_calls = 0
    first = VectorRAG(cfg)
    first.build(documents)

    assert FakeSentenceTransformer.encode_calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*.npz"))) == 1

    FakeSentenceTransformer.encode_calls = 0
    second = VectorRAG(cfg)
    second.build(documents)

    assert FakeSentenceTransformer.encode_calls == 0
    assert [chunk.text for chunk in second.chunks] == [
        chunk.text for chunk in first.chunks
    ]
    assert second.embeddings is not None
    assert first.embeddings is not None
    assert np.array_equal(second.embeddings, first.embeddings)


def test_hybrid_search_uses_bm25_signal(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer,
            CrossEncoder=FakeCrossEncoder,
        ),
    )
    cfg = {
        "cache_dir": str(tmp_path),
        "chunk_strategy": "fixed",
        "chunk_size": 80,
        "chunk_overlap": 0,
        "search_strategy": "hybrid",
        "embedding_model": "fake-embedder",
        "top_k": 1,
        "reranker": {"enabled": False},
    }
    documents = [
        Document("a.txt", "unrelated boilerplate"),
        Document("b.txt", "this clause contains the needle term"),
    ]

    rag = VectorRAG(cfg)
    rag.build(documents)
    result = rag.query("needle")

    assert len(result.spans) == 1
    assert result.spans[0].document_id == "b.txt"
    assert result.spans[0].metadata["search_strategy"] == "hybrid"
    assert result.spans[0].metadata["bm25_score"] > 0
