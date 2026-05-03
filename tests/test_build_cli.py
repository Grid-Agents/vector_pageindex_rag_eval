import argparse

from rag_eval.build_cli import _resolve_methods, apply_overrides


def test_build_cli_force_reindex_only_selected_method():
    cfg = {
        "run": {"methods": ["pageindex"]},
        "vector_rag": {"force_reindex": False, "reranker": {}},
        "pageindex": {"force_reindex": False},
        "data": {},
    }
    args = argparse.Namespace(
        data_dir=None,
        benchmarks=None,
        methods="pageindex",
        n=None,
        seed=None,
        corpus_scope=None,
        chunk_strategy=None,
        top_k=None,
        rerank_top_k=None,
        force_reindex=True,
    )

    apply_overrides(cfg, args)

    assert cfg["pageindex"]["force_reindex"] is True
    assert cfg["vector_rag"]["force_reindex"] is False


def test_build_cli_resolve_methods_rejects_unknown_method():
    cfg = {"run": {"methods": ["vector", "bogus"]}}

    try:
        _resolve_methods(cfg)
    except ValueError as exc:
        assert "Unknown method(s)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown build method")
