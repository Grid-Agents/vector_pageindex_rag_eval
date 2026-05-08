import argparse

from rag_eval.build_cli import _resolve_methods, apply_overrides


def test_build_cli_force_reindex_only_selected_method():
    cfg = {
        "run": {"methods": ["pageindex"]},
        "vector_rag": {"force_reindex": False, "reranker": {}},
        "pageindex": {"force_reindex": False},
        "pageindex_official": {"force_reindex": False},
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
    assert cfg["pageindex_official"]["force_reindex"] is False


def test_build_cli_resolve_methods_rejects_unknown_method():
    cfg = {"run": {"methods": ["vector", "bogus"]}}

    try:
        _resolve_methods(cfg)
    except ValueError as exc:
        assert "Unknown method(s)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown build method")


def test_build_cli_resolve_methods_accepts_official_pageindex():
    cfg = {"run": {"methods": ["vector", "pageindex_official"]}}

    assert _resolve_methods(cfg) == ["vector", "pageindex_official"]


def test_build_cli_resolve_methods_accepts_rlm():
    cfg = {"run": {"methods": ["rlm"]}}

    assert _resolve_methods(cfg) == ["rlm"]


def test_build_cli_resolve_methods_accepts_rlm_recall_plus():
    cfg = {"run": {"methods": ["rlm_recall_plus"]}}

    assert _resolve_methods(cfg) == ["rlm_recall_plus"]


def test_build_cli_force_reindex_for_rlm_recall_plus_enables_vector_cache():
    cfg = {
        "run": {"methods": ["rlm_recall_plus"]},
        "vector_rag": {"force_reindex": False, "reranker": {}},
        "pageindex": {"force_reindex": False},
        "pageindex_official": {"force_reindex": False},
        "rlm": {},
        "rlm_recall_plus": {},
        "data": {},
    }
    args = argparse.Namespace(
        data_dir=None,
        benchmarks=None,
        methods="rlm_recall_plus",
        n=None,
        seed=None,
        corpus_scope=None,
        chunk_strategy=None,
        top_k=None,
        rerank_top_k=None,
        force_reindex=True,
    )

    apply_overrides(cfg, args)

    assert cfg["vector_rag"]["force_reindex"] is True
