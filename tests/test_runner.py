import argparse

from rag_eval.runner import (
    _run_method,
    resolve_rlm_vector_tool_config,
    rlm_variant_configs,
    vector_variant_configs,
)
from rag_eval.vector_config import apply_vector_cli_overrides
from rag_eval.types import (
    Example,
    GoldSpan,
    LLMResponse,
    RetrievedSpan,
    RetrievalOutput,
    Usage,
)


class FakeSystem:
    def __init__(self, spans):
        self._spans = spans

    def query(self, query):
        return RetrievalOutput(spans=self._spans)


class FakeLLM:
    def complete(self, **kwargs):
        return LLMResponse(text="qualitative answer only", usage=Usage(output_tokens=3))


def test_run_method_scores_retrieval_when_answer_generation_is_disabled():
    example = Example(
        example_id="ex-1",
        benchmark="toy",
        query="What text?",
        gold_spans=[GoldSpan("doc.txt", 10, 20)],
    )
    system = FakeSystem([RetrievedSpan("doc.txt", 10, 20, "x" * 10)])

    result = _run_method(
        method_name="fake",
        system=system,
        example=example,
        llm=None,
        answer_with_llm=False,
        max_answer_context_chars=100,
    )

    assert result["evaluation_target"] == "retrieval"
    assert result["answer_generated"] is False
    assert result["answer"] == ""
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_run_method_generated_answer_does_not_change_retrieval_score():
    example = Example(
        example_id="ex-1",
        benchmark="toy",
        query="What text?",
        gold_spans=[GoldSpan("doc.txt", 10, 20)],
    )
    system = FakeSystem([RetrievedSpan("doc.txt", 10, 20, "x" * 10)])

    result = _run_method(
        method_name="fake",
        system=system,
        example=example,
        llm=FakeLLM(),
        answer_with_llm=True,
        max_answer_context_chars=100,
    )

    assert result["answer_generated"] is True
    assert result["answer"] == "qualitative answer only"
    assert result["output_tokens"] == 3
    assert result["f1"] == 1.0


def test_vector_variant_configs_expands_chunk_and_search_combinations():
    cfg = {
        "vector_rag": {
            "evaluate_combinations": True,
            "chunk_strategies": ["fixed", "semantic"],
            "search_strategies": ["vector", "hybrid"],
            "model_profiles": [
                {
                    "name": "bge",
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "fake-embedder",
                    "reranker": {"enabled": False},
                }
            ],
        }
    }

    variants = vector_variant_configs(cfg)

    assert [name for name, _ in variants] == [
        "vector_bge_fixed_vector",
        "vector_bge_fixed_hybrid",
        "vector_bge_semantic_vector",
        "vector_bge_semantic_hybrid",
    ]
    assert variants[3][1]["chunk_strategy"] == "semantic"
    assert variants[3][1]["search_strategy"] == "hybrid"


def test_include_voyage_adds_matching_model_profile():
    cfg = {
        "vector_rag": {
            "evaluate_combinations": True,
            "chunk_strategies": ["fixed", "semantic"],
            "search_strategies": ["vector", "hybrid"],
            "embedding_provider": "sentence_transformers",
            "embedding_model": "fake-embedder",
            "reranker": {
                "enabled": True,
                "provider": "sentence_transformers",
                "model": "fake-reranker",
                "top_k": 3,
            },
            "model_profiles": [
                {
                    "name": "bge",
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "fake-embedder",
                    "reranker": {"enabled": False},
                }
            ],
        }
    }
    args = argparse.Namespace(
        vector_combinations=False,
        no_vector_combinations=False,
        chunk_strategy="semantic",
        search_strategy="hybrid",
        top_k=None,
        rerank_top_k=None,
        embedding_provider=None,
        embedding_model=None,
        reranker_provider=None,
        reranker_model=None,
        include_voyage=True,
        voyage_embedding_model="voyage-4-large",
        voyage_reranker_model="rerank-2.5",
    )

    apply_vector_cli_overrides(cfg["vector_rag"], args)
    variants = vector_variant_configs(cfg)

    assert [name for name, _ in variants] == [
        "vector_bge_semantic_hybrid",
        "vector_voyage_semantic_hybrid",
    ]
    voyage_cfg = variants[1][1]
    assert voyage_cfg["embedding_provider"] == "voyage"
    assert voyage_cfg["query_instruction"] == ""
    assert voyage_cfg["reranker"]["provider"] == "voyage"


def test_rlm_variant_configs_builds_pageindex_from_baseline():
    cfg = {
        "run": {"methods": ["rlm", "rlm_pageindex"]},
        "rlm": {
            "backend": "openai",
            "selected_spans": 5,
        },
        "rlm_pageindex": {
            "selected_spans": 8,
            "prompt_style": "pageindex",
            "pageindex_tool": {"enabled": True},
        },
        "vector_rag": {
            "cache_dir": ".cache/vector",
            "chunk_size": 1200,
            "chunk_overlap": 120,
            "semantic_chunking": {"break_percentile": 82, "window_size": 3},
            "hybrid": {"vector_weight": 0.65, "bm25_weight": 0.35},
            "model_profiles": [
                {
                    "name": "voyage",
                    "embedding_provider": "voyage",
                    "embedding_model": "voyage-law-2",
                    "reranker": {"enabled": True, "provider": "voyage", "model": "rerank-2", "top_k": 5},
                }
            ],
        },
        "pageindex": {"cache_dir": ".cache/pageindex"},
    }

    variants = rlm_variant_configs(cfg)

    assert [name for name, _, _, _ in variants] == ["rlm", "rlm_pageindex"]
    pageindex_cfg = variants[1][1]
    assert pageindex_cfg["backend"] == "openai"
    assert pageindex_cfg["selected_spans"] == 8
    assert pageindex_cfg["prompt_style"] == "pageindex"
    assert variants[1][2] is None
    assert variants[1][3] is not None


def test_resolve_rlm_vector_tool_config_pins_semantic_hybrid_voyage():
    cfg = {
        "vector_rag": {
            "cache_dir": ".cache/vector",
            "chunk_strategy": "hierarchical",
            "search_strategy": "vector",
            "chunk_size": 1200,
            "chunk_overlap": 120,
            "semantic_chunking": {"break_percentile": 82, "window_size": 3},
            "hybrid": {"vector_weight": 0.65, "bm25_weight": 0.35},
            "model_profiles": [
                {
                    "name": "bge",
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "fake-bge",
                    "reranker": {"enabled": False},
                },
                {
                    "name": "voyage",
                    "embedding_provider": "voyage",
                    "embedding_model": "voyage-law-2",
                    "reranker": {"enabled": True, "provider": "voyage", "model": "rerank-2", "top_k": 5},
                },
            ],
        }
    }
    rlm_cfg = {
        "vector_tool": {
            "enabled": True,
            "model_profile": "voyage",
            "chunk_strategy": "semantic",
            "search_strategy": "hybrid",
            "max_results": 8,
        }
    }

    vector_tool_cfg = resolve_rlm_vector_tool_config(cfg, rlm_cfg)

    assert vector_tool_cfg is not None
    assert vector_tool_cfg["embedding_provider"] == "voyage"
    assert vector_tool_cfg["embedding_model"] == "voyage-law-2"
    assert vector_tool_cfg["chunk_strategy"] == "semantic"
    assert vector_tool_cfg["search_strategy"] == "hybrid"
    assert vector_tool_cfg["reranker"]["provider"] == "voyage"
    assert vector_tool_cfg["reranker"]["top_k"] >= 8
