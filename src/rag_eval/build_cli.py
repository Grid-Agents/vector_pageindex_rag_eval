from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path, split_csv
from .data import LegalBenchRAGLoader
from .llm import AnthropicLLM
from .official_pageindex import OfficialPageIndexRAG
from .pageindex_rag import PageIndexRAG
from .rlm_rag import RLMRAG
from .runner import (
    PROJECT_ROOT,
    resolve_methods,
    rlm_variant_configs,
    vector_variant_configs,
)
from .vector_rag import VectorRAG
from .vector_config import add_vector_cli_args, apply_vector_cli_overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector and/or PageIndex caches without running queries."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", help="Override LegalBench-RAG data directory.")
    parser.add_argument(
        "--benchmark",
        "--benchmarks",
        dest="benchmarks",
        help="Benchmark name, comma-separated benchmark names, or all.",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated methods to build: vector,pageindex,pageindex_official,rlm,rlm_recall_plus.",
    )
    parser.add_argument("--n", type=int, help="Number of sampled examples for sampled scope.")
    parser.add_argument("--seed", type=int, help="Sample seed.")
    parser.add_argument("--corpus-scope", choices=["sampled", "all"])
    add_vector_cli_args(parser)
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild the selected method caches instead of reusing them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    apply_overrides(cfg, args)
    build_indexes(cfg)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("data", {})
    cfg.setdefault("run", {})
    cfg.setdefault("vector_rag", {})
    cfg.setdefault("vector_rag", {}).setdefault("reranker", {})
    cfg.setdefault("pageindex", {})
    cfg.setdefault("pageindex_official", {})
    cfg.setdefault("rlm", {})
    cfg.setdefault("rlm_recall_plus", {})

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.benchmarks:
        cfg["data"]["benchmarks"] = split_csv(args.benchmarks)
    if args.methods:
        cfg["run"]["methods"] = split_csv(args.methods)
    if args.n is not None:
        cfg["run"]["n"] = args.n
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.corpus_scope:
        cfg["data"]["corpus_scope"] = args.corpus_scope
    apply_vector_cli_overrides(cfg["vector_rag"], args)
    if args.force_reindex:
        methods = set(_resolve_methods(cfg))
        cfg["vector_rag"]["force_reindex"] = (
            "vector" in methods or "rlm_recall_plus" in methods
        )
        cfg["pageindex"]["force_reindex"] = "pageindex" in methods
        cfg["pageindex_official"]["force_reindex"] = "pageindex_official" in methods


def build_indexes(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    run_cfg = cfg.get("run", {})
    methods = _resolve_methods(cfg)

    loader = LegalBenchRAGLoader(
        resolve_path(PROJECT_ROOT, data_cfg.get("data_dir", "data"))
    )
    corpus_scope = data_cfg.get("corpus_scope", "sampled")
    example_limit = None if corpus_scope == "all" else run_cfg.get("n")
    examples = loader.load_examples(
        data_cfg.get("benchmarks"),
        n=example_limit,
        seed=int(run_cfg.get("seed", 42)),
    )
    documents = loader.load_documents(examples, corpus_scope=corpus_scope)
    print(f"Loaded {len(documents)} documents for cache building.")

    if "vector" in methods:
        for method_name, vector_cfg in vector_variant_configs(cfg):
            vector = VectorRAG(
                vector_cfg,
                cache_dir=resolve_path(
                    PROJECT_ROOT, vector_cfg.get("cache_dir", ".cache/vector")
                ),
            )
            print(f"Building {method_name} index into {vector.cache_dir}...")
            vector.build(documents)
            print(f"{method_name} index ready in {vector.cache_dir}")

    if "pageindex" in methods:
        pageindex_cfg = cfg.get("pageindex", {})
        llm = None
        if pageindex_cfg.get("build_with_llm", True):
            llm = AnthropicLLM(cfg.get("llm", {}))
        pageindex = PageIndexRAG(
            pageindex_cfg,
            llm,
            cache_dir=resolve_path(
                PROJECT_ROOT, pageindex_cfg.get("cache_dir", ".cache/pageindex")
            ),
        )
        print(f"Building PageIndex trees into {pageindex.cache_dir}...")
        pageindex.build(documents)
        print(
            "PageIndex trees ready in "
            f"{pageindex.cache_dir} "
            f"(setup cost ${pageindex.setup_usage.estimated_cost_usd:.4f})"
        )

    if "pageindex_official" in methods:
        official_cfg = cfg.get("pageindex_official", {})
        llm = None
        if official_cfg.get("build_with_llm", True):
            llm = AnthropicLLM(cfg.get("llm", {}))
        official_pageindex = OfficialPageIndexRAG(
            official_cfg,
            llm,
            cache_dir=resolve_path(
                PROJECT_ROOT,
                official_cfg.get("cache_dir", ".cache/pageindex_official"),
            ),
        )
        print(f"Building official PageIndex trees into {official_pageindex.cache_dir}...")
        official_pageindex.build(documents)
        print(
            "Official PageIndex trees ready in "
            f"{official_pageindex.cache_dir} "
            f"(setup cost ${official_pageindex.setup_usage.estimated_cost_usd:.4f})"
        )

    for method_name, rlm_cfg, vector_tool_cfg in rlm_variant_configs(cfg):
        if method_name not in methods:
            continue
        if vector_tool_cfg is None:
            print(
                f"{method_name} has no build-time index to warm; it searches documents at query time."
            )
            continue
        vector_tool_cache_dir = resolve_path(
            PROJECT_ROOT, vector_tool_cfg.get("cache_dir", ".cache/vector")
        )
        rlm = RLMRAG(
            rlm_cfg,
            llm_cfg=cfg.get("llm", {}),
            method_name=method_name,
            vector_tool_cfg=vector_tool_cfg,
            vector_tool_cache_dir=vector_tool_cache_dir,
        )
        print(f"Building {method_name} vector helper into {vector_tool_cache_dir}...")
        rlm.build(documents)
        print(f"{method_name} helper index ready in {vector_tool_cache_dir}")


def _resolve_methods(cfg: dict[str, Any]) -> list[str]:
    methods = resolve_methods(cfg.get("run", {}).get("methods", ["pageindex"]))
    if not methods:
        raise ValueError("No methods selected.")
    return methods


if __name__ == "__main__":
    main()
