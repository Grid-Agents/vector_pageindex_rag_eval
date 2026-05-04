from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path, split_csv
from .data import LegalBenchRAGLoader
from .llm import AnthropicLLM
from .official_pageindex import OfficialPageIndexRAG
from .pageindex_rag import PageIndexRAG
from .runner import PROJECT_ROOT
from .vector_rag import VectorRAG


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
        help="Comma-separated methods to build: vector,pageindex,pageindex_official.",
    )
    parser.add_argument("--n", type=int, help="Number of sampled examples for sampled scope.")
    parser.add_argument("--seed", type=int, help="Sample seed.")
    parser.add_argument("--corpus-scope", choices=["sampled", "all"])
    parser.add_argument(
        "--chunk-strategy", choices=["hierarchical", "recursive", "fixed"]
    )
    parser.add_argument("--top-k", type=int, help="Vector retrieval candidate top-k.")
    parser.add_argument("--rerank-top-k", type=int, help="Reranker output top-k.")
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
    if args.chunk_strategy:
        cfg["vector_rag"]["chunk_strategy"] = args.chunk_strategy
    if args.top_k is not None:
        cfg["vector_rag"]["top_k"] = args.top_k
    if args.rerank_top_k is not None:
        cfg["vector_rag"]["reranker"]["top_k"] = args.rerank_top_k
    if args.force_reindex:
        methods = set(_resolve_methods(cfg))
        cfg["vector_rag"]["force_reindex"] = "vector" in methods
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
        vector_cfg = cfg.get("vector_rag", {})
        vector = VectorRAG(
            vector_cfg,
            cache_dir=resolve_path(
                PROJECT_ROOT, vector_cfg.get("cache_dir", ".cache/vector")
            ),
        )
        print(f"Building vector index into {vector.cache_dir}...")
        vector.build(documents)
        print(f"Vector index ready in {vector.cache_dir}")

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


def _resolve_methods(cfg: dict[str, Any]) -> list[str]:
    methods = [name.lower() for name in cfg.get("run", {}).get("methods", ["pageindex"])]
    unknown_methods = sorted(set(methods) - {"vector", "pageindex", "pageindex_official"})
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}")
    if not methods:
        raise ValueError("No methods selected.")
    return methods


if __name__ == "__main__":
    main()
