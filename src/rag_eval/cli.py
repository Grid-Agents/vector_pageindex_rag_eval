from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, split_csv
from .runner import PROJECT_ROOT, run_experiment
from .vector_config import add_vector_cli_args, apply_vector_cli_overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LegalBench-RAG experiments for vector RAG and PageIndex."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", help="Override LegalBench-RAG data directory.")
    parser.add_argument("--results-dir", help="Override results directory.")
    parser.add_argument(
        "--benchmark",
        "--benchmarks",
        dest="benchmarks",
        help="Benchmark name, comma-separated benchmark names, or all.",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated methods: vector,pageindex,pageindex_official,rlm,rlm_recall_plus.",
    )
    parser.add_argument("--n", type=int, help="Number of examples to run.")
    parser.add_argument("--seed", type=int, help="Sample seed.")
    parser.add_argument("--corpus-scope", choices=["sampled", "all"])
    add_vector_cli_args(parser)
    parser.add_argument(
        "--answer-with-llm",
        action="store_true",
        help="Generate qualitative answers after retrieval. Answers are not scored.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation. This is the default.",
    )
    parser.add_argument(
        "--record-reasoning-trajectory",
        action="store_true",
        help="Record PageIndex traces and RLM turn trajectories.",
    )
    parser.add_argument(
        "--no-record-reasoning-trajectory",
        action="store_true",
        help="Disable PageIndex/RLM reasoning trace recording.",
    )
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild vector and PageIndex caches.")
    parser.add_argument("--run-id", help="Custom run id.")
    parser.add_argument(
        "--merge-results",
        "--combine-results",
        action="store_true",
        help=(
            "Merge this run's method results into an existing --run-id directory "
            "instead of replacing the combined run record."
        ),
    )
    parser.add_argument(
        "--merge-into-run",
        help="Existing run id or results path to merge this run into.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    apply_overrides(cfg, args)
    run_dir = run_experiment(cfg)
    print(f"Run saved to: {run_dir}")
    print(f"Dashboard: {run_dir.parent / 'visualization.html'}")


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
    if args.results_dir:
        cfg["run"]["results_dir"] = args.results_dir
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
    if args.answer_with_llm:
        cfg["run"]["answer_with_llm"] = True
    if args.retrieval_only:
        cfg["run"]["answer_with_llm"] = False
    if args.record_reasoning_trajectory:
        cfg["pageindex"]["record_reasoning_trajectory"] = True
        cfg["pageindex_official"]["record_reasoning_trajectory"] = True
        cfg["rlm"]["record_reasoning_trajectory"] = True
        cfg["rlm_recall_plus"]["record_reasoning_trajectory"] = True
    if args.no_record_reasoning_trajectory:
        cfg["pageindex"]["record_reasoning_trajectory"] = False
        cfg["pageindex_official"]["record_reasoning_trajectory"] = False
        cfg["rlm"]["record_reasoning_trajectory"] = False
        cfg["rlm_recall_plus"]["record_reasoning_trajectory"] = False
    if args.force_reindex:
        cfg["vector_rag"]["force_reindex"] = True
        cfg["pageindex"]["force_reindex"] = True
        cfg["pageindex_official"]["force_reindex"] = True
    if args.run_id:
        cfg["run"]["run_id"] = args.run_id
    if args.merge_into_run:
        merge_target = Path(args.merge_into_run)
        if str(merge_target.parent) != ".":
            cfg["run"]["results_dir"] = str(merge_target.parent)
        cfg["run"]["run_id"] = merge_target.name
        cfg["run"]["merge_results"] = True
    if args.merge_results:
        cfg["run"]["merge_results"] = True


if __name__ == "__main__":
    main()
