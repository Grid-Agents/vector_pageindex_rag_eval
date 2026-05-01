from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer import answer_question
from .config import resolve_path
from .data import LegalBenchRAGLoader
from .llm import AnthropicLLM
from .metrics import score_retrieval
from .pageindex_rag import PageIndexRAG
from .types import Example, RetrievalOutput, Usage
from .vector_rag import VectorRAG
from .visualization import generate_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(cfg: dict[str, Any]) -> Path:
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    started_at = datetime.now(timezone.utc)
    run_id = run_cfg.get("run_id") or started_at.strftime("%Y%m%dT%H%M%SZ")
    results_dir = resolve_path(PROJECT_ROOT, run_cfg.get("results_dir", "results"))
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    loader = LegalBenchRAGLoader(
        resolve_path(PROJECT_ROOT, data_cfg.get("data_dir", "data"))
    )
    examples = loader.load_examples(
        data_cfg.get("benchmarks"),
        n=run_cfg.get("n"),
        seed=int(run_cfg.get("seed", 42)),
    )
    documents = loader.load_documents(
        examples,
        corpus_scope=data_cfg.get("corpus_scope", "sampled"),
    )

    methods = [name.lower() for name in run_cfg.get("methods", ["vector", "pageindex"])]
    unknown_methods = sorted(set(methods) - {"vector", "pageindex"})
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}")
    llm = None
    if run_cfg.get("answer_with_llm", True) or "pageindex" in methods:
        llm = AnthropicLLM(cfg.get("llm", {}))

    systems = {}
    setup_usage_by_method: dict[str, Usage] = {}
    if "vector" in methods:
        vector = VectorRAG(cfg.get("vector_rag", {}))
        print(f"Building vector index for {len(documents)} documents...")
        vector.build(documents)
        systems["vector"] = vector
        setup_usage_by_method["vector"] = Usage()
    if "pageindex" in methods:
        pageindex_cfg = cfg.get("pageindex", {})
        pageindex = PageIndexRAG(
            pageindex_cfg,
            llm,
            cache_dir=resolve_path(
                PROJECT_ROOT, pageindex_cfg.get("cache_dir", ".cache/pageindex")
            ),
        )
        print(f"Building PageIndex ToC trees for {len(documents)} documents...")
        pageindex.build(documents)
        systems["pageindex"] = pageindex
        setup_usage_by_method["pageindex"] = pageindex.setup_usage

    run_examples = []
    rows = []
    for idx, example in enumerate(examples, start=1):
        print(f"[{idx}/{len(examples)}] {example.example_id}: {example.query[:90]}")
        example_record = example.to_dict()
        example_record["methods"] = {}
        for method_name, system in systems.items():
            method_record = _run_method(
                method_name=method_name,
                system=system,
                example=example,
                llm=llm,
                answer_with_llm=bool(run_cfg.get("answer_with_llm", True)),
                max_answer_context_chars=int(
                    run_cfg.get("max_answer_context_chars", 18000)
                ),
            )
            example_record["methods"][method_name] = method_record
            rows.append(
                {
                    "example_id": example.example_id,
                    "benchmark": example.benchmark,
                    "method": method_name,
                    **{
                        key: method_record.get(key)
                        for key in (
                            "precision",
                            "recall",
                            "f1",
                            "retrieved_chars",
                            "overlap_chars",
                            "wall_clock_seconds",
                            "estimated_cost_usd",
                            "input_tokens",
                            "output_tokens",
                            "error",
                        )
                    },
                }
            )
        run_examples.append(example_record)

    toc_trees = []
    if "pageindex" in systems:
        toc_trees = systems["pageindex"].toc_trees()

    run_record = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "config": cfg,
        "setup_costs": {
            method: usage.to_dict() for method, usage in setup_usage_by_method.items()
        },
        "aggregates": _aggregate(rows, setup_usage_by_method),
        "examples": run_examples,
        "toc_trees": toc_trees,
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_summary_csv(run_dir / "summary.csv", rows)
    generate_dashboard(results_dir, results_dir / "visulization.html")
    generate_dashboard(results_dir, results_dir / "visualization.html")
    return run_dir


def _run_method(
    *,
    method_name: str,
    system: Any,
    example: Example,
    llm: Any,
    answer_with_llm: bool,
    max_answer_context_chars: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    usage = Usage()
    answer = ""
    retrieval = RetrievalOutput(spans=[])
    error = ""
    try:
        retrieval = system.query(example.query)
        usage.add(retrieval.usage)
        if answer_with_llm:
            response = answer_question(
                llm,
                query=example.query,
                retrieved_spans=retrieval.spans,
                max_context_chars=max_answer_context_chars,
            )
            answer = response.text
            usage.add(response.usage)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    wall_clock = time.perf_counter() - started
    metrics = score_retrieval(example.gold_spans, retrieval.spans)
    if retrieval.error:
        error = f"{error}; {retrieval.error}".strip("; ")
    return {
        "method": method_name,
        **metrics,
        "answer": answer,
        "retrieved_spans": [span.to_dict() for span in retrieval.spans],
        "retrieval_metadata": retrieval.metadata,
        "wall_clock_seconds": wall_clock,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "error": error,
    }


def _aggregate(
    rows: list[dict[str, Any]], setup_usage_by_method: dict[str, Usage]
) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = row["method"]
        bucket = by_method.setdefault(
            method,
            {
                "n": 0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1": 0.0,
                "query_answer_cost_usd": 0.0,
                "setup_cost_usd": setup_usage_by_method.get(
                    method, Usage()
                ).estimated_cost_usd,
                "total_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "mean_wall_clock_seconds": 0.0,
                "errors": 0,
            },
        )
        bucket["n"] += 1
        bucket["mean_precision"] += float(row.get("precision") or 0.0)
        bucket["mean_recall"] += float(row.get("recall") or 0.0)
        bucket["mean_f1"] += float(row.get("f1") or 0.0)
        bucket["query_answer_cost_usd"] += float(row.get("estimated_cost_usd") or 0.0)
        bucket["input_tokens"] += int(row.get("input_tokens") or 0)
        bucket["output_tokens"] += int(row.get("output_tokens") or 0)
        bucket["mean_wall_clock_seconds"] += float(
            row.get("wall_clock_seconds") or 0.0
        )
        bucket["errors"] += 1 if row.get("error") else 0

    total = 0.0
    for bucket in by_method.values():
        n = max(1, bucket["n"])
        for key in (
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_wall_clock_seconds",
        ):
            bucket[key] /= n
        bucket["total_cost_usd"] = (
            bucket["query_answer_cost_usd"] + bucket["setup_cost_usd"]
        )
        total += bucket["total_cost_usd"]
    return {
        "n_rows": len(rows),
        "total_realized_cost_usd": total,
        "by_method": by_method,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "example_id",
        "benchmark",
        "method",
        "precision",
        "recall",
        "f1",
        "retrieved_chars",
        "overlap_chars",
        "wall_clock_seconds",
        "estimated_cost_usd",
        "input_tokens",
        "output_tokens",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
