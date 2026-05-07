from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer import answer_question
from .config import deep_update, resolve_path
from .data import LegalBenchRAGLoader
from .llm import AnthropicLLM
from .metrics import score_retrieval
from .official_pageindex import OfficialPageIndexRAG
from .pageindex_rag import PageIndexRAG
from .rlm_rag import RLMRAG
from .types import Example, RetrievalOutput, Usage
from .vector_rag import VectorRAG
from .visualization import generate_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(cfg: dict[str, Any]) -> Path:
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    answer_with_llm = bool(run_cfg.get("answer_with_llm", False))
    started_at = datetime.now(timezone.utc)
    run_id = run_cfg.get("run_id") or started_at.strftime("%Y%m%dT%H%M%SZ")
    results_dir = resolve_path(PROJECT_ROOT, run_cfg.get("results_dir", "results"))
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    merge_results = bool(run_cfg.get("merge_results", False))
    existing_run_record = None
    existing_rows: list[dict[str, Any]] = []
    if merge_results and (run_dir / "run.json").exists():
        existing_run_record = _read_json(run_dir / "run.json")
        existing_rows = _read_results_jsonl(run_dir / "results.jsonl")

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
    unknown_methods = sorted(
        set(methods) - {"vector", "pageindex", "pageindex_official", "rlm"}
    )
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {unknown_methods}")
    llm = None
    if answer_with_llm or "pageindex" in methods or "pageindex_official" in methods:
        llm = AnthropicLLM(cfg.get("llm", {}))

    systems = {}
    setup_usage_by_method: dict[str, Usage] = {}
    if "vector" in methods:
        for method_name, vector_cfg in vector_variant_configs(cfg):
            vector = VectorRAG(
                vector_cfg,
                cache_dir=resolve_path(
                    PROJECT_ROOT, vector_cfg.get("cache_dir", ".cache/vector")
                ),
            )
            print(
                f"Building {method_name} index for {len(documents)} documents..."
            )
            vector.build(documents)
            systems[method_name] = vector
            setup_usage_by_method[method_name] = Usage()
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
    if "pageindex_official" in methods:
        official_cfg = cfg.get("pageindex_official", {})
        official_pageindex = OfficialPageIndexRAG(
            official_cfg,
            llm,
            cache_dir=resolve_path(
                PROJECT_ROOT,
                official_cfg.get("cache_dir", ".cache/pageindex_official"),
            ),
        )
        print(
            "Building official PageIndex trees for "
            f"{len(documents)} documents..."
        )
        official_pageindex.build(documents)
        systems["pageindex_official"] = official_pageindex
        setup_usage_by_method["pageindex_official"] = official_pageindex.setup_usage
    if "rlm" in methods:
        rlm_cfg = cfg.get("rlm", {})
        rlm = RLMRAG(rlm_cfg, llm_cfg=cfg.get("llm", {}))
        print(f"Preparing RLM retriever for {len(documents)} documents...")
        rlm.build(documents)
        systems["rlm"] = rlm
        setup_usage_by_method["rlm"] = Usage()

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
                answer_with_llm=answer_with_llm,
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
                            "document_precision",
                            "document_recall",
                            "document_f1",
                            "retrieved_chars",
                            "overlap_chars",
                            "retrieved_document_count",
                            "matched_document_count",
                            "retrieved_span_count",
                            "wall_clock_seconds",
                            "estimated_cost_usd",
                            "input_tokens",
                            "output_tokens",
                            "rlm_turn_count",
                            "rlm_lm_call_count",
                            "error",
                        )
                    },
                }
            )
        run_examples.append(example_record)

    toc_trees = []
    for method_name, system in systems.items():
        if hasattr(system, "toc_trees"):
            for item in system.toc_trees():
                toc_trees.append({"method": method_name, **item})

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
    if existing_run_record is not None:
        run_record, rows = _merge_run_records(
            existing_run_record=existing_run_record,
            current_run_record=run_record,
            existing_rows=existing_rows,
            current_rows=rows,
        )
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_summary_csv(run_dir / "summary.csv", rows)
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
    answer_generated = False
    retrieval = RetrievalOutput(spans=[])
    error = ""
    try:
        retrieval = system.query(example.query)
        usage.add(retrieval.usage)
        if answer_with_llm:
            if llm is None:
                raise RuntimeError("answer_with_llm requires a configured LLM")
            response = answer_question(
                llm,
                query=example.query,
                retrieved_spans=retrieval.spans,
                max_context_chars=max_answer_context_chars,
            )
            answer = response.text
            answer_generated = True
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
        "evaluation_target": "retrieval",
        "answer_generated": answer_generated,
        "answer": answer,
        "retrieved_spans": [span.to_dict() for span in retrieval.spans],
        "retrieval_metadata": retrieval.metadata,
        "reasoning_trajectory": retrieval.metadata.get("reasoning_trajectory"),
        "wall_clock_seconds": wall_clock,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "rlm_turn_count": retrieval.metadata.get("rlm_turn_count"),
        "rlm_lm_call_count": retrieval.metadata.get("rlm_lm_call_count"),
        "error": error,
    }


def _aggregate(
    rows: list[dict[str, Any]], setup_usage_by_method: dict[str, Any]
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
                "mean_document_precision": 0.0,
                "mean_document_recall": 0.0,
                "mean_document_f1": 0.0,
                "query_cost_usd": 0.0,
                "query_answer_cost_usd": 0.0,
                "setup_cost_usd": _setup_cost_usd(
                    setup_usage_by_method.get(method)
                ),
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
        bucket["mean_document_precision"] += float(
            row.get("document_precision") or 0.0
        )
        bucket["mean_document_recall"] += float(row.get("document_recall") or 0.0)
        bucket["mean_document_f1"] += float(row.get("document_f1") or 0.0)
        bucket["query_cost_usd"] += float(row.get("estimated_cost_usd") or 0.0)
        bucket["query_answer_cost_usd"] = bucket["query_cost_usd"]
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
            "mean_document_precision",
            "mean_document_recall",
            "mean_document_f1",
            "mean_wall_clock_seconds",
        ):
            bucket[key] /= n
        bucket["total_cost_usd"] = (
            bucket["query_cost_usd"] + bucket["setup_cost_usd"]
        )
        total += bucket["total_cost_usd"]
    return {
        "n_rows": len(rows),
        "primary_metric": "f1",
        "primary_metric_family": "span",
        "total_realized_cost_usd": total,
        "by_method": by_method,
    }


def _setup_cost_usd(value: Any) -> float:
    if isinstance(value, Usage):
        return value.estimated_cost_usd
    if isinstance(value, dict):
        return float(value.get("estimated_cost_usd") or 0.0)
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_results_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _merge_run_records(
    *,
    existing_run_record: dict[str, Any],
    current_run_record: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _merge_rows(existing_rows, current_rows)
    setup_costs = deepcopy(existing_run_record.get("setup_costs") or {})
    setup_costs.update(deepcopy(current_run_record.get("setup_costs") or {}))
    merged = deepcopy(existing_run_record)
    merged["run_id"] = existing_run_record.get(
        "run_id", current_run_record.get("run_id")
    )
    merged["config"] = _merge_run_config(
        existing_run_record.get("config") or {},
        current_run_record.get("config") or {},
        rows,
    )
    merged["setup_costs"] = setup_costs
    merged["aggregates"] = _aggregate(rows, setup_costs)
    merged["examples"] = _merge_examples(
        existing_run_record.get("examples") or [],
        current_run_record.get("examples") or [],
    )
    merged["toc_trees"] = _merge_toc_trees(
        existing_run_record.get("toc_trees") or [],
        current_run_record.get("toc_trees") or [],
    )
    merged.setdefault("merged_at_utc", []).append(
        current_run_record.get("started_at_utc")
    )
    return merged, rows


def _merge_rows(
    existing_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in [*existing_rows, *current_rows]:
        key = (
            str(row.get("benchmark") or ""),
            str(row.get("example_id") or ""),
            str(row.get("method") or ""),
        )
        if key not in rows_by_key:
            order.append(key)
        rows_by_key[key] = deepcopy(row)
    return [rows_by_key[key] for key in order]


def _merge_examples(
    existing_examples: list[dict[str, Any]], current_examples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    examples_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for example in existing_examples:
        example_id = str(example.get("id") or example.get("example_id") or "")
        if example_id not in examples_by_id:
            order.append(example_id)
        examples_by_id[example_id] = deepcopy(example)
    for example in current_examples:
        example_id = str(example.get("id") or example.get("example_id") or "")
        if example_id not in examples_by_id:
            order.append(example_id)
            examples_by_id[example_id] = deepcopy(example)
            continue
        merged = examples_by_id[example_id]
        merged.setdefault("methods", {})
        merged["methods"].update(deepcopy(example.get("methods") or {}))
    return [examples_by_id[example_id] for example_id in order]


def _merge_toc_trees(
    existing_toc_trees: list[dict[str, Any]], current_toc_trees: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    current_methods = {
        str(item.get("method") or "") for item in current_toc_trees if item.get("method")
    }
    kept_existing = [
        deepcopy(item)
        for item in existing_toc_trees
        if str(item.get("method") or "") not in current_methods
    ]
    return [*kept_existing, *deepcopy(current_toc_trees)]


def _merge_run_config(
    existing_config: dict[str, Any],
    current_config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_config = deepcopy(existing_config)
    merged_config.setdefault("run", {})
    methods = sorted({str(row.get("method") or "") for row in rows if row.get("method")})
    merged_config["run"]["methods"] = methods
    merged_config["run"]["merge_results"] = True
    merged_config.setdefault("merged_configs", []).append(deepcopy(current_config))
    return merged_config


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "example_id",
        "benchmark",
        "method",
        "document_precision",
        "document_recall",
        "document_f1",
        "precision",
        "recall",
        "f1",
        "retrieved_chars",
        "overlap_chars",
        "retrieved_document_count",
        "matched_document_count",
        "retrieved_span_count",
        "wall_clock_seconds",
        "estimated_cost_usd",
        "input_tokens",
        "output_tokens",
        "rlm_turn_count",
        "rlm_lm_call_count",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vector_variant_configs(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    vector_cfg = cfg.get("vector_rag", {})
    explicit_variants = vector_cfg.get("variants") or []
    if explicit_variants:
        base_cfg = _vector_base_config(vector_cfg)
        variants = []
        for idx, variant in enumerate(explicit_variants, start=1):
            if not isinstance(variant, dict):
                raise ValueError("vector_rag.variants entries must be mappings")
            method_name = str(variant.get("name") or f"vector_variant_{idx}")
            variant_cfg = deep_update(
                base_cfg, {key: value for key, value in variant.items() if key != "name"}
            )
            variants.append((_unique_method_name(method_name, variants), variant_cfg))
        return variants

    selected_profiles = _selected_model_profiles(vector_cfg)
    if selected_profiles and not vector_cfg.get("evaluate_combinations", False):
        base_cfg = _vector_base_config(vector_cfg)
        variants = []
        for profile in _filter_model_profiles(
            vector_cfg.get("model_profiles") or [], selected_profiles
        ):
            profile_name = _slug(str(profile.get("name") or _model_profile_name(profile)))
            profile_cfg = {key: value for key, value in profile.items() if key != "name"}
            variant_cfg = deep_update(base_cfg, profile_cfg)
            variants.append(
                (_unique_method_name(f"vector_{profile_name}", variants), variant_cfg)
            )
        return variants

    if not vector_cfg.get("evaluate_combinations", False):
        return [("vector", vector_cfg)]

    base_cfg = _vector_base_config(vector_cfg)
    chunk_strategies = list(
        vector_cfg.get("chunk_strategies")
        or [vector_cfg.get("chunk_strategy", "hierarchical")]
    )
    search_strategies = list(
        vector_cfg.get("search_strategies")
        or [vector_cfg.get("search_strategy", "vector")]
    )
    model_profiles = vector_cfg.get("model_profiles") or [
        {
            "name": _model_profile_name(vector_cfg),
            "embedding_provider": vector_cfg.get(
                "embedding_provider", "sentence_transformers"
            ),
            "embedding_model": vector_cfg.get("embedding_model"),
            "query_instruction": vector_cfg.get("query_instruction", ""),
            "reranker": vector_cfg.get("reranker", {}),
        }
    ]
    if selected_profiles:
        model_profiles = _filter_model_profiles(model_profiles, selected_profiles)

    variants: list[tuple[str, dict[str, Any]]] = []
    for profile in model_profiles:
        if not isinstance(profile, dict):
            raise ValueError("vector_rag.model_profiles entries must be mappings")
        profile_name = _slug(str(profile.get("name") or _model_profile_name(profile)))
        profile_cfg = {key: value for key, value in profile.items() if key != "name"}
        for chunk_strategy in chunk_strategies:
            for search_strategy in search_strategies:
                variant_cfg = deep_update(base_cfg, profile_cfg)
                variant_cfg["chunk_strategy"] = str(chunk_strategy)
                variant_cfg["search_strategy"] = str(search_strategy)
                method_name = (
                    f"vector_{profile_name}_"
                    f"{_slug(str(chunk_strategy))}_{_slug(str(search_strategy))}"
                )
                variants.append(
                    (_unique_method_name(method_name, variants), variant_cfg)
                )
    return variants


def _vector_base_config(vector_cfg: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "variants",
        "evaluate_combinations",
        "chunk_strategies",
        "search_strategies",
        "model_profiles",
        "selected_model_profiles",
    }
    return {key: value for key, value in vector_cfg.items() if key not in excluded}


def _selected_model_profiles(vector_cfg: dict[str, Any]) -> list[str]:
    selected = vector_cfg.get("selected_model_profiles")
    if selected is None:
        selected = vector_cfg.get("model_profile")
    if selected is None:
        return []
    if isinstance(selected, str):
        return [part.strip() for part in selected.split(",") if part.strip()]
    if isinstance(selected, list):
        return [str(part).strip() for part in selected if str(part).strip()]
    raise ValueError("vector_rag.selected_model_profiles must be a string or list")


def _filter_model_profiles(
    model_profiles: list[Any], selected_profiles: list[str]
) -> list[dict[str, Any]]:
    selected = {_slug(name) for name in selected_profiles}
    matched: list[dict[str, Any]] = []
    available: list[str] = []
    for profile in model_profiles:
        if not isinstance(profile, dict):
            raise ValueError("vector_rag.model_profiles entries must be mappings")
        name = str(profile.get("name") or _model_profile_name(profile))
        available.append(name)
        if _slug(name) in selected:
            matched.append(profile)
    if not matched:
        raise ValueError(
            "No vector_rag.model_profiles matched selected_model_profiles "
            f"{selected_profiles}. Available profiles: {available}"
        )
    return matched


def _model_profile_name(cfg: dict[str, Any]) -> str:
    provider = str(cfg.get("embedding_provider", "sentence_transformers"))
    model = str(cfg.get("embedding_model", provider))
    if "bge" in model.lower():
        return "bge"
    if "voyage" in model.lower() or provider.lower().startswith("voyage"):
        return "voyage"
    return model.rsplit("/", 1)[-1]


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "variant"


def _unique_method_name(
    method_name: str, existing: list[tuple[str, dict[str, Any]]]
) -> str:
    existing_names = {name for name, _ in existing}
    if method_name not in existing_names:
        return method_name
    suffix = 2
    while f"{method_name}_{suffix}" in existing_names:
        suffix += 1
    return f"{method_name}_{suffix}"
