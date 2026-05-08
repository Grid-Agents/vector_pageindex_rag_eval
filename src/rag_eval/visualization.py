from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPAN_TEXT_PREVIEW_CHARS = 1200
ANSWER_PREVIEW_CHARS = 2000
RAW_RESPONSE_PREVIEW_CHARS = 3000
NODE_SUMMARY_PREVIEW_CHARS = 700
REASON_PREVIEW_CHARS = 900
CATALOG_PREVIEW_CHARS = 4000


def generate_dashboard(results_dir: str | Path, output_path: str | Path) -> None:
    results_dir = Path(results_dir)
    runs = []
    for path in sorted(results_dir.glob("*/run.json"), reverse=True):
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        run["run_dir"] = str(path.parent.relative_to(results_dir.parent))
        runs.append(_public_run(run))

    html = HTML_TEMPLATE.replace("__RUNS_JSON__", json.dumps(runs, ensure_ascii=False))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "started_at_utc": run.get("started_at_utc"),
        "config": _public_config(run.get("config") or {}),
        "setup_costs": run.get("setup_costs") or {},
        "aggregates": run.get("aggregates") or {},
        "examples": [_public_example(ex) for ex in run.get("examples") or []],
        "toc_trees": [_public_toc_tree(item) for item in run.get("toc_trees") or []],
        "merged_at_utc": run.get("merged_at_utc"),
        "run_dir": run.get("run_dir"),
    }


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    public = dict(config)
    merged_configs = public.get("merged_configs")
    if isinstance(merged_configs, list):
        public["merged_configs"] = [
            _public_config(item) if isinstance(item, dict) else item
            for item in merged_configs
        ]
    return public


def _public_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example.get("id") or example.get("example_id"),
        "benchmark": example.get("benchmark"),
        "query": example.get("query"),
        "gold_spans": [
            _public_span(span, text_limit=SPAN_TEXT_PREVIEW_CHARS)
            for span in example.get("gold_spans") or []
        ],
        "tags": example.get("tags") or [],
        "methods": {
            method: _public_result(result)
            for method, result in (example.get("methods") or {}).items()
        },
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "method",
        "precision",
        "recall",
        "f1",
        "gold_chars",
        "retrieved_chars",
        "overlap_chars",
        "gold_document_count",
        "retrieved_document_count",
        "matched_document_count",
        "document_precision",
        "document_recall",
        "document_f1",
        "gold_span_count",
        "retrieved_span_count",
        "evaluation_target",
        "answer_generated",
        "wall_clock_seconds",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "estimated_cost_usd",
        "rlm_turn_count",
        "rlm_lm_call_count",
        "error",
    )
    public = {key: result.get(key) for key in scalar_keys if key in result}
    public["answer"] = _preview(result.get("answer"), ANSWER_PREVIEW_CHARS)
    public["retrieved_spans"] = [
        _public_span(span, text_limit=SPAN_TEXT_PREVIEW_CHARS)
        for span in result.get("retrieved_spans") or []
    ]
    public["retrieval_metadata"] = _public_retrieval_metadata(
        result.get("retrieval_metadata") or {}
    )
    trajectory = result.get("reasoning_trajectory")
    if trajectory:
        public["reasoning_trajectory"] = _public_reasoning_trajectory(trajectory)
    return public


def _public_span(span: dict[str, Any], *, text_limit: int) -> dict[str, Any]:
    return {
        "document_id": span.get("document_id"),
        "start_char": span.get("start_char"),
        "end_char": span.get("end_char"),
        "score": span.get("score"),
        "text": _preview(span.get("text"), text_limit),
        "metadata": _public_span_metadata(span.get("metadata") or {}),
    }


def _public_span_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "retriever",
        "rank",
        "node_id",
        "node_title",
        "chunk_title",
        "section_title",
        "reason",
        "rlm_reason",
        "unit_start",
        "unit_end",
    )
    return {
        key: _preview(value, REASON_PREVIEW_CHARS) if isinstance(value, str) else value
        for key, value in metadata.items()
        if key in keep
    }


def _public_retrieval_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "document_selections",
        "document_selection_raw",
        "selection_raw",
        "node_selection_raw_by_document",
        "rlm_backend",
        "rlm_model",
        "rlm_turn_count",
        "rlm_lm_call_count",
        "rlm_final_response",
        "rlm_usage_summary",
    )
    public = {key: metadata.get(key) for key in keep if key in metadata}
    if metadata.get("reasoning_trajectory"):
        public["reasoning_trajectory"] = _public_reasoning_trajectory(
            metadata["reasoning_trajectory"]
        )
    return _preview_nested(public)


def _public_reasoning_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    if trajectory.get("type") == "rlm":
        return _public_rlm_reasoning_trajectory(trajectory)
    public = {
        "query": trajectory.get("query"),
        "document_selection": _preview_nested(trajectory.get("document_selection") or {}),
        "document_walks": [
            _public_document_walk(walk)
            for walk in trajectory.get("document_walks") or []
        ],
        "retrieved_nodes": [
            _preview_nested(node) for node in trajectory.get("retrieved_nodes") or []
        ],
        "errors": trajectory.get("errors") or [],
    }
    return public


def _public_rlm_reasoning_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "rlm",
        "query": trajectory.get("query"),
        "turn_count": trajectory.get("turn_count"),
        "llm_call_count": trajectory.get("llm_call_count"),
        "final_response": _preview(
            trajectory.get("final_response"), RAW_RESPONSE_PREVIEW_CHARS
        ),
        "run_metadata": _preview_nested(trajectory.get("run_metadata") or {}),
        "usage_summary": _preview_nested(trajectory.get("usage_summary") or {}),
        "iterations": [
            _public_rlm_iteration(item)
            for item in trajectory.get("iterations") or []
        ],
        "retrieved_spans": [
            _preview_nested(span)
            for span in trajectory.get("retrieved_spans") or []
        ],
        "errors": trajectory.get("errors") or [],
    }


def _public_rlm_iteration(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn": item.get("turn"),
        "timestamp": item.get("timestamp"),
        "llm_output": _preview(item.get("llm_output"), RAW_RESPONSE_PREVIEW_CHARS),
        "final_answer": _preview(item.get("final_answer"), RAW_RESPONSE_PREVIEW_CHARS),
        "iteration_time": item.get("iteration_time"),
        "code_blocks": [
            {
                "block": block.get("block"),
                "code": _preview(block.get("code"), RAW_RESPONSE_PREVIEW_CHARS),
                "stdout": _preview(block.get("stdout"), RAW_RESPONSE_PREVIEW_CHARS),
                "stderr": _preview(block.get("stderr"), RAW_RESPONSE_PREVIEW_CHARS),
                "final_answer": _preview(
                    block.get("final_answer"), RAW_RESPONSE_PREVIEW_CHARS
                ),
                "rlm_call_count": block.get("rlm_call_count"),
                "rlm_calls": [
                    _preview_nested(call) for call in block.get("rlm_calls") or []
                ],
            }
            for block in item.get("code_blocks") or []
        ],
    }


def _public_document_walk(walk: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": walk.get("document_id"),
        "source": walk.get("source"),
        "document_reason": _preview(walk.get("document_reason"), REASON_PREVIEW_CHARS),
        "error": walk.get("error"),
        "final_selections": _preview_nested(walk.get("final_selections") or []),
        "steps": [_public_trace_step(step) for step in walk.get("steps") or []],
    }


def _public_trace_step(step: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "step",
        "current_node",
        "node_id",
        "unit_start",
        "unit_end",
        "selection_source",
        "fallback_reason",
        "candidate_child_count",
        "child_count",
        "prompt_child_list_truncated",
        "accepted_selections",
        "llm_selections",
    )
    public = {key: step.get(key) for key in keep if key in step}
    public["candidate_children"] = [
        _public_toc_node(child) for child in step.get("candidate_children") or []
    ]
    raw = step.get("raw_response") or step.get("selection_raw")
    if raw:
        public["raw_response"] = _preview_nested(raw)
    return public


def _public_toc_tree(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": item.get("method"),
        "document_id": item.get("document_id"),
        "tree": _public_toc_node(item.get("tree") or {}),
    }


def _public_toc_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "title": _preview(node.get("title"), NODE_SUMMARY_PREVIEW_CHARS),
        "summary": _preview(node.get("summary"), NODE_SUMMARY_PREVIEW_CHARS),
        "start_char": node.get("start_char"),
        "end_char": node.get("end_char"),
        "unit_start": node.get("unit_start"),
        "unit_end": node.get("unit_end"),
        "token_count": node.get("token_count"),
        "node_kind": node.get("node_kind"),
        "section_title": _preview(node.get("section_title"), NODE_SUMMARY_PREVIEW_CHARS),
        "children": [_public_toc_node(child) for child in node.get("children") or []],
    }


def _preview_nested(value: Any) -> Any:
    if isinstance(value, str):
        return _preview(value, RAW_RESPONSE_PREVIEW_CHARS)
    if isinstance(value, list):
        return [_preview_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _preview_nested(item) for key, item in value.items()}
    return value


def _preview(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n...[truncated {len(text) - limit} chars]"


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LegalBench-RAG Retrieval Evaluation</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f5; color: #202124; }
button, select { font: inherit; }
.topbar { display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: #fff; border-bottom: 1px solid #d8d8d2; position: sticky; top: 0; z-index: 2; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 14px; margin: 0 0 8px; text-transform: uppercase; color: #5f6368; letter-spacing: .04em; }
h3 { font-size: 15px; margin: 0 0 8px; }
select { padding: 6px 8px; border: 1px solid #c7c7c0; background: #fff; border-radius: 6px; }
.tabs { display: flex; gap: 6px; margin-left: auto; }
.tab { border: 1px solid #c7c7c0; background: #fff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }
.tab.active { background: #243b53; color: #fff; border-color: #243b53; }
.page { padding: 16px 18px 24px; }
.meta { color: #666; font-size: 12px; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin-bottom: 14px; }
.card, .panel { background: #fff; border: 1px solid #d8d8d2; border-radius: 8px; padding: 12px; }
.card .label { font-size: 11px; color: #666; text-transform: uppercase; }
.card .value { font-size: 22px; font-weight: 650; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }
th { color: #555; font-weight: 600; background: #fafafa; }
.layout { display: grid; grid-template-columns: 330px 1fr; gap: 14px; }
.examples { max-height: calc(100vh - 154px); overflow: auto; }
.ex { padding: 9px 10px; border-bottom: 1px solid #eee; cursor: pointer; }
.ex:hover, .ex.active { background: #eef5fb; }
.ex-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #666; }
.ex-q { font-size: 12px; margin-top: 3px; }
.badges { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; background: #eee; }
.good { background: #dff3df; color: #185c1b; }
.mid { background: #fff0c2; color: #6f5300; }
.bad { background: #ffd9d9; color: #8b1a1a; }
.method { border-left: 4px solid #89939e; }
.method.vector { border-left-color: #2f80ed; }
.method.pageindex { border-left-color: #8a5cf6; }
.method.pageindex_official { border-left-color: #00a884; }
.method.rlm { border-left-color: #d66f00; }
.method.rlm_recall_plus { border-left-color: #b44100; }
.query { padding: 10px; background: #fafafa; border: 1px solid #eee; border-radius: 6px; white-space: pre-wrap; }
.answer { margin: 8px 0; padding: 9px; background: #fafafa; border-left: 3px solid #c7c7c0; white-space: pre-wrap; }
.span { margin-top: 8px; border: 1px solid #deded8; border-radius: 6px; overflow: hidden; }
.span-meta { padding: 6px 8px; background: #fafafa; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #555; }
.span-text { padding: 8px; white-space: pre-wrap; max-height: 230px; overflow: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.err { color: #8b1a1a; background: #fff2f2; padding: 6px 8px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.toc-layout { display: grid; grid-template-columns: 360px 1fr; gap: 14px; }
.toc-list { max-height: calc(100vh - 160px); overflow: auto; }
.toc-doc { padding: 8px 10px; border-bottom: 1px solid #eee; cursor: pointer; word-break: break-word; font-size: 12px; }
.toc-doc:hover, .toc-doc.active { background: #f0ecff; }
.tree ul { list-style: none; margin: 0 0 0 18px; padding: 0; border-left: 1px solid #ddd; }
.tree li { margin: 0; padding: 6px 0 6px 10px; }
.node-title { font-weight: 600; }
.node-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #666; font-size: 11px; }
.node-summary { color: #555; font-size: 12px; margin-top: 2px; }
.reasoning-layout { display: grid; grid-template-columns: 360px 1fr; gap: 14px; }
.reasoning-list { max-height: calc(100vh - 160px); overflow: auto; }
.reasoning-item { padding: 9px 10px; border-bottom: 1px solid #eee; cursor: pointer; }
.reasoning-item:hover, .reasoning-item.active { background: #f0ecff; }
.trace-section { margin-bottom: 12px; }
.trace-step { border: 1px solid #deded8; border-radius: 6px; padding: 10px; margin-top: 8px; }
.trace-title { font-weight: 650; margin-bottom: 5px; }
.candidate-list { display: grid; gap: 6px; margin-top: 8px; }
.candidate { border: 1px solid #e5e5df; border-radius: 6px; padding: 7px 8px; background: #fafafa; }
.candidate.selected { border-color: #8a5cf6; background: #f5f1ff; }
.reason { color: #444; font-size: 12px; margin-top: 3px; }
.rlm-trace { margin: 10px 0 12px; border: 1px solid #e5e5df; border-radius: 6px; padding: 8px; background: #fffaf2; }
.rlm-trace > summary { font-weight: 650; color: #6d3900; }
details { margin-top: 8px; }
summary { cursor: pointer; color: #555; font-size: 12px; }
pre { white-space: pre-wrap; word-break: break-word; background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 8px; max-height: 280px; overflow: auto; font-size: 12px; }
.hidden { display: none; }
.empty { color: #888; font-style: italic; padding: 12px; }
@media (max-width: 900px) {
  .cards, .grid2, .layout, .toc-layout, .reasoning-layout { grid-template-columns: 1fr; }
  .tabs { margin-left: 0; }
  .topbar { flex-wrap: wrap; }
}
</style>
</head>
<body>
<div class="topbar">
  <h1>LegalBench-RAG Retrieval Evaluation</h1>
  <label>Run <select id="run-select"></select></label>
  <span id="run-meta" class="meta"></span>
  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="examples">Examples</button>
    <button class="tab" data-tab="reasoning">PageIndex Reasoning</button>
    <button class="tab" data-tab="toc">PageIndex ToC</button>
  </div>
</div>
<main class="page">
  <section id="overview"></section>
  <section id="examples" class="hidden"></section>
  <section id="reasoning" class="hidden"></section>
  <section id="toc" class="hidden"></section>
</main>
<script>
const RUNS = __RUNS_JSON__;
const $ = sel => document.querySelector(sel);
const esc = value => (value == null ? "" : String(value))
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const fmt = (value, digits = 3) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
const money = value => "$" + fmt(value || 0, 4);
const f1Class = value => value >= .8 ? "good" : value >= .4 ? "mid" : "bad";

let currentRun = null;
let currentExample = 0;
let currentReasoningExample = 0;
let currentTocDoc = 0;

function mergeRanges(ranges) {
  const sorted = (ranges || [])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((a, b) => a[0] - b[0]);
  if (!sorted.length) return [];
  const merged = [sorted[0].slice()];
  for (const [start, end] of sorted.slice(1)) {
    const last = merged[merged.length - 1];
    if (start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

function rangeLen(ranges) {
  return (ranges || []).reduce((total, [start, end]) => total + (end - start), 0);
}

function overlapLen(a, b) {
  let i = 0;
  let j = 0;
  let total = 0;
  while (i < a.length && j < b.length) {
    const start = Math.max(a[i][0], b[j][0]);
    const end = Math.min(a[i][1], b[j][1]);
    if (end > start) total += end - start;
    if (a[i][1] < b[j][1]) i += 1;
    else j += 1;
  }
  return total;
}

function computeDocumentMetrics(goldSpans, retrievedSpans) {
  const goldByDoc = new Map();
  const predByDoc = new Map();
  for (const span of goldSpans || []) {
    if (!goldByDoc.has(span.document_id)) goldByDoc.set(span.document_id, []);
    goldByDoc.get(span.document_id).push([span.start_char, span.end_char]);
  }
  for (const span of retrievedSpans || []) {
    if (!predByDoc.has(span.document_id)) predByDoc.set(span.document_id, []);
    predByDoc.get(span.document_id).push([span.start_char, span.end_char]);
  }

  const allDocIds = new Set([...goldByDoc.keys(), ...predByDoc.keys()]);
  let goldTotal = 0;
  let predTotal = 0;
  let overlapTotal = 0;
  for (const docId of allDocIds) {
    const goldRanges = mergeRanges(goldByDoc.get(docId) || []);
    const predRanges = mergeRanges(predByDoc.get(docId) || []);
    goldTotal += rangeLen(goldRanges);
    predTotal += rangeLen(predRanges);
    overlapTotal += overlapLen(goldRanges, predRanges);
  }

  const goldDocIds = new Set(goldByDoc.keys());
  const predDocIds = new Set(predByDoc.keys());
  const matchedDocumentCount = [...goldDocIds].filter(docId => predDocIds.has(docId)).length;
  const precision = predTotal ? overlapTotal / predTotal : 0;
  const recall = goldTotal ? overlapTotal / goldTotal : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const documentPrecision = predDocIds.size ? matchedDocumentCount / predDocIds.size : 0;
  const documentRecall = goldDocIds.size ? matchedDocumentCount / goldDocIds.size : 0;
  const documentF1 = documentPrecision + documentRecall > 0
    ? (2 * documentPrecision * documentRecall) / (documentPrecision + documentRecall)
    : 0;

  return {
    precision,
    recall,
    f1,
    gold_chars: goldTotal,
    retrieved_chars: predTotal,
    overlap_chars: overlapTotal,
    gold_document_count: goldDocIds.size,
    retrieved_document_count: predDocIds.size,
    matched_document_count: matchedDocumentCount,
    document_precision: documentPrecision,
    document_recall: documentRecall,
    document_f1: documentF1,
    gold_span_count: (goldSpans || []).length,
    retrieved_span_count: (retrievedSpans || []).length,
  };
}

function hydrateRun(run) {
  const examples = run.examples || [];
  const rows = [];
  for (const ex of examples) {
    const goldSpans = ex.gold_spans || [];
    for (const [method, result] of Object.entries(ex.methods || {})) {
      if (result.document_f1 == null || result.document_precision == null || result.document_recall == null) {
        Object.assign(result, computeDocumentMetrics(goldSpans, result.retrieved_spans || []));
      }
      rows.push({
        method,
        precision: Number(result.precision) || 0,
        recall: Number(result.recall) || 0,
        f1: Number(result.f1) || 0,
        document_precision: Number(result.document_precision) || 0,
        document_recall: Number(result.document_recall) || 0,
        document_f1: Number(result.document_f1) || 0,
        estimated_cost_usd: Number(result.estimated_cost_usd) || 0,
        wall_clock_seconds: Number(result.wall_clock_seconds) || 0,
        input_tokens: Number(result.input_tokens) || 0,
        output_tokens: Number(result.output_tokens) || 0,
        rlm_turn_count: result.rlm_turn_count == null ? null : Number(result.rlm_turn_count),
        rlm_lm_call_count: result.rlm_lm_call_count == null ? null : Number(result.rlm_lm_call_count),
        error: result.error || "",
      });
    }
  }

  const agg = run.aggregates || {};
  const byMethod = {};
  for (const row of rows) {
    const bucket = byMethod[row.method] ||= {
      n: 0,
      mean_precision: 0,
      mean_recall: 0,
      mean_f1: 0,
      mean_document_precision: 0,
      mean_document_recall: 0,
      mean_document_f1: 0,
      query_cost_usd: 0,
      query_answer_cost_usd: 0,
      setup_cost_usd: Number((((run.setup_costs || {})[row.method] || {}).estimated_cost_usd)) || 0,
      total_cost_usd: 0,
      input_tokens: 0,
      output_tokens: 0,
      mean_wall_clock_seconds: 0,
      errors: 0,
    };
    bucket.n += 1;
    bucket.mean_precision += row.precision;
    bucket.mean_recall += row.recall;
    bucket.mean_f1 += row.f1;
    bucket.mean_document_precision += row.document_precision;
    bucket.mean_document_recall += row.document_recall;
    bucket.mean_document_f1 += row.document_f1;
    bucket.query_cost_usd += row.estimated_cost_usd;
    bucket.query_answer_cost_usd = bucket.query_cost_usd;
    bucket.input_tokens += row.input_tokens;
    bucket.output_tokens += row.output_tokens;
    bucket.mean_wall_clock_seconds += row.wall_clock_seconds;
    bucket.errors += row.error ? 1 : 0;
  }

  let totalRealizedCostUsd = 0;
  for (const bucket of Object.values(byMethod)) {
    const n = Math.max(1, bucket.n);
    bucket.mean_precision /= n;
    bucket.mean_recall /= n;
    bucket.mean_f1 /= n;
    bucket.mean_document_precision /= n;
    bucket.mean_document_recall /= n;
    bucket.mean_document_f1 /= n;
    bucket.mean_wall_clock_seconds /= n;
    bucket.total_cost_usd = bucket.query_cost_usd + bucket.setup_cost_usd;
    totalRealizedCostUsd += bucket.total_cost_usd;
  }

  run.aggregates = {
    ...agg,
    n_rows: rows.length,
    primary_metric: "f1",
    primary_metric_family: "span",
    total_realized_cost_usd: totalRealizedCostUsd,
    by_method: {
      ...(agg.by_method || {}),
      ...byMethod,
    },
  };
  return run;
}

function init() {
  for (let i = 0; i < RUNS.length; i += 1) RUNS[i] = hydrateRun(RUNS[i]);
  const select = $("#run-select");
  select.innerHTML = RUNS.map((run, idx) => `<option value="${idx}">${esc(run.run_id)}</option>`).join("");
  select.onchange = () => renderRun(Number(select.value));
  document.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      ["overview", "examples", "reasoning", "toc"].forEach(id => $("#" + id).classList.add("hidden"));
      $("#" + btn.dataset.tab).classList.remove("hidden");
    };
  });
  if (RUNS.length) renderRun(0);
  else $("#overview").innerHTML = `<div class="panel empty">No runs found. Run an experiment first.</div>`;
}

function renderRun(idx) {
  currentRun = RUNS[idx];
  currentExample = 0;
  currentReasoningExample = 0;
  currentTocDoc = 0;
  const cfg = currentRun.config || {};
  const runCfg = cfg.run || {};
  $("#run-meta").innerHTML = `${esc(currentRun.run_dir || "")} · n=${esc(runCfg.n || "?")} · methods=${esc((runCfg.methods || []).join(", "))}`;
  renderOverview();
  renderExamples();
  renderReasoning();
  renderToc();
}

function renderOverview() {
  const agg = currentRun.aggregates || {};
  const reasoningCount = reasoningTraceCount();
  const rows = Object.entries(agg.by_method || {}).map(([method, s]) => `
    <tr>
      <td><span class="badge">${esc(method)}</span></td>
      <td>${esc(s.n)}</td>
      <td><span class="badge ${f1Class(s.mean_f1)}">${fmt(s.mean_f1)}</span></td>
      <td>${fmt(s.mean_precision)}</td>
      <td>${fmt(s.mean_recall)}</td>
      <td>${fmt(s.mean_document_f1)}</td>
      <td>${fmt(s.mean_document_precision)}</td>
      <td>${fmt(s.mean_document_recall)}</td>
      <td>${money(s.total_cost_usd)}</td>
      <td>${money(s.setup_cost_usd)}</td>
      <td>${fmt(s.mean_wall_clock_seconds, 1)}s</td>
      <td>${esc(s.errors || 0)}</td>
    </tr>`).join("");
  $("#overview").innerHTML = `
    <div class="cards">
      <div class="card"><div class="label">Rows</div><div class="value">${esc(agg.n_rows || 0)}</div></div>
      <div class="card"><div class="label">Total Cost</div><div class="value">${money(agg.total_realized_cost_usd)}</div></div>
      <div class="card"><div class="label">Examples</div><div class="value">${esc((currentRun.examples || []).length)}</div></div>
      <div class="card"><div class="label">Reasoning Traces</div><div class="value">${esc(reasoningCount)}</div></div>
    </div>
    <div class="panel">
      <h2>Method Aggregates</h2>
      <p class="meta">Primary scores are span/token-level overlap metrics. Document-level retrieval metrics from matched <code>document_id</code> values are retained as secondary diagnostics.</p>
      <table><thead><tr><th>Method</th><th>n</th><th>Span F1</th><th>Span P</th><th>Span R</th><th>Doc F1</th><th>Doc P</th><th>Doc R</th><th>Total cost</th><th>Setup cost</th><th>Latency</th><th>Errors</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="12" class="empty">No metrics</td></tr>`}</tbody></table>
    </div>`;
}

function renderExamples() {
  const examples = currentRun.examples || [];
  const list = examples.map((ex, idx) => {
    const badges = Object.entries(ex.methods || {}).map(([m, r]) =>
      `<span class="badge ${f1Class(r.f1)}">${esc(m)} Span F1 ${fmt(r.f1, 2)}</span>`).join("");
    return `<div class="ex ${idx === currentExample ? "active" : ""}" data-idx="${idx}">
      <div class="ex-id">${esc(ex.benchmark)} / ${esc(ex.id)}</div>
      <div class="ex-q">${esc((ex.query || "").slice(0, 160))}${(ex.query || "").length > 160 ? "..." : ""}</div>
      <div class="badges">${badges}</div>
    </div>`;
  }).join("");
  $("#examples").innerHTML = `
    <div class="layout">
      <div class="panel examples">${list || `<div class="empty">No examples</div>`}</div>
      <div id="example-detail"></div>
    </div>`;
  document.querySelectorAll(".ex").forEach(el => {
    el.onclick = () => {
      currentExample = Number(el.dataset.idx);
      renderExamples();
    };
  });
  if (examples.length) renderExampleDetail(examples[currentExample]);
}

function renderExampleDetail(ex) {
  const gold = (ex.gold_spans || []).map(spanBlock).join("");
  const methods = Object.entries(ex.methods || {}).map(([method, result]) => methodPanel(method, result)).join("");
  $("#example-detail").innerHTML = `
    <div class="panel">
      <h2>Question</h2>
      <div class="query">${esc(ex.query)}</div>
    </div>
    <div class="panel">
      <h2>Gold Expected Retrieved Spans</h2>
      ${gold || `<div class="empty">No gold spans</div>`}
    </div>
    <div class="grid2">${methods}</div>`;
}

function methodPanel(method, result) {
  const spans = (result.retrieved_spans || []).map(spanBlock).join("");
  const answerBlock = result.answer
    ? `<h2>Generated Answer Diagnostic</h2><div class="answer">${esc(result.answer)}</div>`
    : "";
  const rlmTraceBlock = renderRlmTraceBlock(result);
  return `<div class="panel method ${esc(method)}">
    <h3><span class="badge">${esc(method)}</span> <span class="badge ${f1Class(result.f1)}">Span F1 ${fmt(result.f1)}</span></h3>
    <div class="badges">
      <span class="badge">Span P ${fmt(result.precision)}</span>
      <span class="badge">Span R ${fmt(result.recall)}</span>
      <span class="badge">overlap ${esc(result.overlap_chars || 0)}/${esc(result.gold_chars || 0)}</span>
      <span class="badge">retrieved ${esc(result.retrieved_chars || 0)} chars</span>
      <span class="badge">Doc F1 ${fmt(result.document_f1)}</span>
      <span class="badge">Doc P ${fmt(result.document_precision)}</span>
      <span class="badge">Doc R ${fmt(result.document_recall)}</span>
      <span class="badge">docs ${esc(result.matched_document_count || 0)}/${esc(result.retrieved_document_count || 0)}</span>
      <span class="badge">spans ${esc(result.retrieved_span_count || (result.retrieved_spans || []).length)}</span>
      <span class="badge">${money(result.estimated_cost_usd)}</span>
      <span class="badge">${fmt(result.wall_clock_seconds, 1)}s</span>
      <span class="badge">in ${esc(result.input_tokens || 0)}</span>
      <span class="badge">out ${esc(result.output_tokens || 0)}</span>
      ${result.rlm_turn_count != null ? `<span class="badge">RLM turns ${esc(result.rlm_turn_count)}</span>` : ""}
      ${result.rlm_lm_call_count != null ? `<span class="badge">RLM calls ${esc(result.rlm_lm_call_count)}</span>` : ""}
    </div>
    ${result.error ? `<p class="err">${esc(result.error)}</p>` : ""}
    ${answerBlock}
    ${rlmTraceBlock}
    <h2>Retrieved Spans</h2>
    ${spans || `<div class="empty">No retrieved spans</div>`}
  </div>`;
}

function spanBlock(span) {
  const meta = span.metadata || {};
  return `<div class="span">
    <div class="span-meta">${esc(span.document_id)} [${esc(span.start_char)}:${esc(span.end_char)}] score=${fmt(span.score)}
      ${meta.node_id ? ` · node=${esc(meta.node_id)}` : ""}
      ${meta.chunk_title ? ` · ${esc(meta.chunk_title)}` : ""}
      ${meta.reason ? ` · reason=${esc(String(meta.reason).slice(0, 120))}` : ""}
    </div>
    <div class="span-text">${esc(span.text)}</div>
  </div>`;
}

function resultTrajectory(result) {
  const metadata = result.retrieval_metadata || {};
  return result.reasoning_trajectory || metadata.reasoning_trajectory || null;
}

function rlmTrajectory(result) {
  const trajectory = resultTrajectory(result);
  if (!trajectory) return null;
  if (trajectory.type === "rlm" || Array.isArray(trajectory.iterations)) return trajectory;
  return null;
}

function renderRlmTraceBlock(result) {
  const trajectory = rlmTrajectory(result);
  if (!trajectory) return "";
  const turns = trajectory.iterations || [];
  const turnCount = trajectory.turn_count != null ? trajectory.turn_count : turns.length;
  const llmCalls = trajectory.llm_call_count != null ? trajectory.llm_call_count : "";
  const retrieved = (trajectory.retrieved_spans || []).map(span => `
    <div class="candidate selected">
      <div class="node-title">${esc(span.document_id)} [${esc(span.start_char)}:${esc(span.end_char)}]</div>
      <div class="node-id">score=${fmt(span.score)}</div>
      ${span.reason ? `<div class="reason">${esc(span.reason)}</div>` : ""}
    </div>`).join("");
  const turnBlocks = turns.map(renderRlmTurn).join("");
  return `<details class="rlm-trace">
    <summary>RLM reasoning trajectory (${esc(turnCount)} turns)</summary>
    <div class="badges">
      <span class="badge">turns ${esc(turnCount)}</span>
      ${llmCalls !== "" ? `<span class="badge">LLM calls ${esc(llmCalls)}</span>` : ""}
    </div>
    ${trajectory.final_response ? jsonDetails("Final RLM response", trajectory.final_response) : ""}
    <h2>Turn Outputs</h2>
    ${turnBlocks || `<div class="empty">No RLM turns recorded.</div>`}
    <h2>RLM Retrieved Spans</h2>
    <div class="candidate-list">${retrieved || `<div class="empty">No retrieved spans recorded in trajectory.</div>`}</div>
  </details>`;
}

function renderRlmTurn(turn) {
  const codeBlocks = (turn.code_blocks || []).map(block => {
    const body = [
      block.code ? `# code\n${block.code}` : "",
      block.stdout ? `# stdout\n${block.stdout}` : "",
      block.stderr ? `# stderr\n${block.stderr}` : "",
      block.final_answer ? `# final_answer\n${block.final_answer}` : "",
    ].filter(Boolean).join("\n\n");
    return body ? jsonDetails(`REPL block ${block.block || ""}`, body) : "";
  }).join("");
  return `<div class="trace-step">
    <div class="trace-title">Turn ${esc(turn.turn || "")}</div>
    <div class="badges">
      ${turn.iteration_time != null ? `<span class="badge">${fmt(turn.iteration_time, 2)}s</span>` : ""}
      ${turn.final_answer ? `<span class="badge good">final</span>` : ""}
    </div>
    <pre>${esc(turn.llm_output || "")}</pre>
    ${turn.final_answer ? jsonDetails("Turn final answer", turn.final_answer) : ""}
    ${codeBlocks}
  </div>`;
}

function reasoningTraceCount() {
  let count = 0;
  (currentRun?.examples || []).forEach(ex => {
    Object.entries(ex.methods || {}).forEach(([_method, result]) => {
      if (rlmTrajectory(result) || pageindexTrajectory(result)) count += 1;
    });
  });
  return count;
}

function pageindexReasoningItems() {
  const items = [];
  (currentRun?.examples || []).forEach((ex, idx) => {
    Object.entries(ex.methods || {}).forEach(([method, result]) => {
      if (!method.includes("pageindex")) return;
      const trajectory = result ? pageindexTrajectory(result) : null;
      if (trajectory) items.push({ ex, idx, method, result, trajectory });
    });
  });
  return items;
}

function pageindexTrajectory(result) {
  const metadata = result.retrieval_metadata || {};
  if (result.reasoning_trajectory) return result.reasoning_trajectory;
  if (metadata.reasoning_trajectory) return metadata.reasoning_trajectory;
  if (!metadata.selection_raw && !metadata.document_selections) return null;
  return {
    query: "",
    document_selection: {
      source: "legacy_metadata",
      raw_response: metadata.document_selection_raw || {},
      accepted_selections: metadata.document_selections || [],
    },
    document_walks: Object.entries(metadata.node_selection_raw_by_document || {})
      .filter(([_docId, raw]) => raw && typeof raw === "object" && !Array.isArray(raw))
      .map(([docId, raw]) => ({
        document_id: docId,
        source: raw.fallback ? "keyword_fallback" : "legacy_metadata",
        steps: raw.trace || [],
        final_selections: raw.selections || [],
      })),
    retrieved_nodes: (result.retrieved_spans || []).map(span => ({
      document_id: span.document_id,
      node_id: (span.metadata || {}).node_id,
      node_title: (span.metadata || {}).node_title,
      reason: (span.metadata || {}).reason,
      start_char: span.start_char,
      end_char: span.end_char,
      score: span.score,
    })),
    errors: result.error ? [result.error] : [],
  };
}

function renderReasoning() {
  const items = pageindexReasoningItems();
  if (currentReasoningExample >= items.length) currentReasoningExample = 0;
  const list = items.map((item, idx) => {
    const selectedDocs = (((item.trajectory.document_selection || {}).accepted_selections) || []).length;
    const retrievedNodes = (item.trajectory.retrieved_nodes || []).length;
    return `<div class="reasoning-item ${idx === currentReasoningExample ? "active" : ""}" data-idx="${idx}">
      <div class="ex-id">${esc(item.ex.benchmark)} / ${esc(item.ex.id)}</div>
      <div class="ex-q">${esc((item.ex.query || "").slice(0, 160))}${(item.ex.query || "").length > 160 ? "..." : ""}</div>
      <div class="badges">
        <span class="badge">${esc(item.method || "pageindex")}</span>
        <span class="badge">docs ${esc(selectedDocs)}</span>
        <span class="badge">nodes ${esc(retrievedNodes)}</span>
      </div>
    </div>`;
  }).join("");
  $("#reasoning").innerHTML = `
    <div class="reasoning-layout">
      <div class="panel reasoning-list">${list || `<div class="empty">No PageIndex reasoning traces in this run.</div>`}</div>
      <div id="reasoning-detail"></div>
    </div>`;
  document.querySelectorAll(".reasoning-item").forEach(el => {
    el.onclick = () => {
      currentReasoningExample = Number(el.dataset.idx);
      renderReasoning();
    };
  });
  if (items.length) renderReasoningDetail(items[currentReasoningExample]);
}

function renderReasoningDetail(item) {
  const trajectory = item.trajectory || {};
  const docSelection = trajectory.document_selection || {};
  const selectedDocs = docSelection.accepted_selections || [];
  const docBlocks = selectedDocs.map(doc => `
    <div class="candidate selected">
      <div class="node-title">${esc(doc.document_id)}</div>
      ${doc.reason ? `<div class="reason">${esc(doc.reason)}</div>` : ""}
    </div>`).join("");
  const walks = (trajectory.document_walks || []).map(renderDocumentWalk).join("");
  const retrieved = (trajectory.retrieved_nodes || []).map(node => `
    <div class="candidate selected">
      <div class="node-title">${esc(node.document_id)} · ${esc(node.node_id || "")} ${node.node_title ? `· ${esc(node.node_title)}` : ""}</div>
      <div class="node-id">[${esc(node.start_char)}:${esc(node.end_char)}] score=${fmt(node.score)}</div>
      ${node.reason ? `<div class="reason">${esc(node.reason)}</div>` : ""}
    </div>`).join("");
  $("#reasoning-detail").innerHTML = `
    <div class="panel trace-section">
      <h2>Question</h2>
      <div class="badges"><span class="badge">${esc(item.method || "pageindex")}</span></div>
      <div class="query">${esc(item.ex.query)}</div>
    </div>
    <div class="panel trace-section">
      <h2>Document Selection</h2>
      <div class="badges">
        <span class="badge">${esc(docSelection.source || "unknown")}</span>
        <span class="badge">candidates ${esc(docSelection.candidate_document_count || "?")}</span>
        ${docSelection.catalog_truncated ? `<span class="badge mid">catalog truncated</span>` : ""}
      </div>
      <div class="candidate-list">${docBlocks || `<div class="empty">No selected documents</div>`}</div>
      ${docSelection.catalog_preview ? jsonDetails("Catalog preview", docSelection.catalog_preview) : ""}
      ${jsonDetails("Raw document response", docSelection.raw_response)}
    </div>
    <div class="panel trace-section">
      <h2>ToC Tree Walks</h2>
      ${walks || `<div class="empty">No tree-walk trace</div>`}
    </div>
    <div class="panel trace-section">
      <h2>Final Retrieved Nodes</h2>
      <div class="candidate-list">${retrieved || `<div class="empty">No retrieved nodes</div>`}</div>
    </div>
    ${(trajectory.errors || []).length ? `<div class="panel trace-section"><h2>Errors and Fallbacks</h2><p class="err">${esc((trajectory.errors || []).join("; "))}</p></div>` : ""}`;
}

function renderDocumentWalk(walk) {
  const finalSelections = (walk.final_selections || []).map(sel =>
    `<span class="badge">${esc(sel.node_id || "")}</span>`).join("");
  const steps = (walk.steps || []).map(renderTraceStep).join("");
  return `<div class="trace-step">
    <div class="trace-title">${esc(walk.document_id)} <span class="badge">${esc(walk.source || "unknown")}</span></div>
    ${walk.document_reason ? `<div class="reason">${esc(walk.document_reason)}</div>` : ""}
    ${walk.error ? `<p class="err">${esc(walk.error)}</p>` : ""}
    <div class="badges">${finalSelections}</div>
    ${steps || `<div class="empty">No child-selection steps recorded.</div>`}
  </div>`;
}

function renderTraceStep(step) {
  const current = step.current_node || {
    node_id: step.node_id,
    unit_start: step.unit_start,
    unit_end: step.unit_end,
  };
  const accepted = new Map((step.accepted_selections || []).map(sel => [String(sel.node_id || ""), sel]));
  const candidates = (step.candidate_children || []).map(child => {
    const selection = accepted.get(String(child.node_id || ""));
    return `<div class="candidate ${selection ? "selected" : ""}">
      <div class="node-title">${esc(child.title || child.node_id)} <span class="node-id">${esc(child.node_id)} ${unitLabel(child)}</span></div>
      ${child.summary ? `<div class="node-summary">${esc(child.summary)}</div>` : ""}
      ${selection?.reason ? `<div class="reason">${esc(selection.reason)}</div>` : ""}
    </div>`;
  }).join("");
  const fallback = step.selection_source === "keyword_fallback"
    ? `<span class="badge mid">${esc(step.fallback_reason || "keyword fallback")}</span>`
    : "";
  return `<div class="trace-step">
    <div class="trace-title">Step ${esc(step.step || "")}: ${esc(current.title || current.node_id || "")} <span class="node-id">${esc(current.node_id || "")} ${unitLabel(current)}</span></div>
    <div class="badges">
      <span class="badge">${esc(step.selection_source || "llm")}</span>
      <span class="badge">children ${esc(step.candidate_child_count || step.child_count || 0)}</span>
      ${step.prompt_child_list_truncated ? `<span class="badge mid">prompt truncated</span>` : ""}
      ${fallback}
    </div>
    <div class="candidate-list">${candidates || renderSelectionList(step.accepted_selections || step.llm_selections || [])}</div>
    ${jsonDetails("Raw node response", step.raw_response || step.selection_raw)}
  </div>`;
}

function renderSelectionList(selections) {
  return (selections || []).map(sel => `
    <div class="candidate selected">
      <div class="node-title">${esc(sel.node_id || "")}</div>
      ${sel.reason ? `<div class="reason">${esc(sel.reason)}</div>` : ""}
    </div>`).join("") || `<div class="empty">No selections</div>`;
}

function unitLabel(node) {
  const start = node.unit_start;
  const end = node.unit_end;
  if (start == null || end == null) return "";
  return start === end ? `[u${esc(start)}]` : `[u${esc(start)}-u${esc(end)}]`;
}

function jsonDetails(title, value) {
  if (value == null || value === "") return "";
  const body = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  if (!body || body === "{}" || body === "[]") return "";
  return `<details><summary>${esc(title)}</summary><pre>${esc(body)}</pre></details>`;
}

function renderToc() {
  const trees = currentRun.toc_trees || [];
  const list = trees.map((item, idx) => {
    const method = item.method || "pageindex";
    return `<div class="toc-doc ${idx === currentTocDoc ? "active" : ""}" data-idx="${idx}">
      <span class="badge">${esc(method)}</span> ${esc(item.document_id)}
    </div>`;
  }).join("");
  $("#toc").innerHTML = `
    <div class="toc-layout">
      <div class="panel toc-list">${list || `<div class="empty">No PageIndex trees in this run.</div>`}</div>
      <div class="panel tree" id="toc-tree"></div>
    </div>`;
  document.querySelectorAll(".toc-doc").forEach(el => {
    el.onclick = () => {
      currentTocDoc = Number(el.dataset.idx);
      renderToc();
    };
  });
  if (trees.length) $("#toc-tree").innerHTML = renderTree(trees[currentTocDoc].tree);
}

function renderTree(node) {
  const children = (node.children || []).map(renderTree).join("");
  return `<ul><li>
    <div class="node-title">${esc(node.title)} <span class="node-id">${esc(node.node_id)} [${esc(node.start_char)}:${esc(node.end_char)}]</span></div>
    <div class="node-summary">${esc(node.summary || "")}</div>
    ${children}
  </li></ul>`;
}

init();
</script>
</body>
</html>
"""
