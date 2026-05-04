from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_dashboard(results_dir: str | Path, output_path: str | Path) -> None:
    results_dir = Path(results_dir)
    runs = []
    for path in sorted(results_dir.glob("*/run.json"), reverse=True):
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        run["run_dir"] = str(path.parent.relative_to(results_dir.parent))
        runs.append(run)

    html = HTML_TEMPLATE.replace("__RUNS_JSON__", json.dumps(runs, ensure_ascii=False))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


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
    primary_metric: "document_f1",
    primary_metric_family: "document",
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
  const reasoningCount = pageindexReasoningItems().length;
  const rows = Object.entries(agg.by_method || {}).map(([method, s]) => `
    <tr>
      <td><span class="badge">${esc(method)}</span></td>
      <td>${esc(s.n)}</td>
      <td><span class="badge ${f1Class(s.mean_document_f1)}">${fmt(s.mean_document_f1)}</span></td>
      <td>${fmt(s.mean_document_precision)}</td>
      <td>${fmt(s.mean_document_recall)}</td>
      <td>${fmt(s.mean_f1)}</td>
      <td>${fmt(s.mean_precision)}</td>
      <td>${fmt(s.mean_recall)}</td>
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
      <p class="meta">Primary scores are document-level retrieval metrics from unique matched <code>document_id</code> values. Span overlap and generated answers are retained only for diagnostics.</p>
      <table><thead><tr><th>Method</th><th>n</th><th>Doc F1</th><th>Doc P</th><th>Doc R</th><th>Span F1</th><th>Span P</th><th>Span R</th><th>Total cost</th><th>Setup cost</th><th>Latency</th><th>Errors</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="12" class="empty">No metrics</td></tr>`}</tbody></table>
    </div>`;
}

function renderExamples() {
  const examples = currentRun.examples || [];
  const list = examples.map((ex, idx) => {
    const badges = Object.entries(ex.methods || {}).map(([m, r]) =>
      `<span class="badge ${f1Class(r.document_f1)}">${esc(m)} Doc F1 ${fmt(r.document_f1, 2)}</span>`).join("");
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
  return `<div class="panel method ${esc(method)}">
    <h3><span class="badge">${esc(method)}</span> <span class="badge ${f1Class(result.document_f1)}">Doc F1 ${fmt(result.document_f1)}</span></h3>
    <div class="badges">
      <span class="badge">Doc P ${fmt(result.document_precision)}</span>
      <span class="badge">Doc R ${fmt(result.document_recall)}</span>
      <span class="badge">docs ${esc(result.matched_document_count || 0)}/${esc(result.retrieved_document_count || 0)}</span>
      <span class="badge">spans ${esc(result.retrieved_span_count || (result.retrieved_spans || []).length)}</span>
      <span class="badge">Span F1 ${fmt(result.f1)}</span>
      <span class="badge">Span P ${fmt(result.precision)}</span>
      <span class="badge">Span R ${fmt(result.recall)}</span>
      <span class="badge">${money(result.estimated_cost_usd)}</span>
      <span class="badge">${fmt(result.wall_clock_seconds, 1)}s</span>
      <span class="badge">in ${esc(result.input_tokens || 0)}</span>
      <span class="badge">out ${esc(result.output_tokens || 0)}</span>
    </div>
    ${result.error ? `<p class="err">${esc(result.error)}</p>` : ""}
    ${answerBlock}
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
    </div>
    <div class="span-text">${esc(span.text)}</div>
  </div>`;
}

function pageindexReasoningItems() {
  return (currentRun?.examples || []).map((ex, idx) => {
    const result = (ex.methods || {}).pageindex;
    const trajectory = result ? pageindexTrajectory(result) : null;
    return { ex, idx, result, trajectory };
  }).filter(item => item.trajectory);
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
  const list = trees.map((item, idx) => `<div class="toc-doc ${idx === currentTocDoc ? "active" : ""}" data-idx="${idx}">${esc(item.document_id)}</div>`).join("");
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
