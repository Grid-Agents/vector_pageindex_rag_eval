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
<title>LegalBench-RAG Evaluation</title>
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
.hidden { display: none; }
.empty { color: #888; font-style: italic; padding: 12px; }
@media (max-width: 900px) {
  .cards, .grid2, .layout, .toc-layout { grid-template-columns: 1fr; }
  .tabs { margin-left: 0; }
  .topbar { flex-wrap: wrap; }
}
</style>
</head>
<body>
<div class="topbar">
  <h1>LegalBench-RAG Evaluation</h1>
  <label>Run <select id="run-select"></select></label>
  <span id="run-meta" class="meta"></span>
  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="examples">Examples</button>
    <button class="tab" data-tab="toc">PageIndex ToC</button>
  </div>
</div>
<main class="page">
  <section id="overview"></section>
  <section id="examples" class="hidden"></section>
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
let currentTocDoc = 0;

function init() {
  const select = $("#run-select");
  select.innerHTML = RUNS.map((run, idx) => `<option value="${idx}">${esc(run.run_id)}</option>`).join("");
  select.onchange = () => renderRun(Number(select.value));
  document.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      ["overview", "examples", "toc"].forEach(id => $("#" + id).classList.add("hidden"));
      $("#" + btn.dataset.tab).classList.remove("hidden");
    };
  });
  if (RUNS.length) renderRun(0);
  else $("#overview").innerHTML = `<div class="panel empty">No runs found. Run an experiment first.</div>`;
}

function renderRun(idx) {
  currentRun = RUNS[idx];
  currentExample = 0;
  currentTocDoc = 0;
  const cfg = currentRun.config || {};
  const runCfg = cfg.run || {};
  $("#run-meta").innerHTML = `${esc(currentRun.run_dir || "")} · n=${esc(runCfg.n || "?")} · methods=${esc((runCfg.methods || []).join(", "))}`;
  renderOverview();
  renderExamples();
  renderToc();
}

function renderOverview() {
  const agg = currentRun.aggregates || {};
  const rows = Object.entries(agg.by_method || {}).map(([method, s]) => `
    <tr>
      <td><span class="badge">${esc(method)}</span></td>
      <td>${esc(s.n)}</td>
      <td><span class="badge ${f1Class(s.mean_f1)}">${fmt(s.mean_f1)}</span></td>
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
      <div class="card"><div class="label">ToC Trees</div><div class="value">${esc((currentRun.toc_trees || []).length)}</div></div>
    </div>
    <div class="panel">
      <h2>Method Aggregates</h2>
      <table><thead><tr><th>Method</th><th>n</th><th>F1</th><th>Precision</th><th>Recall</th><th>Total cost</th><th>Setup cost</th><th>Latency</th><th>Errors</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="9" class="empty">No metrics</td></tr>`}</tbody></table>
    </div>`;
}

function renderExamples() {
  const examples = currentRun.examples || [];
  const list = examples.map((ex, idx) => {
    const badges = Object.entries(ex.methods || {}).map(([m, r]) =>
      `<span class="badge ${f1Class(r.f1)}">${esc(m)} ${fmt(r.f1, 2)}</span>`).join("");
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
      <h2>Golden Answer Spans</h2>
      ${gold || `<div class="empty">No gold spans</div>`}
    </div>
    <div class="grid2">${methods}</div>`;
}

function methodPanel(method, result) {
  const spans = (result.retrieved_spans || []).map(spanBlock).join("");
  return `<div class="panel method ${esc(method)}">
    <h3><span class="badge">${esc(method)}</span> <span class="badge ${f1Class(result.f1)}">F1 ${fmt(result.f1)}</span></h3>
    <div class="badges">
      <span class="badge">P ${fmt(result.precision)}</span>
      <span class="badge">R ${fmt(result.recall)}</span>
      <span class="badge">${money(result.estimated_cost_usd)}</span>
      <span class="badge">${fmt(result.wall_clock_seconds, 1)}s</span>
      <span class="badge">in ${esc(result.input_tokens || 0)}</span>
      <span class="badge">out ${esc(result.output_tokens || 0)}</span>
    </div>
    ${result.error ? `<p class="err">${esc(result.error)}</p>` : ""}
    <div class="answer">${esc(result.answer || "No answer recorded.")}</div>
    <h2>Retrieved Documents</h2>
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

