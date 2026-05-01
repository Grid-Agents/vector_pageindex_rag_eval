from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .json_utils import compact_json, extract_json_object
from .text_splitters import make_chunks
from .types import Document, RetrievedSpan, RetrievalOutput, Usage


class PageIndexRAG:
    """Minimal PageIndex-style RAG for LegalBench text files.

    The upstream PageIndex project is optimized around PDF/Markdown indexing. This
    adapter keeps the same idea for LegalBench's raw text corpus: build a semantic
    ToC tree with an LLM, then ask the LLM to navigate that tree at query time.
    """

    name = "pageindex"

    def __init__(self, cfg: dict[str, Any], llm: Any, *, cache_dir: Path):
        self.cfg = cfg
        self.llm = llm
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, Document] = {}
        self.trees: dict[str, dict[str, Any]] = {}
        self.node_index: dict[str, tuple[str, dict[str, Any]]] = {}
        self.setup_usage = Usage()

    def build(self, documents: list[Document]) -> None:
        self.documents = {doc.document_id: doc for doc in documents}
        for doc in documents:
            tree, usage = self._load_or_build_tree(doc)
            self.setup_usage.add(usage)
            self.trees[doc.document_id] = tree
            self._index_tree(doc.document_id, tree)

    def query(self, query: str) -> RetrievalOutput:
        if not self.trees:
            raise RuntimeError("Call build() before query().")

        usage = Usage()
        metadata: dict[str, Any] = {}
        tree_text = self._format_forest(
            max_chars=int(self.cfg.get("max_tree_chars", 24000))
        )
        system = (
            "You are a PageIndex retrieval agent. Reason over the semantic table "
            "of contents and select the smallest relevant nodes for the legal query."
        )
        user = (
            f"Query:\n{query}\n\n"
            f"Semantic ToC forest:\n{tree_text}\n\n"
            f"Select up to {int(self.cfg.get('selected_nodes', 5))} "
            "relevant node_id values. "
            'Return JSON only: {"selections":[{"node_id":"...","reason":"..."}]}.'
        )
        error = ""
        try:
            response = self.llm.complete(system=system, user=user, max_tokens=700)
            usage.add(response.usage)
            parsed = extract_json_object(response.text)
            selections = parsed.get("selections", [])
            metadata["selection_raw"] = parsed
        except Exception as exc:  # noqa: BLE001
            selections = self._keyword_fallback(query)
            error = f"pageindex_selection_error: {type(exc).__name__}: {exc}"
            metadata["selection_raw"] = {"selections": selections, "fallback": True}

        spans: list[RetrievedSpan] = []
        seen: set[str] = set()
        for rank, item in enumerate(selections):
            node_id = str(item.get("node_id", ""))
            if not node_id or node_id in seen or node_id not in self.node_index:
                continue
            seen.add(node_id)
            doc_id, node = self.node_index[node_id]
            doc = self.documents[doc_id]
            start = int(node["start_char"])
            end = int(node["end_char"])
            max_chars = int(self.cfg.get("max_retrieved_chars_per_node", 5000))
            if max_chars > 0:
                end = min(end, start + max_chars)
            spans.append(
                RetrievedSpan(
                    document_id=doc_id,
                    start_char=start,
                    end_char=end,
                    text=doc.text[start:end],
                    score=1.0 / (rank + 1),
                    metadata={
                        "retriever": "pageindex",
                        "node_id": node_id,
                        "node_title": node.get("title", ""),
                        "reason": item.get("reason", ""),
                    },
                )
            )
        return RetrievalOutput(spans=spans, usage=usage, metadata=metadata, error=error)

    def toc_trees(self) -> list[dict[str, Any]]:
        return [
            {"document_id": doc_id, "tree": tree}
            for doc_id, tree in sorted(self.trees.items())
        ]

    def _load_or_build_tree(self, doc: Document) -> tuple[dict[str, Any], Usage]:
        cache_path = self.cache_dir / f"{self._doc_hash(doc)}.json"
        if cache_path.exists() and not self.cfg.get("force_reindex", False):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f), Usage()

        tree = self._fallback_tree(doc)
        usage = Usage()
        if self.cfg.get("build_with_llm", True):
            try:
                tree, usage = self._semanticize_tree(doc, tree)
            except Exception as exc:  # noqa: BLE001
                tree.setdefault("metadata", {})["build_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)
        return tree, usage

    def _fallback_tree(self, doc: Document) -> dict[str, Any]:
        chunks = make_chunks(
            doc,
            strategy="hierarchical",
            chunk_size=int(self.cfg.get("max_leaf_chars", 2400)),
            chunk_overlap=0,
        )
        slug = hashlib.sha1(doc.document_id.encode("utf-8")).hexdigest()[:8]
        leaves = [
            {
                "node_id": f"{slug}-{idx:04d}",
                "title": chunk.title or f"Section {idx + 1}",
                "summary": _preview(chunk.text),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "children": [],
            }
            for idx, chunk in enumerate(chunks)
        ]
        group_size = max(1, int(self.cfg.get("group_size", 8)))
        groups = []
        for idx in range(0, len(leaves), group_size):
            children = leaves[idx : idx + group_size]
            groups.append(
                {
                    "node_id": f"{slug}-g{idx // group_size:03d}",
                    "title": f"Sections {idx + 1}-{idx + len(children)}",
                    "summary": "; ".join(child["title"] for child in children[:5]),
                    "start_char": children[0]["start_char"],
                    "end_char": children[-1]["end_char"],
                    "children": children,
                }
            )
        return {
            "node_id": f"{slug}-root",
            "title": doc.document_id,
            "summary": _preview(doc.text, 600),
            "start_char": 0,
            "end_char": len(doc.text),
            "children": groups,
            "metadata": {"source": "local_pageindex"},
        }

    def _semanticize_tree(
        self, doc: Document, tree: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]:
        nodes = _flatten_tree(tree)[: int(self.cfg.get("max_build_nodes", 80))]
        outline = []
        for node in nodes:
            outline.append(
                {
                    "node_id": node["node_id"],
                    "title": node.get("title", ""),
                    "range": [node["start_char"], node["end_char"]],
                    "preview": _preview(
                        doc.text[int(node["start_char"]) : int(node["end_char"])], 500
                    ),
                }
            )

        system = (
            "You build semantic table-of-contents metadata for legal documents. "
            "Do not change node_id values or character ranges."
        )
        user = (
            f"Document id: {doc.document_id}\n"
            f"Outline nodes:\n{compact_json(outline)}\n\n"
            'Return JSON only: {"document_summary":"...",'
            '"nodes":[{"node_id":"...","title":"...","summary":"..."}]}.'
        )
        response = self.llm.complete(system=system, user=user, max_tokens=1800)
        parsed = extract_json_object(response.text)
        tree["summary"] = str(parsed.get("document_summary", tree.get("summary", "")))
        by_id = {str(item.get("node_id")): item for item in parsed.get("nodes", [])}
        for node in _flatten_tree(tree):
            item = by_id.get(node["node_id"])
            if not item:
                continue
            if item.get("title"):
                node["title"] = str(item["title"])[:180]
            if item.get("summary"):
                node["summary"] = str(item["summary"])[:800]
        tree.setdefault("metadata", {})["semanticized_by_llm"] = True
        return tree, response.usage

    def _index_tree(self, doc_id: str, tree: dict[str, Any]) -> None:
        for node in _flatten_tree(tree):
            self.node_index[node["node_id"]] = (doc_id, node)

    def _format_forest(self, *, max_chars: int) -> str:
        lines: list[str] = []
        for doc_id, tree in sorted(self.trees.items()):
            lines.append(f"DOCUMENT {doc_id}")
            _format_node(tree, lines, depth=0)
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...TRUNCATED..."

    def _keyword_fallback(self, query: str) -> list[dict[str, str]]:
        terms = {term.lower() for term in query.split() if len(term) > 3}
        scored = []
        for node_id, (_doc_id, node) in self.node_index.items():
            haystack = f"{node.get('title', '')} {node.get('summary', '')}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, node_id))
        scored.sort(reverse=True)
        return [
            {"node_id": node_id, "reason": "keyword fallback"}
            for _score, node_id in scored[: int(self.cfg.get("selected_nodes", 5))]
        ]

    def _doc_hash(self, doc: Document) -> str:
        payload = f"{doc.document_id}\0{len(doc.text)}\0{hashlib.sha1(doc.text.encode('utf-8')).hexdigest()}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _flatten_tree(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        out.extend(_flatten_tree(child))
    return out


def _format_node(node: dict[str, Any], lines: list[str], *, depth: int) -> None:
    indent = "  " * depth
    lines.append(
        f"{indent}- {node['node_id']} [{node['start_char']}:{node['end_char']}] "
        f"{node.get('title', '')}: {node.get('summary', '')}"
    )
    for child in node.get("children", []) or []:
        _format_node(child, lines, depth=depth + 1)


def _preview(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]
