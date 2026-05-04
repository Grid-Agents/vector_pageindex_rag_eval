from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

from ..json_utils import compact_json, extract_json_object
from ..pageindex_rag import _estimate_tokens, _pack_virtual_page_spans, _preview
from ..text_splitters import detect_sections
from ..types import Document, RetrievedSpan, RetrievalOutput, Usage
from .loader import load_official_pageindex_modules


_CHARS_PER_TOKEN = 4


class OfficialPageIndexRAG:
    """Adapter around VectifyAI's official self-hosted PageIndex implementation.

    LegalBench-RAG stores plain text rather than PDFs. The official self-hosted
    repo supports Markdown, so this adapter converts each text document into a
    synthetic Markdown document with section headings and virtual-page headings,
    lets the official `md_to_tree` implementation build the tree, then uses the
    official cookbook's LLM tree-search pattern to retrieve nodes.
    """

    name = "pageindex_official"

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
        for doc in tqdm(
            documents,
            desc="Official PageIndex documents",
            unit="doc",
            disable=len(documents) <= 1,
        ):
            tree, usage = self._load_or_build_tree(doc)
            self.setup_usage.add(usage)
            self.trees[doc.document_id] = tree
            self._index_tree(doc.document_id, tree)

    def query(self, query: str) -> RetrievalOutput:
        if not self.trees:
            raise RuntimeError("Call build() before query().")

        usage = Usage()
        metadata: dict[str, Any] = {}
        error_parts: list[str] = []
        record_reasoning = self._record_reasoning_trajectory()
        reasoning_trajectory: dict[str, Any] | None = None
        if record_reasoning:
            reasoning_trajectory = {
                "query": query,
                "method": self.name,
                "document_selection": {
                    "source": "",
                    "candidate_document_count": len(self.trees),
                    "raw_response": {},
                    "requested_selections": [],
                    "accepted_selections": [],
                },
                "document_walks": [],
                "retrieved_nodes": [],
                "errors": error_parts,
            }

        try:
            document_selections, parsed = self._select_documents(query)
            usage.add(parsed["usage"])
            metadata["document_selection_raw"] = parsed["raw"]
            if reasoning_trajectory is not None:
                reasoning_trajectory["document_selection"].update(
                    {
                        "source": "llm",
                        "catalog_char_count": parsed.get("catalog_char_count", 0),
                        "catalog_truncated": parsed.get("catalog_truncated", False),
                        "catalog_preview": parsed.get("catalog_preview", ""),
                        "raw_response": parsed["raw"],
                        "requested_selections": document_selections,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            document_selections = self._keyword_document_fallback(query)
            error_text = (
                f"pageindex_official_document_selection_error: "
                f"{type(exc).__name__}: {exc}"
            )
            error_parts.append(error_text)
            metadata["document_selection_raw"] = {
                "documents": document_selections,
                "fallback": True,
            }
            if reasoning_trajectory is not None:
                reasoning_trajectory["document_selection"].update(
                    {
                        "source": "keyword_fallback",
                        "error": error_text,
                        "requested_selections": document_selections,
                    }
                )

        if not document_selections:
            document_selections = self._keyword_document_fallback(query)
            if document_selections:
                error_parts.append("pageindex_official_document_selection_empty")
                metadata["document_selection_fallback"] = document_selections

        normalized_document_selections = self._normalize_document_selections(
            document_selections
        )
        if not normalized_document_selections:
            fallback_documents = self._keyword_document_fallback(query)
            normalized_document_selections = self._normalize_document_selections(
                fallback_documents
            )
            if normalized_document_selections:
                error_parts.append("pageindex_official_document_selection_invalid")
                metadata["document_selection_fallback"] = fallback_documents

        metadata["document_selections"] = normalized_document_selections
        if reasoning_trajectory is not None:
            reasoning_trajectory["document_selection"][
                "accepted_selections"
            ] = normalized_document_selections

        spans: list[RetrievedSpan] = []
        node_metadata: dict[str, Any] = {}
        seen_node_ids: set[str] = set()
        rank = 0
        selected_nodes = max(1, int(self.cfg.get("selected_nodes", 10)))

        for document_item in normalized_document_selections[
            : max(1, int(self.cfg.get("selected_documents", 3)))
        ]:
            doc_id = document_item["document_id"]
            walk_record: dict[str, Any] | None = None
            if reasoning_trajectory is not None:
                walk_record = {
                    "document_id": doc_id,
                    "document_reason": document_item.get("reason", ""),
                    "source": "",
                    "steps": [],
                    "final_selections": [],
                }
            try:
                node_selections, parsed = self._select_nodes_for_document(query, doc_id)
                usage.add(parsed["usage"])
                node_metadata[doc_id] = parsed["raw"]
                if walk_record is not None:
                    walk_record.update(
                        {
                            "source": "official_llm_tree_search",
                            "steps": parsed["raw"].get("trace", []),
                            "final_selections": node_selections,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                node_selections = self._keyword_node_fallback(query, doc_id)
                error_text = (
                    f"pageindex_official_node_selection_error[{doc_id}]: "
                    f"{type(exc).__name__}: {exc}"
                )
                error_parts.append(error_text)
                node_metadata[doc_id] = {
                    "selections": node_selections,
                    "fallback": True,
                }
                if walk_record is not None:
                    walk_record.update(
                        {
                            "source": "keyword_fallback",
                            "error": error_text,
                            "final_selections": node_selections,
                        }
                    )

            if not node_selections:
                node_selections = self._keyword_node_fallback(query, doc_id)
                if node_selections:
                    error_parts.append(
                        f"pageindex_official_node_selection_empty[{doc_id}]"
                    )
                    node_metadata[f"{doc_id}__fallback"] = node_selections
                    if walk_record is not None:
                        walk_record.update(
                            {
                                "source": "keyword_fallback",
                                "fallback_reason": "llm_returned_no_nodes",
                                "final_selections": node_selections,
                            }
                        )

            if walk_record is not None:
                reasoning_trajectory["document_walks"].append(walk_record)

            for item in node_selections:
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("node_id", ""))
                if not node_id or node_id in seen_node_ids or node_id not in self.node_index:
                    continue
                node_doc_id, node = self.node_index[node_id]
                if node_doc_id != doc_id:
                    continue
                seen_node_ids.add(node_id)
                doc = self.documents[doc_id]
                start = int(node["start_char"])
                end = int(node["end_char"])
                max_chars = int(self.cfg.get("max_retrieved_chars_per_node", 5000))
                if max_chars > 0:
                    end = min(end, start + max_chars)
                rank += 1
                retrieved_node = {
                    "document_id": doc_id,
                    "document_reason": document_item.get("reason", ""),
                    "node_id": node_id,
                    "node_title": node.get("title", ""),
                    "reason": item.get("reason", ""),
                    "unit_start": node.get("unit_start"),
                    "unit_end": node.get("unit_end"),
                    "start_char": start,
                    "end_char": end,
                    "score": 1.0 / rank,
                }
                if reasoning_trajectory is not None:
                    reasoning_trajectory["retrieved_nodes"].append(retrieved_node)
                spans.append(
                    RetrievedSpan(
                        document_id=doc_id,
                        start_char=start,
                        end_char=end,
                        text=doc.text[start:end],
                        score=1.0 / rank,
                        metadata={
                            "retriever": self.name,
                            "document_reason": document_item.get("reason", ""),
                            "node_id": node_id,
                            "official_node_id": node.get("official_node_id", ""),
                            "node_title": node.get("title", ""),
                            "reason": item.get("reason", ""),
                            "unit_start": node.get("unit_start"),
                            "unit_end": node.get("unit_end"),
                        },
                    )
                )
                if len(spans) >= selected_nodes:
                    break
            if len(spans) >= selected_nodes:
                break

        metadata["node_selection_raw_by_document"] = node_metadata
        metadata["selection_raw"] = {
            "documents": normalized_document_selections,
            "nodes_by_document": node_metadata,
        }
        if reasoning_trajectory is not None:
            reasoning_trajectory["errors"] = error_parts
            metadata["reasoning_trajectory"] = reasoning_trajectory
        return RetrievalOutput(
            spans=spans,
            usage=usage,
            metadata=metadata,
            error="; ".join(error_parts),
        )

    def toc_trees(self) -> list[dict[str, Any]]:
        return [
            {"document_id": doc_id, "tree": tree}
            for doc_id, tree in sorted(self.trees.items())
        ]

    def _load_or_build_tree(self, doc: Document) -> tuple[dict[str, Any], Usage]:
        cache_path = self.cache_dir / f"{self._doc_hash(doc)}.json"
        if cache_path.exists() and not self.cfg.get("force_reindex", False):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload["tree"], Usage()

        tree, usage, build_metadata = self._build_tree(doc)
        payload = {
            "document_id": doc.document_id,
            "official_pageindex_source": build_metadata.get("official_source", ""),
            "markdown_path": build_metadata.get("markdown_path", ""),
            "usage": usage.to_dict(),
            "tree": tree,
        }
        self._write_json_atomic(cache_path, payload)
        return tree, usage

    def _build_tree(self, doc: Document) -> tuple[dict[str, Any], Usage, dict[str, Any]]:
        markdown_text, line_map = self._document_to_markdown(doc)
        markdown_dir = self.cache_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_dir / f"{self._doc_hash(doc)}.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")

        usage = Usage()
        build_with_llm = bool(self.cfg.get("build_with_llm", True))
        if_add_node_summary = (
            _yes_no(self.cfg.get("if_add_node_summary"), "yes")
            if build_with_llm
            else "no"
        )
        if_add_doc_description = (
            _yes_no(self.cfg.get("if_add_doc_description"), "yes")
            if build_with_llm
            else "no"
        )
        modules = self._load_official_modules()
        with _patch_official_llm(modules.utils, self.llm, usage):
            result = _run_async(
                modules.md_to_tree(
                    md_path=str(markdown_path),
                    if_thinning=bool(self.cfg.get("if_thinning", False)),
                    min_token_threshold=int(self.cfg.get("thinning_threshold", 5000)),
                    if_add_node_summary=if_add_node_summary,
                    summary_token_threshold=int(
                        self.cfg.get("summary_token_threshold", 200)
                    ),
                    model=self.cfg.get("model"),
                    if_add_doc_description=if_add_doc_description,
                    if_add_node_text=_yes_no(self.cfg.get("if_add_node_text"), "no"),
                    if_add_node_id="yes",
                )
            )

        tree = self._normalize_official_tree(doc, result, line_map)
        tree.setdefault("metadata", {})["official_pageindex_source"] = modules.source
        tree.setdefault("metadata", {})["generated_markdown_path"] = str(markdown_path)
        return tree, usage, {
            "official_source": modules.source,
            "markdown_path": str(markdown_path),
        }

    def _load_official_modules(self) -> Any:
        return load_official_pageindex_modules(self.cfg, cache_dir=self.cache_dir)

    def _document_to_markdown(
        self, doc: Document
    ) -> tuple[str, dict[int, dict[str, Any]]]:
        line_map: dict[int, dict[str, Any]] = {}
        lines: list[str] = []
        current_line = 1
        unit_index = 1
        section_units: list[dict[str, Any]] = []
        for section_index, (title, start, end) in enumerate(detect_sections(doc.text)):
            section_text = doc.text[start:end]
            spans = _pack_virtual_page_spans(
                section_text,
                target_chars=self._virtual_page_target_tokens() * _CHARS_PER_TOKEN,
                max_chars=self._virtual_page_max_tokens() * _CHARS_PER_TOKEN,
            )
            if not spans:
                continue
            section_start_unit = unit_index
            section_end_unit = unit_index + len(spans) - 1
            section_line = current_line
            section_title = title if title != "Document start" else doc.document_id
            lines.append(f"# {_clean_heading(section_title)}")
            current_line += 1
            line_map[section_line] = {
                "start_char": start,
                "end_char": end,
                "unit_start": section_start_unit,
                "unit_end": section_end_unit,
                "section_title": "" if title == "Document start" else title,
                "node_kind": "section",
                "section_index": section_index,
            }
            lines.append("")
            current_line += 1

            for part_index, (local_start, local_end) in enumerate(spans, start=1):
                abs_start = start + local_start
                abs_end = start + local_end
                page_line = current_line
                page_title = _page_title(
                    title="" if title == "Document start" else title,
                    unit_index=unit_index,
                    part_index=part_index,
                    section_parts=len(spans),
                )
                lines.append(f"## {_clean_heading(page_title)}")
                current_line += 1
                page_text = _escape_markdown_headings(doc.text[abs_start:abs_end])
                text_lines = page_text.splitlines()
                lines.extend(text_lines)
                current_line += len(text_lines)
                lines.append("")
                current_line += 1
                line_map[page_line] = {
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "unit_start": unit_index,
                    "unit_end": unit_index,
                    "section_title": "" if title == "Document start" else title,
                    "node_kind": "page",
                    "section_index": section_index,
                    "part_index": part_index,
                    "section_parts": len(spans),
                }
                section_units.append(line_map[page_line])
                unit_index += 1

        markdown = "\n".join(lines).rstrip() + "\n"
        if not section_units:
            raise RuntimeError(f"Could not create official PageIndex markdown for {doc.document_id}")
        return markdown, line_map

    def _normalize_official_tree(
        self,
        doc: Document,
        result: dict[str, Any],
        line_map: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        structure = result.get("structure", [])
        if not isinstance(structure, list):
            structure = []
        slug = hashlib.sha1(doc.document_id.encode("utf-8")).hexdigest()[:8]
        children = [
            self._normalize_official_node(slug, node, line_map)
            for node in structure
            if isinstance(node, dict)
        ]
        children = [child for child in children if child is not None]
        unit_end = max((int(child.get("unit_end", 0)) for child in children), default=0)
        root = {
            "node_id": f"{slug}-official-root",
            "official_node_id": "",
            "title": doc.document_id,
            "summary": str(result.get("doc_description") or _preview(doc.text, 600)),
            "start_char": 0,
            "end_char": len(doc.text),
            "unit_start": 1 if unit_end else 0,
            "unit_end": unit_end,
            "token_count": _estimate_tokens(doc.text),
            "node_kind": "root",
            "section_title": "",
            "children": children,
            "metadata": {
                "source": "official_pageindex_markdown",
                "official_doc_name": result.get("doc_name", ""),
                "line_count": result.get("line_count", 0),
                "unit_type": "virtual_page_markdown_heading",
                "virtual_page_target_tokens": self._virtual_page_target_tokens(),
                "virtual_page_max_tokens": self._virtual_page_max_tokens(),
            },
        }
        return root

    def _normalize_official_node(
        self,
        slug: str,
        node: dict[str, Any],
        line_map: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        official_node_id = str(node.get("node_id", "")).strip()
        line_num = int(node.get("line_num", 0) or 0)
        mapped = line_map.get(line_num)
        raw_children = node.get("nodes", []) or []
        children = [
            self._normalize_official_node(slug, child, line_map)
            for child in raw_children
            if isinstance(child, dict)
        ]
        children = [child for child in children if child is not None]
        if mapped is None and children:
            mapped = {
                "start_char": min(int(child["start_char"]) for child in children),
                "end_char": max(int(child["end_char"]) for child in children),
                "unit_start": min(int(child["unit_start"]) for child in children),
                "unit_end": max(int(child["unit_end"]) for child in children),
                "section_title": "",
                "node_kind": "section",
            }
        if mapped is None:
            return None
        summary = str(node.get("summary") or node.get("prefix_summary") or "")
        if not summary:
            summary = _preview(str(node.get("text") or node.get("title") or ""), 500)
        node_id = f"{slug}-official-{official_node_id or line_num:0>4}"
        return {
            "node_id": node_id,
            "official_node_id": official_node_id,
            "title": str(node.get("title") or ""),
            "summary": summary,
            "start_char": int(mapped["start_char"]),
            "end_char": int(mapped["end_char"]),
            "unit_start": int(mapped["unit_start"]),
            "unit_end": int(mapped["unit_end"]),
            "token_count": _estimate_tokens(summary),
            "node_kind": "section" if children else str(mapped.get("node_kind", "page")),
            "section_title": str(mapped.get("section_title", "")),
            "line_num": line_num,
            "children": children,
        }

    def _select_documents(self, query: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if self.llm is None:
            raise RuntimeError("Official PageIndex document selection requires an LLM")
        max_catalog_chars = int(self.cfg.get("max_document_catalog_chars", 120000))
        catalog_text = self._format_document_catalog(max_chars=max_catalog_chars)
        system = (
            "You are a PageIndex retrieval agent. Select which documents are most "
            "likely to contain the answer to the legal query."
        )
        user = (
            f"Query:\n{query}\n\n"
            f"Document catalog:\n{catalog_text}\n\n"
            f"Select up to {max(1, int(self.cfg.get('selected_documents', 3)))} "
            "document_id values. "
            'Return JSON only: {"documents":[{"document_id":"...","reason":"..."}]}.'
        )
        response = self.llm.complete(system=system, user=user, max_tokens=700)
        parsed = extract_json_object(response.text)
        documents = parsed.get("documents", [])
        if not isinstance(documents, list):
            documents = []
        return documents, {
            "raw": parsed,
            "usage": response.usage,
            "candidate_document_count": len(self.trees),
            "catalog_char_count": len(catalog_text),
            "catalog_truncated": catalog_text.endswith("\n...TRUNCATED..."),
            "catalog_preview": self._reasoning_preview(
                catalog_text,
                int(self.cfg.get("reasoning_max_catalog_chars", 12000)),
            ),
        }

    def _select_nodes_for_document(
        self, query: str, doc_id: str
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if self.llm is None:
            raise RuntimeError("Official PageIndex tree search requires an LLM")
        usage = Usage()
        tree_text = self._format_tree_for_selection(
            self.trees[doc_id],
            max_chars=int(self.cfg.get("max_tree_chars", 24000)),
        )
        system = (
            "You are using the official PageIndex tree-search workflow. Given a "
            "question and a PageIndex tree, identify the smallest relevant nodes."
        )
        user = (
            f"Question: {query}\n\n"
            f"Document id: {doc_id}\n"
            f"PageIndex tree structure:\n{tree_text}\n\n"
            f"Return up to {max(1, int(self.cfg.get('selected_nodes', 10)))} node_id "
            "values from the tree. Prefer leaf page nodes. "
            'Return JSON only: {"thinking":"...","node_list":["node_id_1"]}.'
        )
        response = self.llm.complete(system=system, user=user, max_tokens=900)
        usage.add(response.usage)
        parsed = extract_json_object(response.text)
        selections = self._normalize_node_selections(doc_id, parsed)
        trace = [
            {
                "step": 1,
                "document_id": doc_id,
                "current_node": self._trace_node(self.trees[doc_id]),
                "candidate_child_count": len(_flatten_tree(self.trees[doc_id])) - 1,
                "candidate_children": [
                    self._trace_node(node)
                    for node in _flatten_tree(self.trees[doc_id])[1 : 1
                        + int(self.cfg.get("reasoning_max_trace_nodes", 250))]
                ],
                "prompt_tree_truncated": tree_text.endswith("\n...TRUNCATED..."),
                "raw_response": parsed,
                "selection_raw": parsed,
                "selection_source": "llm",
                "llm_selections": selections,
                "accepted_selections": selections,
            }
        ]
        return selections, {"raw": {"trace": trace, "selections": selections}, "usage": usage}

    def _normalize_node_selections(
        self, doc_id: str, parsed: dict[str, Any]
    ) -> list[dict[str, str]]:
        raw_items = (
            parsed.get("node_list")
            or parsed.get("nodes")
            or parsed.get("selections")
            or []
        )
        if not isinstance(raw_items, list):
            return []
        thinking = str(parsed.get("thinking") or parsed.get("reason") or "")
        selections: list[dict[str, str]] = []
        limit = max(1, int(self.cfg.get("selected_nodes", 10)))
        for item in raw_items:
            node_id = ""
            reason = thinking
            if isinstance(item, str):
                node_id = item
            elif isinstance(item, dict):
                node_id = str(item.get("node_id") or item.get("id") or "")
                reason = str(item.get("reason") or item.get("thinking") or thinking)
            node_id = self._resolve_node_id_alias(doc_id, node_id)
            if node_id and node_id in self.node_index:
                selections.append({"node_id": node_id, "reason": reason})
            if len(selections) >= limit:
                break
        return selections

    def _resolve_node_id_alias(self, doc_id: str, node_id: str) -> str:
        if node_id in self.node_index:
            return node_id
        for indexed_doc_id, node in self.node_index.values():
            if indexed_doc_id == doc_id and str(node.get("official_node_id", "")) == node_id:
                return str(node.get("node_id", ""))
        return node_id

    def _normalize_document_selections(
        self, selections: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        normalized = []
        for item in selections:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("document_id", ""))
            if doc_id not in self.trees:
                continue
            normalized.append(
                {"document_id": doc_id, "reason": str(item.get("reason", ""))}
            )
        return normalized

    def _format_document_catalog(self, *, max_chars: int) -> str:
        lines = []
        for doc_id, tree in sorted(self.trees.items()):
            summary = _preview(
                str(tree.get("summary", "")),
                int(self.cfg.get("max_document_summary_chars", 180)),
            )
            titles = "; ".join(_representative_titles(tree, limit=4))
            if titles:
                lines.append(f"- {doc_id}: {summary} | sections: {_preview(titles, 160)}")
            else:
                lines.append(f"- {doc_id}: {summary}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...TRUNCATED..."

    def _format_tree_for_selection(self, tree: dict[str, Any], *, max_chars: int) -> str:
        text = compact_json(_tree_for_prompt(tree))
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...TRUNCATED..."

    def _trace_node(self, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_id": str(node.get("node_id", "")),
            "title": str(node.get("title", "")),
            "summary": self._reasoning_preview(
                str(node.get("summary", "")),
                int(self.cfg.get("reasoning_max_node_summary_chars", 320)),
            ),
            "node_kind": str(node.get("node_kind", "")),
            "unit_start": node.get("unit_start"),
            "unit_end": node.get("unit_end"),
            "start_char": node.get("start_char"),
            "end_char": node.get("end_char"),
            "child_count": len(node.get("children", []) or []),
        }

    def _reasoning_preview(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "\n...TRUNCATED..."

    def _keyword_document_fallback(self, query: str) -> list[dict[str, str]]:
        terms = {term.lower() for term in query.split() if len(term) > 3}
        scored = []
        for doc_id, tree in self.trees.items():
            titles = " ".join(_representative_titles(tree, limit=6))
            haystack = f"{doc_id} {tree.get('title', '')} {tree.get('summary', '')} {titles}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, doc_id))
        scored.sort(reverse=True)
        return [
            {"document_id": doc_id, "reason": "keyword fallback"}
            for _score, doc_id in scored[
                : max(1, int(self.cfg.get("selected_documents", 3)))
            ]
        ]

    def _keyword_node_fallback(self, query: str, doc_id: str) -> list[dict[str, str]]:
        terms = {term.lower() for term in query.split() if len(term) > 3}
        leaves = [
            node
            for node in _flatten_tree(self.trees[doc_id])
            if not node.get("children")
        ]
        scored = []
        for node in leaves:
            haystack = (
                f"{node.get('title', '')} {node.get('summary', '')} "
                f"{node.get('section_title', '')}"
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, str(node["node_id"])))
        scored.sort(reverse=True)
        return [
            {"node_id": node_id, "reason": "keyword fallback"}
            for _score, node_id in scored[: max(1, int(self.cfg.get("selected_nodes", 10)))]
        ]

    def _index_tree(self, doc_id: str, tree: dict[str, Any]) -> None:
        for node in _flatten_tree(tree):
            self.node_index[node["node_id"]] = (doc_id, node)

    def _doc_hash(self, doc: Document) -> str:
        payload = {
            "document_id": doc.document_id,
            "length": len(doc.text),
            "sha1": hashlib.sha1(doc.text.encode("utf-8")).hexdigest(),
            "build_config": self._build_cache_config(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _build_cache_config(self) -> dict[str, Any]:
        return {
            "official_repo_url": self.cfg.get("repo_url"),
            "official_repo_ref": self.cfg.get("repo_ref"),
            "build_with_llm": bool(self.cfg.get("build_with_llm", True)),
            "virtual_page_target_tokens": self._virtual_page_target_tokens(),
            "virtual_page_max_tokens": self._virtual_page_max_tokens(),
            "if_thinning": bool(self.cfg.get("if_thinning", False)),
            "thinning_threshold": int(self.cfg.get("thinning_threshold", 5000)),
            "summary_token_threshold": int(self.cfg.get("summary_token_threshold", 200)),
            "if_add_node_summary": _yes_no(self.cfg.get("if_add_node_summary"), "yes"),
            "if_add_doc_description": _yes_no(
                self.cfg.get("if_add_doc_description"), "yes"
            ),
        }

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _virtual_page_target_tokens(self) -> int:
        return max(64, int(self.cfg.get("virtual_page_target_tokens", 900)))

    def _virtual_page_max_tokens(self) -> int:
        return max(
            self._virtual_page_target_tokens(),
            int(self.cfg.get("virtual_page_max_tokens", 1200)),
        )

    def _record_reasoning_trajectory(self) -> bool:
        return bool(self.cfg.get("record_reasoning_trajectory", True))


@contextmanager
def _patch_official_llm(
    utils_module: Any, llm: Any, usage: Usage
) -> Iterator[None]:
    old_completion = getattr(utils_module, "llm_completion", None)
    old_acompletion = getattr(utils_module, "llm_acompletion", None)

    def llm_completion(
        model: str | None,
        prompt: str,
        chat_history: list[dict[str, str]] | None = None,
        return_finish_reason: bool = False,
    ) -> str | tuple[str, str]:
        del model, chat_history
        if llm is None:
            text = ""
        else:
            response = llm.complete(
                system="You are building PageIndex document tree summaries.",
                user=prompt,
                max_tokens=700,
            )
            usage.add(response.usage)
            text = response.text
        if return_finish_reason:
            return text, "finished"
        return text

    async def llm_acompletion(model: str | None, prompt: str) -> str:
        value = llm_completion(model, prompt)
        if isinstance(value, tuple):
            return value[0]
        return value

    utils_module.llm_completion = llm_completion
    utils_module.llm_acompletion = llm_acompletion
    try:
        yield
    finally:
        if old_completion is not None:
            utils_module.llm_completion = old_completion
        if old_acompletion is not None:
            utils_module.llm_acompletion = old_acompletion


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _flatten_tree(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        out.extend(_flatten_tree(child))
    return out


def _tree_for_prompt(node: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "node_id": node.get("node_id", ""),
        "title": node.get("title", ""),
        "summary": node.get("summary", ""),
        "unit_start": node.get("unit_start"),
        "unit_end": node.get("unit_end"),
    }
    children = [_tree_for_prompt(child) for child in node.get("children", []) or []]
    if children:
        payload["nodes"] = children
    return payload


def _representative_titles(tree: dict[str, Any], *, limit: int) -> list[str]:
    titles = []
    for child in tree.get("children", []) or []:
        title = str(child.get("title", "")).strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            return titles
    for node in _flatten_tree(tree):
        if node.get("children"):
            continue
        title = str(node.get("title", "")).strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _clean_heading(text: str) -> str:
    text = " ".join(str(text).split())
    return text[:180] or "Untitled"


def _page_title(
    *, title: str, unit_index: int, part_index: int, section_parts: int
) -> str:
    if title:
        if section_parts > 1:
            return f"{title} / virtual page {part_index}"
        return f"{title} / virtual page"
    return f"Virtual page {unit_index}"


def _escape_markdown_headings(text: str) -> str:
    return "\n".join(
        re.sub(r"^(#{1,6})(\s+)", r"\\\1\2", line)
        for line in text.splitlines()
    )


def _yes_no(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    lowered = str(value).strip().lower()
    return "yes" if lowered in {"1", "true", "yes", "y"} else "no"
