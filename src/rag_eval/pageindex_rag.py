from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .json_utils import compact_json, extract_json_object
from .text_splitters import detect_sections
from .types import Document, RetrievedSpan, RetrievalOutput, Usage


_CHARS_PER_TOKEN = 4
_BUILD_CHECKPOINT_VERSION = 1
_PAGE_BREAK_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]
_TOC_ENTRY_RE = re.compile(
    r"""(?ix)^(
        table\ of\ contents
        |contents
        |(article|section|clause|schedule|exhibit|appendix)\s+[\w\dIVXLC().-]+
        |\d+(\.\d+)*[.)]?\s+[A-Z]
        |[A-Z][A-Z0-9 ,;:'"()/&-]{8,}
    )"""
)


class PageIndexRAG:
    """PageIndex-style RAG for LegalBench text documents.

    Raw text files do not have physical PDF pages, so this implementation creates
    contiguous "virtual pages" under token budgets, semanticizes every virtual page
    with the LLM, builds a shallow ToC tree from those pages, and then traverses the
    tree level by level at query time until it selects page nodes as retrieval
    context.
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
        for doc in tqdm(
            documents,
            desc="PageIndex documents",
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
        selected_nodes = max(1, int(self.cfg.get("selected_nodes", 5)))
        error_parts: list[str] = []
        record_reasoning = self._record_reasoning_trajectory()
        reasoning_trajectory: dict[str, Any] | None = None
        if record_reasoning:
            reasoning_trajectory = {
                "query": query,
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

        document_selections = []
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
                f"pageindex_document_selection_error: {type(exc).__name__}: {exc}"
            )
            error_parts.append(error_text)
            if reasoning_trajectory is not None:
                reasoning_trajectory["document_selection"].update(
                    {
                        "source": "keyword_fallback",
                        "error": error_text,
                        "requested_selections": document_selections,
                    }
                )
            metadata["document_selection_raw"] = {
                "documents": document_selections,
                "fallback": True,
            }

        if not document_selections:
            document_selections = self._keyword_document_fallback(query)
            if document_selections:
                error_parts.append("pageindex_document_selection_empty")
                metadata["document_selection_fallback"] = document_selections
                if reasoning_trajectory is not None:
                    reasoning_trajectory["document_selection"].update(
                        {
                            "source": "keyword_fallback",
                            "fallback_reason": "llm_returned_no_documents",
                            "fallback_selections": document_selections,
                        }
                    )

        normalized_document_selections = []
        for item in document_selections:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("document_id", ""))
            if doc_id not in self.trees:
                continue
            normalized_document_selections.append(
                {"document_id": doc_id, "reason": str(item.get("reason", ""))}
            )
        if not normalized_document_selections:
            fallback_documents = self._keyword_document_fallback(query)
            for item in fallback_documents:
                doc_id = str(item.get("document_id", ""))
                if doc_id not in self.trees:
                    continue
                normalized_document_selections.append(
                    {"document_id": doc_id, "reason": str(item.get("reason", ""))}
                )
            if normalized_document_selections:
                error_parts.append("pageindex_document_selection_invalid")
                metadata["document_selection_fallback"] = fallback_documents
                if reasoning_trajectory is not None:
                    reasoning_trajectory["document_selection"].update(
                        {
                            "source": "keyword_fallback",
                            "fallback_reason": "llm_selected_unknown_documents",
                            "fallback_selections": fallback_documents,
                        }
                    )
        metadata["document_selections"] = normalized_document_selections
        if reasoning_trajectory is not None:
            reasoning_trajectory["document_selection"][
                "accepted_selections"
            ] = normalized_document_selections

        spans: list[RetrievedSpan] = []
        node_metadata: dict[str, Any] = {}
        seen_node_ids: set[str] = set()
        rank = 0
        for document_item in normalized_document_selections[
            : max(1, int(self.cfg.get("selected_documents", 3)))
        ]:
            doc_id = document_item["document_id"]
            page_selections = []
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
                page_selections, parsed = self._select_nodes_for_document(query, doc_id)
                usage.add(parsed["usage"])
                node_metadata[doc_id] = parsed["raw"]
                if walk_record is not None:
                    walk_record.update(
                        {
                            "source": "llm_tree_walk",
                            "steps": parsed["raw"].get("trace", []),
                            "final_selections": page_selections,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                page_selections = self._keyword_node_fallback(query, doc_id)
                error_text = (
                    f"pageindex_node_selection_error[{doc_id}]: "
                    f"{type(exc).__name__}: {exc}"
                )
                error_parts.append(error_text)
                node_metadata[doc_id] = {
                    "selections": page_selections,
                    "fallback": True,
                }
                if walk_record is not None:
                    walk_record.update(
                        {
                            "source": "keyword_fallback",
                            "error": error_text,
                            "final_selections": page_selections,
                        }
                    )

            if not page_selections:
                page_selections = self._keyword_node_fallback(query, doc_id)
                if page_selections:
                    error_parts.append(f"pageindex_node_selection_empty[{doc_id}]")
                    node_metadata[f"{doc_id}__fallback"] = page_selections
                    if walk_record is not None:
                        walk_record.update(
                            {
                                "source": "keyword_fallback",
                                "fallback_reason": "llm_returned_no_nodes",
                                "final_selections": page_selections,
                            }
                        )
            elif walk_record is not None:
                walk_record["final_selections"] = page_selections

            if walk_record is not None:
                reasoning_trajectory["document_walks"].append(walk_record)

            for item in page_selections:
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
                            "retriever": "pageindex",
                            "document_reason": document_item.get("reason", ""),
                            "node_id": node_id,
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
        checkpoint_path = self._checkpoint_path(cache_path)
        if cache_path.exists() and not self.cfg.get("force_reindex", False):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f), Usage()
        if self.cfg.get("force_reindex", False) and checkpoint_path.exists():
            checkpoint_path.unlink()

        tree, usage = self._build_tree(doc)
        self._write_json_atomic(cache_path, tree)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return tree, usage

    def _build_tree(self, doc: Document) -> tuple[dict[str, Any], Usage]:
        pages = self._build_virtual_pages(doc)
        toc_entries = self._extract_toc_entries(pages)
        slug = hashlib.sha1(doc.document_id.encode("utf-8")).hexdigest()[:8]
        checkpoint = self._load_build_checkpoint(doc)
        usage = checkpoint["usage"] if checkpoint is not None else Usage()

        page_nodes = list(checkpoint["page_nodes"]) if checkpoint is not None else []
        pages_completed = len(page_nodes)
        for page in tqdm(
            pages[pages_completed:],
            desc=f"PageIndex pages [{doc.document_id}]",
            unit="page",
            leave=False,
            initial=pages_completed,
            total=len(pages),
            disable=len(pages) <= 1,
        ):
            node = self._make_page_node(slug, page)
            if self.cfg.get("build_with_llm", True) and self.llm is not None:
                try:
                    node, node_usage = self._semanticize_page_node(doc, node, page, toc_entries)
                    usage.add(node_usage)
                except Exception as exc:  # noqa: BLE001
                    node.setdefault("metadata", {})["semanticize_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            page_nodes.append(node)
            self._write_build_checkpoint(
                doc,
                usage=usage,
                page_nodes=page_nodes,
                top_children=[],
            )

        top_children = (
            list(checkpoint["top_children"]) if checkpoint is not None else []
        )
        parent_batches = self._batch_page_nodes(page_nodes)
        spans_completed = len(top_children)
        for batch_idx, batch in enumerate(
            tqdm(
                parent_batches[spans_completed:],
                desc=f"PageIndex spans [{doc.document_id}]",
                unit="span",
                leave=False,
                initial=spans_completed,
                total=len(parent_batches),
                disable=len(parent_batches) <= 1,
            ),
            start=spans_completed,
        ):
            if len(batch) == 1:
                top_children.append(batch[0])
                self._write_build_checkpoint(
                    doc,
                    usage=usage,
                    page_nodes=page_nodes,
                    top_children=top_children,
                )
                continue
            node = self._make_parent_node(
                slug=slug,
                suffix=f"s{batch_idx:04d}",
                children=batch,
                node_kind="span",
            )
            if self.cfg.get("build_with_llm", True) and self.llm is not None:
                try:
                    node, node_usage = self._semanticize_parent_node(
                        doc,
                        node,
                        batch,
                        toc_entries=toc_entries,
                        is_root=False,
                    )
                    usage.add(node_usage)
                except Exception as exc:  # noqa: BLE001
                    node.setdefault("metadata", {})["semanticize_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            top_children.append(node)
            self._write_build_checkpoint(
                doc,
                usage=usage,
                page_nodes=page_nodes,
                top_children=top_children,
            )

        root = {
            "node_id": f"{slug}-root",
            "title": doc.document_id,
            "summary": self._heuristic_root_summary(top_children, doc.text),
            "start_char": 0,
            "end_char": len(doc.text),
            "unit_start": page_nodes[0]["unit_start"] if page_nodes else 0,
            "unit_end": page_nodes[-1]["unit_end"] if page_nodes else 0,
            "token_count": _estimate_tokens(doc.text),
            "node_kind": "root",
            "section_title": "",
            "children": top_children,
            "metadata": {
                "source": "virtual_page_pageindex",
                "unit_type": "virtual_page",
                "virtual_page_target_tokens": self._virtual_page_target_tokens(),
                "virtual_page_max_tokens": self._virtual_page_max_tokens(),
                "toc_check_units": self._toc_check_units(),
                "max_units_per_node": self._max_units_per_node(),
                "max_tokens_per_node": self._max_tokens_per_node(),
                "toc_entries": toc_entries,
            },
        }
        if self.cfg.get("build_with_llm", True) and self.llm is not None and top_children:
            try:
                root, root_usage = self._semanticize_parent_node(
                    doc,
                    root,
                    top_children,
                    toc_entries=toc_entries,
                    is_root=True,
                )
                usage.add(root_usage)
                root["metadata"]["semanticized_by_llm"] = True
            except Exception as exc:  # noqa: BLE001
                root["metadata"]["build_error"] = f"{type(exc).__name__}: {exc}"
        return root, usage

    def _checkpoint_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(".partial.json")

    def _load_build_checkpoint(self, doc: Document) -> dict[str, Any] | None:
        cache_path = self.cache_dir / f"{self._doc_hash(doc)}.json"
        checkpoint_path = self._checkpoint_path(cache_path)
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:  # noqa: BLE001
            return None

        if (
            int(checkpoint.get("checkpoint_version", 0)) != _BUILD_CHECKPOINT_VERSION
            or str(checkpoint.get("doc_hash", "")) != self._doc_hash(doc)
            or str(checkpoint.get("document_id", "")) != doc.document_id
        ):
            return None

        usage_data = checkpoint.get("usage", {})
        page_nodes = checkpoint.get("page_nodes", [])
        top_children = checkpoint.get("top_children", [])
        if not isinstance(page_nodes, list) or not isinstance(top_children, list):
            return None

        return {
            "usage": Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                cache_read_input_tokens=int(
                    usage_data.get("cache_read_input_tokens", 0) or 0
                ),
                cache_creation_input_tokens=int(
                    usage_data.get("cache_creation_input_tokens", 0) or 0
                ),
                estimated_cost_usd=float(
                    usage_data.get("estimated_cost_usd", 0.0) or 0.0
                ),
            ),
            "page_nodes": page_nodes,
            "top_children": top_children,
        }

    def _write_build_checkpoint(
        self,
        doc: Document,
        *,
        usage: Usage,
        page_nodes: list[dict[str, Any]],
        top_children: list[dict[str, Any]],
    ) -> None:
        cache_path = self.cache_dir / f"{self._doc_hash(doc)}.json"
        checkpoint_path = self._checkpoint_path(cache_path)
        payload = {
            "checkpoint_version": _BUILD_CHECKPOINT_VERSION,
            "doc_hash": self._doc_hash(doc),
            "document_id": doc.document_id,
            "usage": usage.to_dict(),
            "page_nodes": page_nodes,
            "top_children": top_children,
        }
        self._write_json_atomic(checkpoint_path, payload)

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _build_virtual_pages(self, doc: Document) -> list[dict[str, Any]]:
        target_chars = self._virtual_page_target_tokens() * _CHARS_PER_TOKEN
        max_chars = self._virtual_page_max_tokens() * _CHARS_PER_TOKEN
        pages: list[dict[str, Any]] = []
        unit_index = 1
        for section_index, (title, start, end) in enumerate(detect_sections(doc.text)):
            section_text = doc.text[start:end]
            local_spans = _pack_virtual_page_spans(
                section_text,
                target_chars=target_chars,
                max_chars=max_chars,
            )
            for part_index, (local_start, local_end) in enumerate(local_spans, start=1):
                abs_start = start + local_start
                abs_end = start + local_end
                text = doc.text[abs_start:abs_end]
                pages.append(
                    {
                        "unit_index": unit_index,
                        "section_index": section_index,
                        "section_title": title if title != "Document start" else "",
                        "part_index": part_index,
                        "section_parts": len(local_spans),
                        "start_char": abs_start,
                        "end_char": abs_end,
                        "text": text,
                        "token_count": _estimate_tokens(text),
                    }
                )
                unit_index += 1
        return pages

    def _make_page_node(self, slug: str, page: dict[str, Any]) -> dict[str, Any]:
        page_number = int(page["unit_index"])
        title = self._heuristic_page_title(page)
        return {
            "node_id": f"{slug}-u{page_number:04d}",
            "title": title,
            "summary": _preview(str(page["text"]), 320),
            "start_char": int(page["start_char"]),
            "end_char": int(page["end_char"]),
            "unit_start": page_number,
            "unit_end": page_number,
            "token_count": int(page["token_count"]),
            "node_kind": "page",
            "section_title": str(page.get("section_title", "")),
            "children": [],
        }

    def _make_parent_node(
        self,
        *,
        slug: str,
        suffix: str,
        children: list[dict[str, Any]],
        node_kind: str,
    ) -> dict[str, Any]:
        first = children[0]
        last = children[-1]
        title = self._heuristic_parent_title(children)
        return {
            "node_id": f"{slug}-{suffix}",
            "title": title,
            "summary": self._heuristic_parent_summary(children),
            "start_char": int(first["start_char"]),
            "end_char": int(last["end_char"]),
            "unit_start": int(first["unit_start"]),
            "unit_end": int(last["unit_end"]),
            "token_count": sum(int(child.get("token_count", 0)) for child in children),
            "node_kind": node_kind,
            "section_title": self._common_section_title(children),
            "children": children,
        }

    def _semanticize_page_node(
        self,
        doc: Document,
        node: dict[str, Any],
        page: dict[str, Any],
        toc_entries: list[str],
    ) -> tuple[dict[str, Any], Usage]:
        system = (
            "You build semantic table-of-contents nodes for legal documents. "
            "Read the full virtual page text and return a short title and concise summary."
        )
        toc_hint = "\n".join(f"- {entry}" for entry in toc_entries[:12])
        user = (
            f"Document id: {doc.document_id}\n"
            f"Node id: {node['node_id']}\n"
            f"Virtual page: {node['unit_start']}\n"
            f"Heuristic heading: {page.get('section_title', '') or node['title']}\n"
            f"Existing ToC hints:\n{toc_hint or '(none)'}\n\n"
            f"Text:\n{page['text']}\n\n"
            'Return JSON only: {"title":"...","summary":"..."}'
        )
        response = self.llm.complete(
            system=system,
            user=user,
            max_tokens=self._node_summary_max_tokens(),
        )
        parsed = extract_json_object(response.text)
        self._apply_node_semantics(node, parsed)
        node.setdefault("metadata", {})["semanticized_by_llm"] = True
        return node, response.usage

    def _semanticize_parent_node(
        self,
        doc: Document,
        node: dict[str, Any],
        children: list[dict[str, Any]],
        *,
        toc_entries: list[str],
        is_root: bool,
    ) -> tuple[dict[str, Any], Usage]:
        outline = [
            {
                "node_id": child["node_id"],
                "units": [child["unit_start"], child["unit_end"]],
                "title": child.get("title", ""),
                "summary": child.get("summary", ""),
            }
            for child in children
        ]
        system = (
            "You build semantic table-of-contents nodes for legal documents. "
            "Given child node summaries, produce a short title and concise summary "
            "for their parent span."
        )
        toc_hint = "\n".join(f"- {entry}" for entry in toc_entries[:12])
        user = (
            f"Document id: {doc.document_id}\n"
            f"Node id: {node['node_id']}\n"
            f"Virtual page range: {node['unit_start']}-{node['unit_end']}\n"
            f"Node kind: {node['node_kind']}\n"
            f"Heuristic title: {node.get('title', '')}\n"
            f"Existing ToC hints:\n{toc_hint or '(none)'}\n\n"
            f"Child nodes:\n{compact_json(outline)}\n\n"
            'Return JSON only: {"title":"...","summary":"..."}'
        )
        max_tokens = self._root_summary_max_tokens() if is_root else self._node_summary_max_tokens()
        response = self.llm.complete(system=system, user=user, max_tokens=max_tokens)
        parsed = extract_json_object(response.text)
        self._apply_node_semantics(node, parsed)
        node.setdefault("metadata", {})["semanticized_by_llm"] = True
        return node, response.usage

    def _apply_node_semantics(self, node: dict[str, Any], parsed: dict[str, Any]) -> None:
        title = str(parsed.get("title", "")).strip()
        summary = str(parsed.get("summary", "")).strip()
        if title:
            node["title"] = title[:180]
        if summary:
            node["summary"] = summary[:800]

    def _batch_page_nodes(self, page_nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        max_units = self._max_units_per_node()
        max_tokens = self._max_tokens_per_node()
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_tokens = 0
        current_section = ""
        for node in page_nodes:
            node_tokens = int(node.get("token_count", 0))
            node_section = str(node.get("section_title", ""))
            should_split = bool(
                current
                and (
                    len(current) >= max_units
                    or current_tokens + node_tokens > max_tokens
                    or (
                        node_section != current_section
                        and current_section
                        and node_section
                    )
                )
            )
            if should_split:
                batches.append(current)
                current = []
                current_tokens = 0
                current_section = ""
            current.append(node)
            current_tokens += node_tokens
            current_section = current_section or node_section
        if current:
            batches.append(current)
        return batches

    def _extract_toc_entries(self, pages: list[dict[str, Any]]) -> list[str]:
        max_units = min(len(pages), self._toc_check_units())
        prefix_text = "\n".join(page["text"] for page in pages[:max_units])
        prefix_text = prefix_text[: int(self.cfg.get("max_toc_scan_chars", 40000))]
        lines = [line.strip() for line in prefix_text.splitlines() if line.strip()]
        toc_start = None
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if lowered == "table of contents" or lowered == "contents":
                toc_start = idx + 1
                break
        entries: list[str] = []
        if toc_start is not None:
            for line in lines[toc_start:]:
                if _TOC_ENTRY_RE.match(line) or "...." in line:
                    entries.append(line[:140])
                elif entries:
                    break
        if entries:
            return entries[:20]
        section_titles = [
            title
            for title, _start, _end in detect_sections(prefix_text)
            if title and title != "Document start"
        ]
        seen: list[str] = []
        for title in section_titles:
            if title not in seen:
                seen.append(title)
            if len(seen) >= 12:
                break
        return seen

    def _heuristic_page_title(self, page: dict[str, Any]) -> str:
        section_title = str(page.get("section_title", "")).strip()
        unit_index = int(page["unit_index"])
        if section_title:
            if int(page.get("section_parts", 1)) > 1:
                return f"{section_title} / part {int(page['part_index'])}"
            return section_title
        return f"Virtual page {unit_index}"

    def _heuristic_parent_title(self, children: list[dict[str, Any]]) -> str:
        section_title = self._common_section_title(children)
        if section_title:
            return section_title
        return f"Virtual pages {children[0]['unit_start']}-{children[-1]['unit_end']}"

    def _heuristic_parent_summary(self, children: list[dict[str, Any]]) -> str:
        summaries = []
        for child in children[:4]:
            title = str(child.get("title", "")).strip()
            summary = str(child.get("summary", "")).strip()
            if title and summary:
                summaries.append(f"{title}: {summary}")
            elif title:
                summaries.append(title)
        return _preview(" ".join(summaries), 500)

    def _heuristic_root_summary(self, children: list[dict[str, Any]], text: str) -> str:
        if children:
            return self._heuristic_parent_summary(children)
        return _preview(text, 600)

    def _common_section_title(self, children: list[dict[str, Any]]) -> str:
        titles = [str(child.get("section_title", "")).strip() for child in children]
        titles = [title for title in titles if title]
        if len(titles) == len(children) and len(set(titles)) == 1:
            return titles[0]
        return ""

    def _index_tree(self, doc_id: str, tree: dict[str, Any]) -> None:
        for node in _flatten_tree(tree):
            self.node_index[node["node_id"]] = (doc_id, node)

    def _select_documents(self, query: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
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
        usage = Usage()
        trace: list[dict[str, Any]] = []
        tree = self.trees[doc_id]
        selections = self._traverse_node_selection(
            query=query,
            doc_id=doc_id,
            node=tree,
            usage=usage,
            trace=trace,
            budget=max(1, int(self.cfg.get("selected_nodes", 5))),
        )
        return selections, {"raw": {"trace": trace, "selections": selections}, "usage": usage}

    def _traverse_node_selection(
        self,
        *,
        query: str,
        doc_id: str,
        node: dict[str, Any],
        usage: Usage,
        trace: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, str]]:
        if budget <= 0:
            return []
        children = list(node.get("children", []) or [])
        if not children:
            return [{"node_id": str(node["node_id"]), "reason": "selected page node"}]

        trace_start = len(trace)
        raw_selections = self._select_child_nodes(
            query=query,
            doc_id=doc_id,
            node=node,
            children=children,
            usage=usage,
            trace=trace,
            budget=budget,
        )
        trace_entry = trace[-1] if len(trace) > trace_start else None
        child_by_id = {str(child["node_id"]): child for child in children}
        normalized = []
        invalid_selection_ids = []
        for item in raw_selections:
            if not isinstance(item, dict):
                continue
            child = child_by_id.get(str(item.get("node_id", "")))
            if child is None:
                invalid_selection_ids.append(str(item.get("node_id", "")))
                continue
            normalized.append(
                {"node_id": str(child["node_id"]), "reason": str(item.get("reason", ""))}
            )
        if not normalized:
            normalized = self._keyword_child_fallback(query, children, budget)
            if trace_entry is not None:
                trace_entry["selection_source"] = "keyword_fallback"
                trace_entry["fallback_reason"] = "llm_selected_no_valid_children"
                trace_entry["accepted_selections"] = normalized
        elif trace_entry is not None:
            trace_entry["selection_source"] = "llm"
            trace_entry["accepted_selections"] = normalized
        if trace_entry is not None and invalid_selection_ids:
            trace_entry["invalid_selection_ids"] = invalid_selection_ids

        results: list[dict[str, str]] = []
        for item in normalized:
            if len(results) >= budget:
                break
            child = child_by_id[item["node_id"]]
            if child.get("children"):
                nested = self._traverse_node_selection(
                    query=query,
                    doc_id=doc_id,
                    node=child,
                    usage=usage,
                    trace=trace,
                    budget=budget - len(results),
                )
                if nested:
                    results.extend(nested[: budget - len(results)])
                    continue
            results.append(item)
        return results[:budget]

    def _select_child_nodes(
        self,
        *,
        query: str,
        doc_id: str,
        node: dict[str, Any],
        children: list[dict[str, Any]],
        usage: Usage,
        trace: list[dict[str, Any]],
        budget: int,
    ) -> list[dict[str, str]]:
        child_text = self._format_children_for_selection(
            children,
            max_chars=int(self.cfg.get("max_tree_chars", 24000)),
        )
        system = (
            "You are a PageIndex retrieval agent. Select the most relevant child nodes "
            "for this legal query. Prefer the smallest child nodes that are still likely "
            "to contain the answer."
        )
        user = (
            f"Query:\n{query}\n\n"
            f"Document id: {doc_id}\n"
            f"Current node: {node.get('title', '')}\n"
            f"Current node summary: {node.get('summary', '')}\n"
            f"Virtual page range: {node.get('unit_start', 0)}-{node.get('unit_end', 0)}\n\n"
            f"Child nodes:\n{child_text}\n\n"
            f"Select up to {max(1, budget)} child node_id values from this list only. "
            'Return JSON only: {"selections":[{"node_id":"...","reason":"..."}]}.'
        )
        response = self.llm.complete(system=system, user=user, max_tokens=700)
        usage.add(response.usage)
        parsed = extract_json_object(response.text)
        selections = parsed.get("selections", [])
        if not isinstance(selections, list):
            selections = []
        trace.append(
            {
                "step": len(trace) + 1,
                "document_id": doc_id,
                "current_node": self._trace_node(node),
                "candidate_child_count": len(children),
                "candidate_children": [
                    self._trace_node(child) for child in children
                ],
                "prompt_child_list_truncated": child_text.endswith(
                    "\n...TRUNCATED..."
                ),
                "raw_response": parsed,
                "selection_raw": parsed,
                "llm_selections": selections,
            }
        )
        return selections

    def _format_children_for_selection(
        self, children: list[dict[str, Any]], *, max_chars: int
    ) -> str:
        lines = []
        for child in children:
            lines.append(
                f"- {child['node_id']} [{_unit_label(child)}] "
                f"{child.get('title', '')}: {child.get('summary', '')}"
            )
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...TRUNCATED..."

    def _format_document_catalog(self, *, max_chars: int) -> str:
        lines = []
        for doc_id, tree in sorted(self.trees.items()):
            summary = _preview(
                str(tree.get("summary", "")),
                int(self.cfg.get("max_document_summary_chars", 180)),
            )
            section_titles = "; ".join(_representative_titles(tree, limit=4))
            if section_titles:
                section_titles = _preview(
                    section_titles,
                    int(self.cfg.get("max_document_section_chars", 120)),
                )
                lines.append(f"- {doc_id}: {summary} | sections: {section_titles}")
            else:
                lines.append(f"- {doc_id}: {summary}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...TRUNCATED..."

    def _format_document_tree(
        self, doc_id: str, tree: dict[str, Any], *, max_chars: int
    ) -> str:
        lines = [f"DOCUMENT {doc_id}"]
        _format_node(tree, lines, depth=0)
        text = "\n".join(lines)
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
            haystack = (
                f"{doc_id} {tree.get('title', '')} {tree.get('summary', '')} {titles}"
            ).lower()
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
        leaves = [
            node
            for node in _flatten_tree(self.trees[doc_id])
            if not node.get("children")
        ]
        return self._keyword_child_fallback(
            query,
            leaves,
            max(1, int(self.cfg.get("selected_nodes", 5))),
        )

    def _keyword_child_fallback(
        self, query: str, children: list[dict[str, Any]], limit: int
    ) -> list[dict[str, str]]:
        terms = {term.lower() for term in query.split() if len(term) > 3}
        scored = []
        for child in children:
            haystack = (
                f"{child.get('title', '')} {child.get('summary', '')} "
                f"{child.get('section_title', '')}"
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, str(child["node_id"])))
        scored.sort(reverse=True)
        return [
            {"node_id": node_id, "reason": "keyword fallback"}
            for _score, node_id in scored[:limit]
        ]

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
            "build_with_llm": bool(self.cfg.get("build_with_llm", True)),
            "virtual_page_target_tokens": self._virtual_page_target_tokens(),
            "virtual_page_max_tokens": self._virtual_page_max_tokens(),
            "toc_check_units": self._toc_check_units(),
            "max_units_per_node": self._max_units_per_node(),
            "max_tokens_per_node": self._max_tokens_per_node(),
            "node_summary_max_tokens": self._node_summary_max_tokens(),
            "root_summary_max_tokens": self._root_summary_max_tokens(),
        }

    def _virtual_page_target_tokens(self) -> int:
        if self.cfg.get("virtual_page_target_tokens") is not None:
            return max(64, int(self.cfg["virtual_page_target_tokens"]))
        legacy_chars = max(400, int(self.cfg.get("max_leaf_chars", 3600)))
        return max(64, int((legacy_chars / _CHARS_PER_TOKEN) * 0.8))

    def _virtual_page_max_tokens(self) -> int:
        if self.cfg.get("virtual_page_max_tokens") is not None:
            return max(
                self._virtual_page_target_tokens(),
                int(self.cfg["virtual_page_max_tokens"]),
            )
        legacy_chars = max(400, int(self.cfg.get("max_leaf_chars", 3600)))
        return max(
            self._virtual_page_target_tokens(),
            int(legacy_chars / _CHARS_PER_TOKEN),
        )

    def _toc_check_units(self) -> int:
        return max(1, int(self.cfg.get("toc_check_units", 20)))

    def _max_units_per_node(self) -> int:
        return max(1, int(self.cfg.get("max_units_per_node", self.cfg.get("group_size", 10))))

    def _max_tokens_per_node(self) -> int:
        return max(
            self._virtual_page_max_tokens(),
            int(self.cfg.get("max_tokens_per_node", 20000)),
        )

    def _node_summary_max_tokens(self) -> int:
        return max(96, int(self.cfg.get("node_summary_max_tokens", 220)))

    def _root_summary_max_tokens(self) -> int:
        return max(self._node_summary_max_tokens(), int(self.cfg.get("root_summary_max_tokens", 260)))

    def _record_reasoning_trajectory(self) -> bool:
        return bool(self.cfg.get("record_reasoning_trajectory", True))


def _flatten_tree(node: dict[str, Any]) -> list[dict[str, Any]]:
    out = [node]
    for child in node.get("children", []) or []:
        out.extend(_flatten_tree(child))
    return out


def _format_node(node: dict[str, Any], lines: list[str], *, depth: int) -> None:
    indent = "  " * depth
    lines.append(
        f"{indent}- {node['node_id']} [{node['start_char']}:{node['end_char']}] "
        f"[{_unit_label(node)}] {node.get('title', '')}: {node.get('summary', '')}"
    )
    for child in node.get("children", []) or []:
        _format_node(child, lines, depth=depth + 1)


def _preview(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _unit_label(node: dict[str, Any]) -> str:
    start = int(node.get("unit_start", 0))
    end = int(node.get("unit_end", 0))
    if start <= 0:
        return "u?"
    if start == end:
        return f"u{start}"
    return f"u{start}-u{end}"


def _representative_titles(tree: dict[str, Any], *, limit: int) -> list[str]:
    titles = []
    for child in tree.get("children", []) or []:
        title = str(child.get("title", "")).strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            return titles
    return _leaf_titles(tree, limit=limit)


def _leaf_titles(tree: dict[str, Any], *, limit: int) -> list[str]:
    titles = []
    for node in _flatten_tree(tree):
        if node.get("children"):
            continue
        title = str(node.get("title", "")).strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _pack_virtual_page_spans(
    text: str, *, target_chars: int, max_chars: int
) -> list[tuple[int, int]]:
    if not text:
        return []
    target_chars = max(256, target_chars)
    max_chars = max(target_chars, max_chars)
    spans: list[tuple[int, int]] = []
    start = 0
    min_break = max(64, int(target_chars * 0.6))
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        preferred_end = min(len(text), start + target_chars)
        if hard_end == len(text):
            end = len(text)
        else:
            end = _find_best_break(text, start, min_break, preferred_end)
            if end <= start:
                end = _find_best_break(text, start, min_break, hard_end)
            if end <= start:
                end = hard_end
        spans.append((start, end))
        start = end
    return spans


def _find_best_break(text: str, start: int, min_break: int, search_end: int) -> int:
    if search_end <= start:
        return start
    left = min(len(text), start + min_break)
    best = -1
    for sep in _PAGE_BREAK_SEPARATORS:
        pos = text.rfind(sep, left, search_end)
        if pos > best:
            best = pos + len(sep)
    return best if best > start else start
