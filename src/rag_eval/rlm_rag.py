from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .json_utils import extract_json_object
from .types import Document, RetrievedSpan, RetrievalOutput, Usage
from .vector_rag import VectorRAG


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "any",
    "are",
    "between",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "must",
    "not",
    "of",
    "on",
    "or",
    "shall",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
}

_CITATION_SCHEMA_INSTRUCTION = """
Return your final answer as a single JSON object (RFC 8259) with this exact
shape and nothing else (you may wrap it in a ```json fenced block):

{
  "answer": "<short natural-language answer to the query>",
  "spans": [
    {
      "document_id": "<filename as it appears in the corpus>",
      "start_char": <integer offset into the document>,
      "end_char": <integer offset, exclusive>,
      "snippet": "<exact substring document[start_char:end_char]>"
    }
  ]
}

CRITICAL - JSON, not Python: keys and string values must use double quotes.
Do NOT emit Python dict syntax with single quotes.

CRITICAL - offsets: start_char and end_char must be the actual character
offsets such that document[start_char:end_char] == snippet. Offsets that
don't match the text will be corrected from `snippet` when possible and
discarded otherwise.
""".strip()

_BASE_ROOT_PROMPT = f"""\
You are answering a legal-document retrieval query.

The variable `context` in your Python REPL holds:
- context["query"]: the question
- context["corpus"]: a dict mapping {{document_id: full_text}}

Find the spans of text that answer the query. You may use Python freely:
substring search, regex, slicing, and recursive sub-LM calls on candidate
documents. Be efficient - do not print large slices to stdout; manipulate the
data through code.

You also have a helper `make_span(document_id, snippet)` available in the
global namespace. It locates `snippet` inside the named corpus document and
returns a fully-formed span dict with correct character offsets:

    span = make_span("doc.txt", "exact substring of the contract")
    # -> {{"document_id": "doc.txt", "start_char": 1234,
    #     "end_char": 1280, "snippet": "exact substring of the contract"}}

Always use `make_span` to construct each span in your final answer instead of
computing start_char / end_char by hand. The harness scoring is strict about
offsets - a one-character drift discards the citation. `make_span` guarantees
the offsets match the snippet.

Optional helper tools are also available for browsing and reading document
windows, but you can directly inspect `context["corpus"]` in Python when that
is more efficient.

{_CITATION_SCHEMA_INSTRUCTION}
"""

_PAGEINDEX_ROOT_PROMPT = f"""\
You are answering a legal-document retrieval query with an agentic PageIndex workflow.

The variable `context` in your Python REPL holds:
- context["query"]: the question
- context["corpus"]: a dict mapping {{document_id: full_text}}

You also have PageIndex cache tools that expose document structure already built in `.cache/pageindex`.
Use them first to navigate efficiently:
- `list_pageindex_documents(...)` to inspect available structured documents
- `search_pageindex_nodes(...)` to find relevant semantic nodes by title/summary/section
- `list_pageindex_children(...)` to walk down the tree from root nodes into narrower children
- `read_pageindex_node(...)` to inspect a node and its exact source span
- `make_span_from_pageindex_node(...)` when a whole cached node is the evidence span

Search policy:
- Start with PageIndex structure before broad raw-text searching.
- Prefer the smallest relevant node or set of nodes that answers the query.
- After locating strong candidate nodes, verify the exact text with `read_pageindex_node(...)` or direct Python access.
- Use `search_documents(...)`, regex, or direct corpus inspection only as fallback or to refine text inside a chosen document.
- Keep searching for exceptions, carve-outs, definitions, and follow-on clauses before finalizing.
- Return multiple spans when the answer is distributed across clauses or modified elsewhere.

You also have a helper `make_span(document_id, snippet)` available in the
global namespace. It locates `snippet` inside the named corpus document and
returns a fully-formed span dict with correct character offsets:

    span = make_span("doc.txt", "exact substring of the contract")
    # -> {{"document_id": "doc.txt", "start_char": 1234,
    #     "end_char": 1280, "snippet": "exact substring of the contract"}}

Always use `make_span` or `make_span_from_pageindex_node` to construct each
span in your final answer instead of computing offsets by hand.

{_CITATION_SCHEMA_INSTRUCTION}
"""

_RESULT_KEYS = {"answer", "spans", "retrieved_spans", "documents", "result_json"}


class RLMRAG:
    """Official RLM-backed retriever for LegalBench-RAG documents.

    The official RLM package runs a REPL-oriented model loop. This adapter keeps
    the LegalBench contract local: RLM receives direct REPL access to the full
    `{document_id: text}` corpus, plus helper tools for browsing/searching and a
    `make_span()` utility for exact offsets. The final response is parsed back
    into exact source-document character spans.
    """

    name = "rlm"

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        llm_cfg: dict[str, Any] | None = None,
        method_name: str | None = None,
        vector_tool_cfg: dict[str, Any] | None = None,
        vector_tool_cache_dir: Path | None = None,
        pageindex_tool_cfg: dict[str, Any] | None = None,
        pageindex_tool_cache_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.llm_cfg = llm_cfg or {}
        self.method_name = str(method_name or cfg.get("method_name") or self.name)
        self.documents: dict[str, Document] = {}
        self.document_catalog: list[dict[str, Any]] = []
        self.vector_tool_cfg = dict(vector_tool_cfg or {})
        self.vector_tool_cache_dir = vector_tool_cache_dir
        self.vector_tool: VectorRAG | None = None
        self.pageindex_tool_cfg = dict(pageindex_tool_cfg or {})
        self.pageindex_tool_cache_dir = pageindex_tool_cache_dir
        self.pageindex_documents: dict[str, dict[str, Any]] = {}
        self.pageindex_document_catalog: list[dict[str, Any]] = []
        self.pageindex_node_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    def build(self, documents: list[Document]) -> None:
        self.documents = {doc.document_id: doc for doc in documents}
        preview_chars = max(0, int(self.cfg.get("catalog_preview_chars", 240)))
        self.document_catalog = [
            {
                "document_id": doc.document_id,
                "length": len(doc.text),
                "preview": doc.text[:preview_chars],
            }
            for doc in documents
        ]
        self.vector_tool = None
        if self._vector_tool_enabled():
            self.vector_tool = self._build_vector_tool(documents)
        self.pageindex_documents = {}
        self.pageindex_document_catalog = []
        self.pageindex_node_lookup = {}
        if self._pageindex_tool_enabled():
            self._load_pageindex_tool(documents)

    def query(self, query: str) -> RetrievalOutput:
        if not self.documents:
            raise RuntimeError("Call build() before query().")

        RLM, RLMLogger = _import_official_rlm()
        record_reasoning = bool(self.cfg.get("record_reasoning_trajectory", True))
        logger = self._make_logger(RLMLogger) if record_reasoning else None
        rlm = RLM(logger=logger, custom_tools=self._custom_tools(), **self._rlm_kwargs())
        completion = rlm.completion(
            prompt=self._prompt_payload(query),
            root_prompt=self._root_prompt(),
        )

        response = str(getattr(completion, "response", "") or "")
        usage = _usage_from_completion(completion)
        spans, parse_error, parse_source = self._spans_from_completion(
            completion=completion,
            response=response,
        )
        trajectory = self._reasoning_trajectory(
            query=query,
            completion=completion,
            response=response,
            spans=spans,
        )

        metadata = {
            "retriever": self.method_name,
            "rlm_variant": self.method_name,
            "rlm_backend": self._backend(),
            "rlm_model": self._model_name(),
            "rlm_turn_count": trajectory["turn_count"],
            "rlm_lm_call_count": trajectory["llm_call_count"],
            "rlm_final_response": response,
            "rlm_usage_summary": trajectory["usage_summary"],
            "rlm_parse_source": parse_source,
            "rlm_vector_tool_enabled": self.vector_tool is not None,
            "rlm_pageindex_tool_enabled": bool(self.pageindex_documents),
        }
        if record_reasoning:
            metadata["reasoning_trajectory"] = trajectory
        return RetrievalOutput(
            spans=spans,
            usage=usage,
            metadata=metadata,
            error=parse_error,
        )

    def _make_logger(self, logger_cls: Any) -> Any:
        log_dir = self.cfg.get("log_dir")
        if log_dir:
            return logger_cls(log_dir=str(log_dir), file_name=str(self.cfg.get("log_name", "rlm")))
        return logger_cls()

    def _rlm_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "backend": self._backend(),
            "backend_kwargs": self._backend_kwargs(),
            "environment": str(self.cfg.get("environment", "local")),
            "environment_kwargs": dict(self.cfg.get("environment_kwargs") or {}),
            "max_depth": int(self.cfg.get("max_depth", 1)),
            "max_iterations": int(self.cfg.get("max_iterations", 8)),
            "verbose": bool(self.cfg.get("verbose", False)),
        }
        for key in (
            "max_budget",
            "max_timeout",
            "max_tokens",
            "max_errors",
            "custom_system_prompt",
            "compaction",
            "compaction_threshold_pct",
            "max_concurrent_subcalls",
        ):
            if key in self.cfg and self.cfg[key] is not None:
                kwargs[key] = self.cfg[key]
        return kwargs

    def _backend(self) -> str:
        return str(self.cfg.get("backend") or "anthropic")

    def _backend_kwargs(self) -> dict[str, Any]:
        backend = self._backend()
        kwargs = dict(self.cfg.get("backend_kwargs") or {})
        kwargs.setdefault("model_name", self._model_name())
        api_key_env = self._api_key_env(backend)
        if "api_key" not in kwargs and api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing RLM API key. Set {api_key_env} or configure "
                    "rlm.backend_kwargs.api_key."
                )
            kwargs["api_key"] = api_key
        if backend == "anthropic" and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(
                self.cfg.get("backend_max_tokens", self.llm_cfg.get("max_tokens", 700))
            )
        return kwargs

    def _model_name(self) -> str:
        if self.cfg.get("model_name"):
            return str(self.cfg["model_name"])
        if self.cfg.get("model"):
            return str(self.cfg["model"])
        if self._backend() == "anthropic" and self.llm_cfg.get("model"):
            return str(self.llm_cfg["model"])
        return "gpt-5-nano"

    def _api_key_env(self, backend: str) -> str:
        if self.cfg.get("api_key_env"):
            return str(self.cfg["api_key_env"])
        if backend == "anthropic":
            return str(self.llm_cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
        if backend in {"openai", "vllm"}:
            return "OPENAI_API_KEY"
        if backend == "openrouter":
            return "OPENROUTER_API_KEY"
        if backend == "portkey":
            return "PORTKEY_API_KEY"
        if backend == "gemini":
            return "GEMINI_API_KEY"
        return ""

    def _prompt_payload(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "corpus": {doc_id: doc.text for doc_id, doc in self.documents.items()},
        }

    def _root_prompt(self) -> str:
        prompt_style = str(self.cfg.get("prompt_style", "baseline")).lower()
        if prompt_style == "pageindex":
            return _PAGEINDEX_ROOT_PROMPT
        if prompt_style != "recall_plus":
            return _BASE_ROOT_PROMPT

        min_candidates = max(1, int(self.cfg.get("min_candidate_regions_before_finalize", 3)))
        sections = [_BASE_ROOT_PROMPT.rstrip(), "", "Search policy:"]
        if self._vector_tool_enabled():
            sections.extend(
                [
                    "- Use `vector_search` early to surface semantic candidate regions before broad manual searching.",
                    "- You still have full direct corpus access in Python if you need to verify or expand beyond the retrieved regions.",
                ]
            )
        sections.extend(
            [
                "- During search, optimize for recall first and precision second.",
                f"- Do not finalize after the first plausible hit. Inspect at least {min_candidates} candidate regions unless the document is clearly exhausted.",
                "- When you find a likely clause, keep searching for exceptions, carve-outs, follow-on provisions, and related clauses elsewhere in the document.",
                "- Return multiple spans when the answer is distributed across multiple clauses or when one clause is modified by another clause.",
            ]
        )
        return "\n".join(sections)

    def _custom_tools(self) -> dict[str, Any]:
        tools = {
            "make_span": {
                "tool": self._tool_make_span,
                "description": (
                    "make_span(document_id: str, snippet: str) -> dict. "
                    "Locates `snippet` in the named corpus document and returns "
                    "a span dict with correctly computed start_char / end_char. "
                    "Use this for every span in your final answer."
                ),
            },
            "list_documents": {
                "tool": self._tool_list_documents,
                "description": (
                    "Return source document ids, lengths, and short previews. "
                    "Accepts optional limit and offset integer arguments."
                ),
            },
            "search_documents": {
                "tool": self._tool_search_documents,
                "description": (
                    "Keyword-search the full source corpus. Arguments: query, "
                    "limit=10, per_document_limit=3, window_chars=900. Returns "
                    "candidate spans with document_id/start_char/end_char/text/score."
                ),
            },
            "read_document_span": {
                "tool": self._tool_read_document_span,
                "description": (
                    "Read an exact character span from a source document. "
                    "Arguments: document_id, start_char, end_char."
                ),
            },
            "get_document_length": {
                "tool": self._tool_get_document_length,
                "description": "Return the character length for a document_id.",
            },
        }
        if self.pageindex_documents:
            tools.update(
                {
                    "list_pageindex_documents": {
                        "tool": self._tool_list_pageindex_documents,
                        "description": (
                            "List documents with cached PageIndex trees. "
                            "Accepts optional limit and offset integer arguments."
                        ),
                    },
                    "search_pageindex_nodes": {
                        "tool": self._tool_search_pageindex_nodes,
                        "description": (
                            "Search cached PageIndex nodes by semantic summaries/title/section. "
                            "Arguments: query, document_id=None, limit=10, include_text=false."
                        ),
                    },
                    "list_pageindex_children": {
                        "tool": self._tool_list_pageindex_children,
                        "description": (
                            "List child nodes for a cached PageIndex node. "
                            "Arguments: document_id, node_id=None, limit=50."
                        ),
                    },
                    "read_pageindex_node": {
                        "tool": self._tool_read_pageindex_node,
                        "description": (
                            "Read one cached PageIndex node and optionally its exact source text. "
                            "Arguments: document_id, node_id, include_text=true."
                        ),
                    },
                    "make_span_from_pageindex_node": {
                        "tool": self._tool_make_span_from_pageindex_node,
                        "description": (
                            "Create an exact span dict from a cached PageIndex node. "
                            "Arguments: document_id, node_id, max_chars=None."
                        ),
                    },
                }
            )
        if self.vector_tool is not None:
            tools["vector_search"] = {
                "tool": self._tool_vector_search,
                "description": (
                    "Semantic-hybrid vector retrieval over the corpus. Arguments: "
                    "query, top_k=8, include_text=true. Returns candidate spans with "
                    "document_id/start_char/end_char/text/score/metadata."
                ),
            }
        return tools

    def _tool_list_documents(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        return self.document_catalog[offset : offset + limit]

    def _tool_get_document_length(self, document_id: str) -> int:
        doc = self.documents.get(str(document_id))
        if doc is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        return len(doc.text)

    def _tool_make_span(self, document_id: str, snippet: str) -> dict[str, Any]:
        doc = self.documents.get(str(document_id))
        if doc is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        snippet_text = str(snippet)
        start = doc.text.find(snippet_text)
        if start < 0:
            raise ValueError(
                f"snippet not found in {document_id!r} "
                f"(first 80 chars: {snippet_text[:80]!r})"
            )
        return {
            "document_id": doc.document_id,
            "start_char": start,
            "end_char": start + len(snippet_text),
            "snippet": snippet_text,
        }

    def _tool_read_document_span(
        self, document_id: str, start_char: int, end_char: int
    ) -> dict[str, Any]:
        doc = self.documents.get(str(document_id))
        if doc is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        start = max(0, min(int(start_char), len(doc.text)))
        end = max(start, min(int(end_char), len(doc.text)))
        return {
            "document_id": doc.document_id,
            "start_char": start,
            "end_char": end,
            "text": doc.text[start:end],
        }

    def _tool_search_documents(
        self,
        query: str,
        limit: int = 10,
        per_document_limit: int = 3,
        window_chars: int = 900,
    ) -> list[dict[str, Any]]:
        terms = _query_terms(str(query), max_terms=int(self.cfg.get("max_search_terms", 12)))
        limit = max(1, min(int(limit), int(self.cfg.get("max_tool_results", 50))))
        per_document_limit = max(1, min(int(per_document_limit), 10))
        window_chars = max(120, min(int(window_chars), int(self.cfg.get("max_tool_window_chars", 2500))))
        if not terms:
            return [
                {
                    "document_id": item["document_id"],
                    "start_char": 0,
                    "end_char": min(item["length"], window_chars),
                    "text": self.documents[item["document_id"]].text[:window_chars],
                    "score": 0.0,
                    "matched_terms": [],
                }
                for item in self.document_catalog[:limit]
            ]

        results: list[dict[str, Any]] = []
        max_hits_per_term = int(self.cfg.get("max_hits_per_term_per_document", 30))
        for doc in self.documents.values():
            lower = doc.text.lower()
            candidates: list[tuple[float, int, int, list[str]]] = []
            for term in terms:
                pos = lower.find(term)
                hits = 0
                while pos >= 0 and hits < max_hits_per_term:
                    start = max(0, pos - window_chars // 2)
                    end = min(len(doc.text), start + window_chars)
                    start = max(0, end - window_chars)
                    window = lower[start:end]
                    matched = [candidate for candidate in terms if candidate in window]
                    score = sum(window.count(candidate) for candidate in matched)
                    if matched:
                        candidates.append((float(score), start, end, matched))
                    hits += 1
                    pos = lower.find(term, pos + max(1, len(term)))
            if not candidates:
                continue
            deduped: dict[tuple[int, int], tuple[float, list[str]]] = {}
            for score, start, end, matched in candidates:
                key = (start, end)
                current = deduped.get(key)
                if current is None or score > current[0]:
                    deduped[key] = (score, matched)
            for (start, end), (score, matched) in sorted(
                deduped.items(), key=lambda item: item[1][0], reverse=True
            )[:per_document_limit]:
                results.append(
                    {
                        "document_id": doc.document_id,
                        "start_char": start,
                        "end_char": end,
                        "text": doc.text[start:end],
                        "score": score,
                        "matched_terms": matched,
                    }
                )
        results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return results[:limit]

    def _tool_vector_search(
        self,
        query: str,
        top_k: int = 8,
        include_text: bool = True,
    ) -> list[dict[str, Any]]:
        if self.vector_tool is None:
            return []
        tool_cfg = self.cfg.get("vector_tool") or {}
        limit = max(1, min(int(top_k), int(tool_cfg.get("max_results", 8))))
        include_text = bool(include_text if include_text is not None else tool_cfg.get("include_text", True))
        retrieval = self.vector_tool.query(str(query))
        items: list[dict[str, Any]] = []
        for span in retrieval.spans[:limit]:
            metadata = span.metadata or {}
            item = {
                "document_id": span.document_id,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "score": span.score,
                "metadata": {
                    key: metadata.get(key)
                    for key in (
                        "chunk_title",
                        "chunk_level",
                        "search_strategy",
                        "vector_score",
                        "bm25_score",
                        "hybrid_score",
                    )
                    if key in metadata
                },
            }
            if include_text:
                item["text"] = span.text
            items.append(item)
        return items

    def _tool_list_pageindex_documents(
        self, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        return self.pageindex_document_catalog[offset : offset + limit]

    def _tool_search_pageindex_nodes(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        include_text: bool = False,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), int(self.cfg.get("max_tool_results", 50))))
        doc_filter = str(document_id or "").strip()
        terms = _query_terms(str(query), max_terms=int(self.cfg.get("max_search_terms", 12)))
        if not terms:
            terms = [str(query).strip().lower()] if str(query).strip() else []

        results: list[dict[str, Any]] = []
        for doc_id, tree in self.pageindex_documents.items():
            if doc_filter and doc_id != doc_filter:
                continue
            for node in _walk_pageindex_nodes(tree):
                haystacks = {
                    "title": str(node.get("title", "")).lower(),
                    "summary": str(node.get("summary", "")).lower(),
                    "section": str(node.get("section_title", "")).lower(),
                }
                score = 0.0
                matched_terms: list[str] = []
                for term in terms:
                    if not term:
                        continue
                    matched = False
                    if term in haystacks["title"]:
                        score += 4.0
                        matched = True
                    if term in haystacks["summary"]:
                        score += 2.0
                        matched = True
                    if term in haystacks["section"]:
                        score += 1.0
                        matched = True
                    if matched:
                        matched_terms.append(term)
                if score <= 0:
                    continue
                item = self._pageindex_node_payload(doc_id, node, include_text=include_text)
                item["score"] = score
                item["matched_terms"] = matched_terms
                results.append(item)
        results.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                -int(item.get("start_char") or 0),
            ),
            reverse=True,
        )
        return results[:limit]

    def _tool_list_pageindex_children(
        self, document_id: str, node_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        doc_id = str(document_id)
        tree = self.pageindex_documents.get(doc_id)
        if tree is None:
            raise KeyError(f"Unknown PageIndex document_id: {document_id}")
        node = tree if not node_id else self._get_pageindex_node(doc_id, str(node_id))
        children = node.get("children") or []
        if not isinstance(children, list):
            return []
        return [
            self._pageindex_node_payload(doc_id, child, include_text=False)
            for child in children[:limit]
            if isinstance(child, dict)
        ]

    def _tool_read_pageindex_node(
        self, document_id: str, node_id: str, include_text: bool = True
    ) -> dict[str, Any]:
        doc_id = str(document_id)
        node = self._get_pageindex_node(doc_id, str(node_id))
        return self._pageindex_node_payload(doc_id, node, include_text=bool(include_text))

    def _tool_make_span_from_pageindex_node(
        self, document_id: str, node_id: str, max_chars: int | None = None
    ) -> dict[str, Any]:
        doc_id = str(document_id)
        node = self._get_pageindex_node(doc_id, str(node_id))
        doc = self.documents.get(doc_id)
        if doc is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        start = max(0, min(int(node.get("start_char", 0)), len(doc.text)))
        end = max(start, min(int(node.get("end_char", start)), len(doc.text)))
        if max_chars is not None:
            end = min(end, start + max(1, int(max_chars)))
        snippet = doc.text[start:end]
        return {
            "document_id": doc_id,
            "start_char": start,
            "end_char": end,
            "snippet": snippet,
            "node_id": str(node.get("node_id", "")),
        }

    def _spans_from_completion(
        self,
        *,
        completion: Any,
        response: str,
    ) -> tuple[list[RetrievedSpan], str, str]:
        errors: list[str] = []
        for source, candidate in self._response_candidates(completion=completion, response=response):
            spans, parse_error = self._spans_from_response(candidate)
            if not parse_error:
                return spans, "", source
            errors.append(f"{source}: {parse_error}")
        return [], "; ".join(errors), ""

    def _response_candidates(
        self,
        *,
        completion: Any,
        response: str,
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(source: str, value: Any) -> None:
            if not isinstance(value, str):
                return
            stripped = value.strip()
            if not stripped or stripped in seen:
                return
            seen.add(stripped)
            candidates.append((source, stripped))

        add("response", response)
        raw_metadata = getattr(completion, "metadata", None) or {}
        if isinstance(raw_metadata, dict):
            add("metadata.final_answer", raw_metadata.get("final_answer"))
            iterations = raw_metadata.get("iterations") or []
            if isinstance(iterations, list):
                for idx, item in enumerate(iterations, start=1):
                    if not isinstance(item, dict):
                        continue
                    add(f"iterations[{idx}].final_answer", item.get("final_answer"))
                    for block_idx, block in enumerate(item.get("code_blocks") or [], start=1):
                        if not isinstance(block, dict):
                            continue
                        result = block.get("result") or {}
                        if isinstance(result, dict):
                            add(
                                f"iterations[{idx}].code_blocks[{block_idx}].final_answer",
                                result.get("final_answer"),
                            )
        return candidates

    def _spans_from_response(self, response: str) -> tuple[list[RetrievedSpan], str]:
        try:
            parsed = _parse_response_mapping(response)
        except Exception as exc:  # noqa: BLE001
            return [], f"rlm_response_parse_error: {type(exc).__name__}: {exc}"

        raw_spans = parsed.get("spans") or parsed.get("retrieved_spans") or []
        if not raw_spans and isinstance(parsed.get("documents"), list):
            raw_spans = [
                {
                    "document_id": item.get("document_id"),
                    "start_char": 0,
                    "end_char": self._default_document_span_end(str(item.get("document_id", ""))),
                    "reason": item.get("reason", "RLM selected document"),
                }
                for item in parsed["documents"]
                if isinstance(item, dict)
            ]
        if not isinstance(raw_spans, list):
            return [], "rlm_response_parse_error: `spans` is not a list"

        spans: list[RetrievedSpan] = []
        errors: list[str] = []
        max_chars = int(self.cfg.get("max_retrieved_chars_per_span", 5000))
        for rank, item in enumerate(raw_spans, start=1):
            if not isinstance(item, dict):
                continue
            try:
                span = self._span_from_item(item, rank=rank, max_chars=max_chars)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"span[{rank}]: {type(exc).__name__}: {exc}")
                continue
            if span is not None:
                spans.append(span)
            if len(spans) >= max(1, int(self.cfg.get("selected_spans", 5))):
                break
        return spans, "; ".join(errors)

    def _span_from_item(
        self, item: dict[str, Any], *, rank: int, max_chars: int
    ) -> RetrievedSpan | None:
        doc_id = str(item.get("document_id") or item.get("doc_id") or "")
        if doc_id not in self.documents:
            raise KeyError(f"Unknown document_id: {doc_id}")
        doc = self.documents[doc_id]
        snippet = str(item.get("snippet") or item.get("text") or item.get("excerpt") or "")
        start = _coerce_optional_int(item.get("start_char", item.get("start")))
        end = _coerce_optional_int(item.get("end_char", item.get("end")))

        if snippet:
            if start is not None and end is not None:
                clamped_start = max(0, min(int(start), len(doc.text)))
                clamped_end = max(clamped_start, min(int(end), len(doc.text)))
                if doc.text[clamped_start:clamped_end] != snippet:
                    found = doc.text.find(snippet)
                    if found >= 0:
                        start = found
                        end = found + len(snippet)
            else:
                found = doc.text.find(snippet)
                if found >= 0:
                    start = found
                    end = found + len(snippet)

        if start is None or end is None:
            if not snippet:
                raise ValueError("Missing start_char/end_char and text excerpt")
            found = doc.text.find(snippet)
            if found < 0:
                raise ValueError("Could not locate text excerpt in document")
            start = found
            end = found + len(snippet)

        start = max(0, min(int(start), len(doc.text)))
        end = max(start, min(int(end), len(doc.text)))
        if max_chars > 0:
            end = min(end, start + max_chars)
        if end <= start:
            return None

        reason = str(item.get("reason") or item.get("rationale") or "")
        score = _coerce_float(item.get("score"), default=1.0 / rank)
        return RetrievedSpan(
            document_id=doc_id,
            start_char=start,
            end_char=end,
            text=doc.text[start:end],
            score=score,
            metadata={
                "retriever": self.method_name,
                "rlm_variant": self.method_name,
                "rank": rank,
                "reason": reason,
                "rlm_reason": reason,
            },
        )

    def _default_document_span_end(self, doc_id: str) -> int:
        doc = self.documents.get(doc_id)
        if doc is None:
            return 0
        max_chars = int(self.cfg.get("max_retrieved_chars_per_span", 5000))
        return min(len(doc.text), max_chars if max_chars > 0 else len(doc.text))

    def _reasoning_trajectory(
        self,
        *,
        query: str,
        completion: Any,
        response: str,
        spans: list[RetrievedSpan],
    ) -> dict[str, Any]:
        raw_metadata = getattr(completion, "metadata", None) or {}
        iterations = raw_metadata.get("iterations") if isinstance(raw_metadata, dict) else []
        normalized_turns = [_normalize_iteration(item, idx) for idx, item in enumerate(iterations or [], start=1)]
        usage_summary = _usage_summary_dict(getattr(completion, "usage_summary", None))
        return {
            "type": self.method_name,
            "method_name": self.method_name,
            "query": query,
            "turn_count": len(normalized_turns),
            "llm_call_count": _usage_call_count(usage_summary),
            "final_response": response,
            "run_metadata": raw_metadata.get("run_metadata", {}) if isinstance(raw_metadata, dict) else {},
            "usage_summary": usage_summary,
            "iterations": normalized_turns,
            "retrieved_spans": [
                {
                    "document_id": span.document_id,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "score": span.score,
                    "reason": span.metadata.get("reason", ""),
                }
                for span in spans
            ],
        }

    def _vector_tool_enabled(self) -> bool:
        return bool((self.cfg.get("vector_tool") or {}).get("enabled", False))

    def _pageindex_tool_enabled(self) -> bool:
        return bool((self.cfg.get("pageindex_tool") or {}).get("enabled", False))

    def _build_vector_tool(self, documents: list[Document]) -> VectorRAG:
        if not self.vector_tool_cfg:
            raise RuntimeError(
                f"{self.method_name} enables vector_tool but no vector helper config was provided."
            )
        cache_dir = self.vector_tool_cache_dir or Path(
            self.vector_tool_cfg.get("cache_dir", ".cache/vector")
        )
        vector_tool = VectorRAG(self.vector_tool_cfg, cache_dir=cache_dir)
        vector_tool.build(documents)
        return vector_tool

    def _load_pageindex_tool(self, documents: list[Document]) -> None:
        cache_dir = self.pageindex_tool_cache_dir or Path(
            self.pageindex_tool_cfg.get("cache_dir", ".cache/pageindex")
        )
        for doc in documents:
            cache_path = cache_dir / f"{self._pageindex_doc_hash(doc)}.json"
            if not cache_path.exists():
                raise RuntimeError(
                    f"{self.method_name} requires PageIndex cache for {doc.document_id}, "
                    f"but {cache_path} does not exist."
                )
            with open(cache_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
            self.pageindex_documents[doc.document_id] = tree
            nodes = [node for node in _walk_pageindex_nodes(tree)]
            for node in nodes:
                node_id = str(node.get("node_id", ""))
                if node_id:
                    self.pageindex_node_lookup[(doc.document_id, node_id)] = node
            self.pageindex_document_catalog.append(
                {
                    "document_id": doc.document_id,
                    "title": str(tree.get("title", "")),
                    "summary": str(tree.get("summary", "")),
                    "unit_start": tree.get("unit_start"),
                    "unit_end": tree.get("unit_end"),
                    "node_count": len(nodes),
                    "root_node_id": str(tree.get("node_id", "")),
                }
            )
        self.pageindex_document_catalog.sort(key=lambda item: item["document_id"])

    def _get_pageindex_node(self, document_id: str, node_id: str) -> dict[str, Any]:
        node = self.pageindex_node_lookup.get((document_id, node_id))
        if node is None:
            raise KeyError(
                f"Unknown PageIndex node_id {node_id!r} for document_id {document_id!r}"
            )
        return node

    def _pageindex_node_payload(
        self, document_id: str, node: dict[str, Any], *, include_text: bool
    ) -> dict[str, Any]:
        doc = self.documents.get(document_id)
        if doc is None:
            raise KeyError(f"Unknown document_id: {document_id}")
        start = max(0, min(int(node.get("start_char", 0)), len(doc.text)))
        end = max(start, min(int(node.get("end_char", start)), len(doc.text)))
        payload = {
            "document_id": document_id,
            "node_id": str(node.get("node_id", "")),
            "node_kind": str(node.get("node_kind", "")),
            "title": str(node.get("title", "")),
            "summary": str(node.get("summary", "")),
            "section_title": str(node.get("section_title", "")),
            "start_char": start,
            "end_char": end,
            "unit_start": node.get("unit_start"),
            "unit_end": node.get("unit_end"),
            "child_count": len(node.get("children") or []),
        }
        if include_text:
            payload["text"] = doc.text[start:end]
        return payload

    def _pageindex_doc_hash(self, doc: Document) -> str:
        payload = {
            "document_id": doc.document_id,
            "length": len(doc.text),
            "sha1": hashlib.sha1(doc.text.encode("utf-8")).hexdigest(),
            "build_config": self._pageindex_build_cache_config(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _pageindex_build_cache_config(self) -> dict[str, Any]:
        return {
            "build_with_llm": bool(self.pageindex_tool_cfg.get("build_with_llm", True)),
            "virtual_page_target_tokens": int(
                self.pageindex_tool_cfg.get("virtual_page_target_tokens", 900)
            ),
            "virtual_page_max_tokens": int(
                self.pageindex_tool_cfg.get("virtual_page_max_tokens", 1200)
            ),
            "toc_check_units": int(self.pageindex_tool_cfg.get("toc_check_units", 20)),
            "max_units_per_node": int(self.pageindex_tool_cfg.get("max_units_per_node", 10)),
            "max_tokens_per_node": int(
                self.pageindex_tool_cfg.get("max_tokens_per_node", 20000)
            ),
            "node_summary_max_tokens": int(
                self.pageindex_tool_cfg.get("node_summary_max_tokens", 220)
            ),
            "root_summary_max_tokens": int(
                self.pageindex_tool_cfg.get("root_summary_max_tokens", 260)
            ),
        }


def _walk_pageindex_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        nodes.append(node)
        children = node.get("children") or []
        if isinstance(children, list):
            stack.extend(reversed([child for child in children if isinstance(child, dict)]))
    return nodes


def _import_official_rlm() -> tuple[Any, Any]:
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except ImportError as exc:
        raise RuntimeError(
            "The official RLM package is required for method `rlm`. "
            "Install it from https://github.com/alexzhang13/rlm "
            "(package name `rlms`). "
            f"Current Python is {sys.version.split()[0]}; "
            "the official package currently declares Python >=3.11. "
            "Recommended fix: `uv venv --python 3.11 && uv sync --dev --extra rlm`."
        ) from exc
    return RLM, RLMLogger


def _usage_from_completion(completion: Any) -> Usage:
    summary = getattr(completion, "usage_summary", None)
    return Usage(
        input_tokens=int(getattr(summary, "total_input_tokens", 0) or 0),
        output_tokens=int(getattr(summary, "total_output_tokens", 0) or 0),
        estimated_cost_usd=float(getattr(summary, "total_cost", 0.0) or 0.0),
    )


def _usage_summary_dict(summary: Any) -> dict[str, Any]:
    if summary is None:
        return {}
    if hasattr(summary, "to_dict"):
        value = summary.to_dict()
        return value if isinstance(value, dict) else {}
    if isinstance(summary, dict):
        return summary
    return {}


def _usage_call_count(usage_summary: dict[str, Any]) -> int:
    models = usage_summary.get("model_usage_summaries") or {}
    if not isinstance(models, dict):
        return 0
    total = 0
    for item in models.values():
        if isinstance(item, dict):
            total += int(item.get("total_calls") or 0)
    return total


def _normalize_iteration(item: Any, idx: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"turn": idx, "llm_output": str(item)}
    code_blocks = []
    for block_idx, block in enumerate(item.get("code_blocks") or [], start=1):
        if not isinstance(block, dict):
            continue
        result = block.get("result") or {}
        if not isinstance(result, dict):
            result = {}
        code_blocks.append(
            {
                "block": block_idx,
                "code": str(block.get("code") or ""),
                "stdout": str(result.get("stdout") or ""),
                "stderr": str(result.get("stderr") or ""),
                "final_answer": result.get("final_answer"),
                "rlm_call_count": len(result.get("rlm_calls") or []),
                "rlm_calls": [
                    {
                        "response": str(call.get("response") or ""),
                        "execution_time": call.get("execution_time"),
                    }
                    for call in result.get("rlm_calls") or []
                    if isinstance(call, dict)
                ],
            }
        )
    return {
        "turn": int(item.get("iteration") or idx),
        "timestamp": item.get("timestamp"),
        "llm_output": str(item.get("response") or ""),
        "final_answer": item.get("final_answer"),
        "iteration_time": item.get("iteration_time"),
        "code_blocks": code_blocks,
    }


def _parse_response_mapping(text: str) -> dict[str, Any]:
    stripped = text.strip()
    parse_errors: list[str] = []
    for candidate in _mapping_candidates(stripped):
        try:
            return _parse_mapping_candidate(candidate)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{type(exc).__name__}: {exc}")
    if parse_errors:
        raise ValueError("; ".join(parse_errors))
    raise ValueError("No response mapping found")


def _mapping_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("result_json"), str):
            return _parse_response_mapping(value["result_json"])
        return value
    if isinstance(value, list):
        return {"spans": value}
    if isinstance(value, str):
        return _parse_response_mapping(value)
    raise ValueError(f"Expected JSON object or span list, got {type(value).__name__}")


def _query_terms(query: str, *, max_terms: int) -> list[str]:
    terms = []
    seen = set()
    for match in _TOKEN_RE.finditer(query.lower()):
        term = match.group(0)
        if len(term) < 3 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_mapping_candidate(text: str) -> dict[str, Any]:
    try:
        return _mapping_from_value(json.loads(text))
    except Exception:  # noqa: BLE001
        pass
    return _mapping_from_value(ast.literal_eval(text))


def _mapping_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            return
        seen.add(stripped)
        candidates.append(stripped)

    add(text)
    for block in _fenced_block_contents(text):
        add(block)
    try:
        extracted = extract_json_object(text)
    except Exception:  # noqa: BLE001
        extracted = None
    if extracted is not None:
        add(json.dumps(extracted))
    for candidate in _iter_balanced_mapping_texts(text):
        add(candidate)
    return [candidate for candidate in candidates if _looks_like_mapping_text(candidate)]


def _fenced_block_contents(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:json|python)?\s*(.*?)```", text, re.DOTALL):
        blocks.append(match.group(1))
    return blocks


def _looks_like_mapping_text(text: str) -> bool:
    return any(key in text for key in _RESULT_KEYS)


def _iter_balanced_mapping_texts(text: str):
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        end = _find_balanced_end(text, start)
        if end is None:
            continue
        candidate = text[start : end + 1]
        if _looks_like_mapping_text(candidate):
            yield candidate


def _find_balanced_end(text: str, start: int) -> int | None:
    depth = 0
    quote: str | None = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if quote is not None:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None
