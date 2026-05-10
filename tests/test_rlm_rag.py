import json
import sys
from types import SimpleNamespace

from rag_eval.rlm_rag import RLMRAG
from rag_eval.types import Document, RetrievedSpan, RetrievalOutput
from rag_eval.visualization import _public_result


class FakeUsageSummary:
    total_input_tokens = 12
    total_output_tokens = 7
    total_cost = 0.03

    def to_dict(self):
        return {
            "model_usage_summaries": {
                "fake-model": {
                    "total_calls": 2,
                    "total_input_tokens": 12,
                    "total_output_tokens": 7,
                    "total_cost": 0.03,
                }
            },
            "total_cost": 0.03,
        }


class FakeLogger:
    def __init__(self, *args, **kwargs):
        pass


class FakeRLM:
    last_kwargs = None
    last_prompt = None
    last_root_prompt = None

    def __init__(self, **kwargs):
        FakeRLM.last_kwargs = kwargs

    def completion(self, prompt, root_prompt=None):
        FakeRLM.last_prompt = prompt
        FakeRLM.last_root_prompt = root_prompt
        return SimpleNamespace(
            response=json.dumps(
                {
                    "spans": [
                        {
                            "document_id": "doc.txt",
                            "start_char": 6,
                            "end_char": 24,
                            "reason": "contains the target clause",
                        }
                    ]
                }
            ),
            usage_summary=FakeUsageSummary(),
            execution_time=1.5,
            metadata={
                "run_metadata": {"root_model": "fake-model"},
                "iterations": [
                    {
                        "type": "iteration",
                        "iteration": 1,
                        "timestamp": "2026-01-01T00:00:00",
                        "response": "I will search the documents.",
                        "final_answer": None,
                        "iteration_time": 0.25,
                        "code_blocks": [
                            {
                                "code": "hits = search_documents('target clause')",
                                "result": {
                                    "stdout": "1 hit",
                                    "stderr": "",
                                    "final_answer": None,
                                    "rlm_calls": [],
                                },
                            }
                        ],
                    }
                ],
            },
        )


def install_fake_rlm(monkeypatch, *, rlm_cls=FakeRLM):
    monkeypatch.setitem(sys.modules, "rlm", SimpleNamespace(RLM=rlm_cls))
    monkeypatch.setitem(sys.modules, "rlm.logger", SimpleNamespace(RLMLogger=FakeLogger))


def write_pageindex_cache(rag, tmp_path, document):
    cache_dir = tmp_path / "pageindex"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{rag._pageindex_doc_hash(document)}.json"
    cache_path.write_text(
        json.dumps(
            {
                "node_id": "root",
                "title": document.document_id,
                "summary": "Root summary",
                "start_char": 0,
                "end_char": len(document.text),
                "unit_start": 1,
                "unit_end": 2,
                "token_count": 16,
                "node_kind": "root",
                "section_title": "",
                "children": [
                    {
                        "node_id": "node-1",
                        "title": "Payment Terms",
                        "summary": "Covers target clause obligations and payment timing.",
                        "start_char": 6,
                        "end_char": 24,
                        "unit_start": 1,
                        "unit_end": 1,
                        "token_count": 8,
                        "node_kind": "page",
                        "section_title": "Section 1",
                        "children": [],
                    }
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    return cache_dir


def test_rlm_rag_parses_spans_and_turn_trajectory(monkeypatch):
    install_fake_rlm(monkeypatch)
    rag = RLMRAG(
        {
            "backend": "openai",
            "backend_kwargs": {"api_key": "fake-key", "model_name": "fake-model"},
            "selected_spans": 2,
        }
    )
    rag.build([Document("doc.txt", "alpha target clause text beta")])

    result = rag.query("Where is the target clause?")

    assert FakeRLM.last_kwargs["backend"] == "openai"
    assert "make_span" in FakeRLM.last_kwargs["custom_tools"]
    assert "search_documents" in FakeRLM.last_kwargs["custom_tools"]
    assert "vector_search" not in FakeRLM.last_kwargs["custom_tools"]
    assert FakeRLM.last_prompt["query"] == "Where is the target clause?"
    assert FakeRLM.last_prompt["corpus"]["doc.txt"] == "alpha target clause text beta"
    assert "make_span" in FakeRLM.last_root_prompt
    assert result.spans[0].document_id == "doc.txt"
    assert result.spans[0].start_char == 6
    assert result.spans[0].text == "target clause text"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 7
    assert result.usage.estimated_cost_usd == 0.03
    trajectory = result.metadata["reasoning_trajectory"]
    assert trajectory["type"] == "rlm"
    assert trajectory["turn_count"] == 1
    assert trajectory["llm_call_count"] == 2
    assert trajectory["iterations"][0]["llm_output"] == "I will search the documents."


def test_rlm_pageindex_registers_pageindex_tools_and_prompt(monkeypatch, tmp_path):
    install_fake_rlm(monkeypatch)
    document = Document("doc.txt", "alpha target clause text beta")
    rag = RLMRAG(
        {
            "backend": "openai",
            "backend_kwargs": {"api_key": "fake-key", "model_name": "fake-model"},
            "prompt_style": "pageindex",
            "pageindex_tool": {"enabled": True},
        },
        method_name="rlm_pageindex",
        pageindex_tool_cfg={"enabled": True, "build_with_llm": True},
    )
    rag.pageindex_tool_cache_dir = write_pageindex_cache(rag, tmp_path, document)
    rag.build([document])

    result = rag.query("Where is the target clause?")
    node_hits = rag._tool_search_pageindex_nodes("target clause", limit=1, include_text=True)
    node_span = rag._tool_make_span_from_pageindex_node("doc.txt", "node-1")

    assert "list_pageindex_documents" in FakeRLM.last_kwargs["custom_tools"]
    assert "search_pageindex_nodes" in FakeRLM.last_kwargs["custom_tools"]
    assert "make_span_from_pageindex_node" in FakeRLM.last_kwargs["custom_tools"]
    assert "agentic PageIndex workflow" in FakeRLM.last_root_prompt
    assert "Start with PageIndex structure" in FakeRLM.last_root_prompt
    assert result.metadata["retriever"] == "rlm_pageindex"
    assert result.metadata["rlm_variant"] == "rlm_pageindex"
    assert result.metadata["reasoning_trajectory"]["type"] == "rlm_pageindex"
    assert result.metadata["rlm_pageindex_tool_enabled"] is True
    assert node_hits[0]["document_id"] == "doc.txt"
    assert node_hits[0]["node_id"] == "node-1"
    assert node_hits[0]["text"] == "target clause text"
    assert node_span["start_char"] == 6
    assert node_span["end_char"] == 24
    assert node_span["snippet"] == "target clause text"


def test_rlm_rag_repairs_offsets_from_snippet_and_python_literal(monkeypatch):
    class FakeLiteralRLM(FakeRLM):
        def completion(self, prompt, root_prompt=None):
            FakeRLM.last_prompt = prompt
            FakeRLM.last_root_prompt = root_prompt
            return SimpleNamespace(
                response=(
                    "Found likely evidence.\n"
                    "{'answer': '', 'spans': [{'document_id': 'doc.txt', "
                    "'start_char': 0, 'end_char': 5, "
                    "'snippet': 'target clause text'}]}"
                ),
                usage_summary=FakeUsageSummary(),
                execution_time=1.0,
                metadata={},
            )

    install_fake_rlm(monkeypatch, rlm_cls=FakeLiteralRLM)
    rag = RLMRAG(
        {
            "backend": "openai",
            "backend_kwargs": {"api_key": "fake-key", "model_name": "fake-model"},
        }
    )
    rag.build([Document("doc.txt", "alpha target clause text beta")])

    result = rag.query("Where is the target clause?")

    assert result.error == ""
    assert result.spans[0].start_char == 6
    assert result.spans[0].end_char == 24
    assert result.spans[0].text == "target clause text"


def test_rlm_search_tool_returns_exact_offsets():
    rag = RLMRAG({})
    rag.build(
        [
            Document("a.txt", "unrelated boilerplate"),
            Document("b.txt", "before needle clause after"),
        ]
    )

    hits = rag._tool_search_documents("needle clause", limit=1, window_chars=120)

    assert hits[0]["document_id"] == "b.txt"
    assert hits[0]["start_char"] == 0
    assert "needle clause" in hits[0]["text"]
    assert hits[0]["score"] > 0


def test_rlm_make_span_tool_returns_exact_offsets():
    rag = RLMRAG({})
    rag.build([Document("doc.txt", "alpha target clause text beta")])

    span = rag._tool_make_span("doc.txt", "target clause text")

    assert span["start_char"] == 6
    assert span["end_char"] == 24
    assert span["snippet"] == "target clause text"


def test_public_result_preserves_rlm_turn_outputs():
    result = {
        "method": "rlm",
        "retrieved_spans": [],
        "retrieval_metadata": {},
        "reasoning_trajectory": {
            "type": "rlm",
            "query": "q",
            "turn_count": 1,
            "llm_call_count": 1,
            "final_response": '{"spans":[]}',
            "iterations": [
                {
                    "turn": 1,
                    "llm_output": "searching",
                    "code_blocks": [{"block": 1, "code": "x = 1", "stdout": "ok"}],
                }
            ],
        },
    }

    public = _public_result(result)

    assert public["reasoning_trajectory"]["type"] == "rlm"
    assert public["reasoning_trajectory"]["iterations"][0]["llm_output"] == "searching"
    assert public["reasoning_trajectory"]["iterations"][0]["code_blocks"][0]["code"] == "x = 1"


def test_public_result_preserves_rlm_pageindex_turn_outputs():
    result = {
        "method": "rlm_pageindex",
        "retrieved_spans": [],
        "retrieval_metadata": {},
        "reasoning_trajectory": {
            "type": "rlm_pageindex",
            "method_name": "rlm_pageindex",
            "query": "q",
            "turn_count": 1,
            "llm_call_count": 1,
            "final_response": '{"spans":[]}',
            "iterations": [
                {
                    "turn": 1,
                    "llm_output": "searching with pageindex tools",
                    "code_blocks": [{"block": 1, "code": "hits = search_pageindex_nodes(...)", "stdout": "ok"}],
                }
            ],
        },
    }

    public = _public_result(result)

    assert public["reasoning_trajectory"]["type"] == "rlm_pageindex"
    assert public["reasoning_trajectory"]["method_name"] == "rlm_pageindex"
    assert (
        public["reasoning_trajectory"]["iterations"][0]["llm_output"]
        == "searching with pageindex tools"
    )
    assert (
        public["reasoning_trajectory"]["iterations"][0]["code_blocks"][0]["code"]
        == "hits = search_pageindex_nodes(...)"
    )
