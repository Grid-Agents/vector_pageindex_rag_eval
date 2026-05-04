from types import SimpleNamespace

from rag_eval.official_pageindex import OfficialPageIndexRAG
from rag_eval.types import Document, LLMResponse, Usage


class SequencedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, *, system, user, max_tokens=None, temperature=None):
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return LLMResponse(text=self._responses.pop(0), usage=Usage(input_tokens=1))


async def fake_md_to_tree(**kwargs):
    return {
        "doc_name": "license",
        "doc_description": "A license agreement.",
        "line_count": 8,
        "structure": [
            {
                "title": "LICENSE TERMS",
                "node_id": "0000",
                "line_num": 1,
                "summary": "License section",
                "nodes": [
                    {
                        "title": "LICENSE TERMS / virtual page",
                        "node_id": "0001",
                        "line_num": 3,
                        "summary": "Grant of license language",
                    }
                ],
            }
        ],
    }


class FakeOfficialPageIndexRAG(OfficialPageIndexRAG):
    def _load_official_modules(self):
        return SimpleNamespace(
            md_to_tree=fake_md_to_tree,
            utils=SimpleNamespace(),
            source="fake-official-pageindex",
        )


def test_official_pageindex_builds_tree_and_returns_spans(tmp_path):
    docs = [
        Document(
            "cuad/license.txt",
            "LICENSE TERMS\nGrant of license to use the software in the territory.\n",
        )
    ]
    llm = SequencedLLM(
        [
            '{"documents":[{"document_id":"cuad/license.txt","reason":"license query"}]}',
            '{"thinking":"grant language is relevant","node_list":["0001"]}',
        ]
    )
    rag = FakeOfficialPageIndexRAG(
        {
            "build_with_llm": False,
            "virtual_page_target_tokens": 120,
            "virtual_page_max_tokens": 160,
            "selected_documents": 1,
            "selected_nodes": 1,
        },
        llm,
        cache_dir=tmp_path,
    )

    rag.build(docs)
    result = rag.query("What licenses are granted under this contract?")

    assert result.spans
    assert result.spans[0].document_id == "cuad/license.txt"
    assert "Grant of license" in result.spans[0].text
    assert result.spans[0].metadata["retriever"] == "pageindex_official"
    assert result.spans[0].metadata["official_node_id"] == "0001"
    assert result.metadata["reasoning_trajectory"]["method"] == "pageindex_official"
    assert result.metadata["reasoning_trajectory"]["document_walks"][0]["source"] == (
        "official_llm_tree_search"
    )


def test_official_pageindex_cache_reuses_normalized_tree(tmp_path):
    doc = Document("cuad/license.txt", "LICENSE TERMS\nGrant of license.\n")
    rag = FakeOfficialPageIndexRAG(
        {"build_with_llm": False},
        llm=None,
        cache_dir=tmp_path,
    )

    rag.build([doc])
    cached = OfficialPageIndexRAG(
        {"build_with_llm": False},
        llm=None,
        cache_dir=tmp_path,
    )
    cached.build([doc])

    assert cached.trees[doc.document_id] == rag.trees[doc.document_id]
