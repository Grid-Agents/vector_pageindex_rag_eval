from rag_eval.pageindex_rag import PageIndexRAG, _flatten_tree
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
        return LLMResponse(text=self._responses.pop(0), usage=Usage())


def test_pageindex_selects_documents_then_nodes(tmp_path):
    docs = [
        Document(
            "cuad/license.txt",
            "LICENSE TERMS\nGrant of license to use the software in the territory.\n",
        ),
        Document(
            "cuad/insurance.txt",
            "INSURANCE\nMaintain commercial general liability coverage.\n",
        ),
    ]
    llm = SequencedLLM(
        [
            '{"documents":[{"document_id":"cuad/license.txt","reason":"license query"}]}',
            '{"selections":[{"node_id":"PLACEHOLDER","reason":"contains grant language"}]}',
        ]
    )
    rag = PageIndexRAG(
        {
            "build_with_llm": False,
            "virtual_page_target_tokens": 60,
            "virtual_page_max_tokens": 80,
            "max_units_per_node": 4,
            "selected_documents": 1,
            "selected_nodes": 1,
        },
        llm,
        cache_dir=tmp_path,
    )
    rag.build(docs)
    leaf_id = next(
        node["node_id"]
        for node in _flatten_tree(rag.trees["cuad/license.txt"])
        if not node.get("children")
    )
    llm._responses[1] = (
        '{"selections":[{"node_id":"%s","reason":"contains grant language"}]}'
        % leaf_id
    )

    result = rag.query("What licenses are granted under this contract?")

    assert len(llm.calls) == 2
    assert result.spans
    assert result.spans[0].document_id == "cuad/license.txt"
    assert "license" in result.spans[0].text.lower()
    assert result.metadata["document_selections"] == [
        {"document_id": "cuad/license.txt", "reason": "license query"}
    ]
    trajectory = result.metadata["reasoning_trajectory"]
    assert trajectory["document_selection"]["source"] == "llm"
    assert trajectory["document_selection"]["accepted_selections"] == [
        {"document_id": "cuad/license.txt", "reason": "license query"}
    ]
    assert trajectory["document_walks"][0]["document_id"] == "cuad/license.txt"
    assert trajectory["document_walks"][0]["steps"][0]["accepted_selections"] == [
        {"node_id": leaf_id, "reason": "contains grant language"}
    ]
    assert trajectory["retrieved_nodes"][0]["node_id"] == leaf_id


def test_pageindex_builds_virtual_page_leaves(tmp_path):
    text = (
        "SECTION 1\n"
        + ("Alpha beta gamma. " * 100)
        + "\n\nSECTION 2\n"
        + ("Delta epsilon zeta. " * 100)
    )
    rag = PageIndexRAG(
        {
            "build_with_llm": False,
            "virtual_page_target_tokens": 60,
            "virtual_page_max_tokens": 80,
            "max_units_per_node": 2,
        },
        llm=None,
        cache_dir=tmp_path,
    )
    rag.build([Document("cuad/long.txt", text)])

    tree = rag.trees["cuad/long.txt"]
    leaves = [node for node in _flatten_tree(tree) if not node.get("children")]

    assert leaves
    assert tree["metadata"]["unit_type"] == "virtual_page"
    assert tree["unit_end"] == len(leaves)
    assert all(node["node_kind"] == "page" for node in leaves)
    assert all(node["unit_start"] == node["unit_end"] for node in leaves)
    assert tree["children"]


def test_pageindex_can_resume_from_partial_checkpoint(tmp_path):
    class InterruptingPageIndexRAG(PageIndexRAG):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._checkpoint_writes = 0

        def _write_build_checkpoint(self, doc, *, usage, page_nodes, top_children):
            super()._write_build_checkpoint(
                doc,
                usage=usage,
                page_nodes=page_nodes,
                top_children=top_children,
            )
            self._checkpoint_writes += 1
            if self._checkpoint_writes == 1:
                raise KeyboardInterrupt("simulate interrupted build")

    text = (
        "SECTION 1\n"
        + ("Alpha beta gamma. " * 100)
        + "\n\nSECTION 2\n"
        + ("Delta epsilon zeta. " * 100)
        + "\n\nSECTION 3\n"
        + ("Eta theta iota. " * 100)
    )
    doc = Document("cuad/resume.txt", text)
    cfg = {
        "build_with_llm": False,
        "virtual_page_target_tokens": 60,
        "virtual_page_max_tokens": 80,
        "max_units_per_node": 2,
    }
    interrupted = InterruptingPageIndexRAG(cfg, llm=None, cache_dir=tmp_path)

    try:
        interrupted.build([doc])
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected simulated interruption")

    doc_hash = interrupted._doc_hash(doc)
    checkpoint_path = tmp_path / f"{doc_hash}.partial.json"
    final_path = tmp_path / f"{doc_hash}.json"

    assert checkpoint_path.exists()
    assert not final_path.exists()

    resumed = PageIndexRAG(cfg, llm=None, cache_dir=tmp_path)
    resumed.build([doc])

    assert final_path.exists()
    assert not checkpoint_path.exists()

    fresh_dir = tmp_path / "fresh"
    fresh = PageIndexRAG(cfg, llm=None, cache_dir=fresh_dir)
    fresh.build([doc])

    assert resumed.trees["cuad/resume.txt"] == fresh.trees["cuad/resume.txt"]
