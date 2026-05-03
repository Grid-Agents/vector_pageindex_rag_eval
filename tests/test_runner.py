from rag_eval.runner import _run_method
from rag_eval.types import (
    Example,
    GoldSpan,
    LLMResponse,
    RetrievedSpan,
    RetrievalOutput,
    Usage,
)


class FakeSystem:
    def __init__(self, spans):
        self._spans = spans

    def query(self, query):
        return RetrievalOutput(spans=self._spans)


class FakeLLM:
    def complete(self, **kwargs):
        return LLMResponse(text="qualitative answer only", usage=Usage(output_tokens=3))


def test_run_method_scores_retrieval_when_answer_generation_is_disabled():
    example = Example(
        example_id="ex-1",
        benchmark="toy",
        query="What text?",
        gold_spans=[GoldSpan("doc.txt", 10, 20)],
    )
    system = FakeSystem([RetrievedSpan("doc.txt", 10, 20, "x" * 10)])

    result = _run_method(
        method_name="fake",
        system=system,
        example=example,
        llm=None,
        answer_with_llm=False,
        max_answer_context_chars=100,
    )

    assert result["evaluation_target"] == "retrieval"
    assert result["answer_generated"] is False
    assert result["answer"] == ""
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_run_method_generated_answer_does_not_change_retrieval_score():
    example = Example(
        example_id="ex-1",
        benchmark="toy",
        query="What text?",
        gold_spans=[GoldSpan("doc.txt", 10, 20)],
    )
    system = FakeSystem([RetrievedSpan("doc.txt", 10, 20, "x" * 10)])

    result = _run_method(
        method_name="fake",
        system=system,
        example=example,
        llm=FakeLLM(),
        answer_with_llm=True,
        max_answer_context_chars=100,
    )

    assert result["answer_generated"] is True
    assert result["answer"] == "qualitative answer only"
    assert result["output_tokens"] == 3
    assert result["f1"] == 1.0
