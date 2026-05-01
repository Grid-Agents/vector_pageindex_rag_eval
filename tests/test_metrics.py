from rag_eval.metrics import score_retrieval
from rag_eval.types import GoldSpan, RetrievedSpan


def test_score_retrieval_char_overlap():
    gold = [GoldSpan("doc.txt", 10, 20)]
    pred = [RetrievedSpan("doc.txt", 15, 25, "x" * 10)]

    scores = score_retrieval(gold, pred)

    assert scores["overlap_chars"] == 5
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5


def test_score_retrieval_merges_overlapping_predictions():
    gold = [GoldSpan("doc.txt", 10, 20)]
    pred = [
        RetrievedSpan("doc.txt", 8, 16, "x" * 8),
        RetrievedSpan("doc.txt", 14, 22, "x" * 8),
    ]

    scores = score_retrieval(gold, pred)

    assert scores["retrieved_chars"] == 14
    assert scores["overlap_chars"] == 10
    assert round(scores["precision"], 4) == round(10 / 14, 4)
    assert scores["recall"] == 1.0

