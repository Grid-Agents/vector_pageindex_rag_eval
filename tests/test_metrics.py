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
    assert scores["document_precision"] == 1.0
    assert scores["document_recall"] == 1.0
    assert scores["document_f1"] == 1.0


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
    assert scores["retrieved_span_count"] == 2


def test_score_retrieval_records_document_level_metrics():
    gold = [
        GoldSpan("doc-a.txt", 10, 20),
        GoldSpan("doc-b.txt", 30, 40),
    ]
    pred = [
        RetrievedSpan("doc-a.txt", 100, 120, "x" * 20),
        RetrievedSpan("doc-c.txt", 0, 10, "x" * 10),
    ]

    scores = score_retrieval(gold, pred)

    assert scores["matched_document_count"] == 1
    assert scores["retrieved_document_count"] == 2
    assert scores["gold_document_count"] == 2
    assert scores["document_precision"] == 0.5
    assert scores["document_recall"] == 0.5
    assert scores["document_f1"] == 0.5
    assert scores["f1"] == 0.0
