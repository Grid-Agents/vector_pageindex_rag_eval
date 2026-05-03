from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import GoldSpan, RetrievedSpan


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ranges = sorted((s, e) for s, e in ranges if e > s)
    if not sorted_ranges:
        return []
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def range_len(ranges: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def overlap_len(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def score_retrieval(
    gold_spans: list[GoldSpan], retrieved_spans: list[RetrievedSpan]
) -> dict[str, float | int]:
    gold_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pred_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in gold_spans:
        gold_by_doc[span.document_id].append((span.start_char, span.end_char))
    for span in retrieved_spans:
        pred_by_doc[span.document_id].append((span.start_char, span.end_char))

    gold_total = 0
    pred_total = 0
    overlap_total = 0
    for doc_id in set(gold_by_doc) | set(pred_by_doc):
        gold_ranges = merge_ranges(gold_by_doc.get(doc_id, []))
        pred_ranges = merge_ranges(pred_by_doc.get(doc_id, []))
        gold_total += range_len(gold_ranges)
        pred_total += range_len(pred_ranges)
        overlap_total += overlap_len(gold_ranges, pred_ranges)

    precision = overlap_total / pred_total if pred_total else 0.0
    recall = overlap_total / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    gold_doc_ids = set(gold_by_doc)
    pred_doc_ids = set(pred_by_doc)
    matched_doc_count = len(gold_doc_ids & pred_doc_ids)
    document_precision = (
        matched_doc_count / len(pred_doc_ids) if pred_doc_ids else 0.0
    )
    document_recall = matched_doc_count / len(gold_doc_ids) if gold_doc_ids else 0.0
    document_f1 = (
        2
        * document_precision
        * document_recall
        / (document_precision + document_recall)
        if document_precision + document_recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_chars": gold_total,
        "retrieved_chars": pred_total,
        "overlap_chars": overlap_total,
        "gold_document_count": len(gold_doc_ids),
        "retrieved_document_count": len(pred_doc_ids),
        "matched_document_count": matched_doc_count,
        "document_precision": document_precision,
        "document_recall": document_recall,
        "document_f1": document_f1,
        "gold_span_count": len(gold_spans),
        "retrieved_span_count": len(retrieved_spans),
    }
