from __future__ import annotations

from collections.abc import Callable
import re

import numpy as np

from .types import Chunk, Document


HEADING_RE = re.compile(
    r"""^\s*(
        (article|section|clause|schedule|exhibit|appendix)\s+[\w\dIVXLC().-]+
        |\d+(\.\d+)*[.)]?\s+[A-Z][A-Za-z0-9 ,;:'"()/&-]{3,}
        |[A-Z][A-Z0-9 ,;:'"()/&-]{8,}
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

EmbeddingFunction = Callable[[list[str]], np.ndarray]


def make_chunks(
    document: Document,
    *,
    strategy: str = "hierarchical",
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
    semantic_embedding_fn: EmbeddingFunction | None = None,
    semantic_break_percentile: float = 82.0,
    semantic_window_size: int = 3,
    semantic_min_chunk_size: int | None = None,
) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(document, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return recursive_chunks(document, chunk_size, chunk_overlap)
    if strategy == "hierarchical":
        return hierarchical_chunks(document, chunk_size, chunk_overlap)
    if strategy == "semantic":
        return semantic_chunks(
            document,
            chunk_size,
            chunk_overlap,
            embedding_fn=semantic_embedding_fn,
            break_percentile=semantic_break_percentile,
            window_size=semantic_window_size,
            min_chunk_size=semantic_min_chunk_size,
        )
    raise ValueError(f"Unknown chunk strategy: {strategy}")


def fixed_chunks(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(document.text):
        end = min(len(document.text), start + chunk_size)
        chunks.append(
            Chunk(
                document_id=document.document_id,
                start_char=start,
                end_char=end,
                text=document.text[start:end],
                title=f"chunk {idx}",
            )
        )
        if end == len(document.text):
            break
        start = max(start + 1, end - overlap)
        idx += 1
    return chunks


def recursive_chunks(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    spans = _recursive_spans(document.text, chunk_size, overlap)
    return [
        Chunk(
            document_id=document.document_id,
            start_char=start,
            end_char=end,
            text=document.text[start:end],
            title=f"chunk {idx}",
        )
        for idx, (start, end) in enumerate(spans)
    ]


def hierarchical_chunks(
    document: Document, chunk_size: int, overlap: int
) -> list[Chunk]:
    sections = detect_sections(document.text)
    if len(sections) <= 1:
        return recursive_chunks(document, chunk_size, overlap)

    chunks: list[Chunk] = []
    for section_idx, (title, start, end) in enumerate(sections):
        section_text = document.text[start:end]
        if len(section_text) <= chunk_size:
            chunks.append(
                Chunk(document.document_id, start, end, section_text, title, level=1)
            )
            continue
        for part_idx, (part_start, part_end) in enumerate(
            _recursive_spans(section_text, chunk_size, overlap)
        ):
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    start_char=start + part_start,
                    end_char=start + part_end,
                    text=document.text[start + part_start : start + part_end],
                    title=f"{title} / part {part_idx + 1}",
                    level=2,
                )
            )
    return chunks


def semantic_chunks(
    document: Document,
    chunk_size: int,
    overlap: int,
    *,
    embedding_fn: EmbeddingFunction | None,
    break_percentile: float = 82.0,
    window_size: int = 3,
    min_chunk_size: int | None = None,
) -> list[Chunk]:
    if embedding_fn is None:
        raise ValueError("semantic chunking requires semantic_embedding_fn")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    unit_spans = _semantic_unit_spans(document.text, max(200, chunk_size // 2))
    if len(unit_spans) <= 1:
        return recursive_chunks(document, chunk_size, overlap)

    embedded_texts = _semantic_window_texts(
        document.text, unit_spans, max(1, window_size)
    )
    unit_embeddings = np.asarray(embedding_fn(embedded_texts), dtype=np.float32)
    if unit_embeddings.ndim != 2 or unit_embeddings.shape[0] != len(unit_spans):
        raise ValueError("semantic_embedding_fn returned an unexpected shape")
    unit_embeddings = _normalize_rows(unit_embeddings)

    similarities = np.sum(unit_embeddings[:-1] * unit_embeddings[1:], axis=1)
    distances = 1.0 - similarities
    if len(distances) == 0 or float(np.ptp(distances)) <= 1e-6:
        break_after: set[int] = set()
    else:
        threshold = float(
            np.percentile(distances, min(99.0, max(1.0, break_percentile)))
        )
        break_after = {
            idx for idx, distance in enumerate(distances) if distance >= threshold
        }

    minimum = (
        max(1, int(min_chunk_size))
        if min_chunk_size is not None
        else max(1, int(chunk_size * 0.35))
    )
    overlap = max(0, min(overlap, chunk_size - 1))
    spans = _semantic_chunk_spans(unit_spans, break_after, chunk_size, overlap, minimum)

    return [
        Chunk(
            document_id=document.document_id,
            start_char=start,
            end_char=end,
            text=document.text[start:end],
            title=f"semantic chunk {idx}",
            level=1,
        )
        for idx, (start, end) in enumerate(spans)
    ]


def detect_sections(text: str) -> list[tuple[str, int, int]]:
    headings: list[tuple[str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and len(stripped) <= 140 and HEADING_RE.match(stripped):
            headings.append((stripped[:140], offset))
        offset += len(line)

    if not headings or headings[0][1] != 0:
        headings.insert(0, ("Document start", 0))

    sections: list[tuple[str, int, int]] = []
    for idx, (title, start) in enumerate(headings):
        end = headings[idx + 1][1] if idx + 1 < len(headings) else len(text)
        if end > start:
            sections.append((title, start, end))
    return sections


def _semantic_unit_spans(text: str, max_unit_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|$)", text):
        start, end = match.span()
        if end <= start:
            continue
        if end - start <= max_unit_chars:
            spans.append((start, end))
            continue
        for local_start, local_end in _recursive_spans(
            text[start:end], max_unit_chars, 0
        ):
            spans.append((start + local_start, start + local_end))
    return spans


def _semantic_window_texts(
    text: str, unit_spans: list[tuple[int, int]], window_size: int
) -> list[str]:
    radius = max(0, window_size // 2)
    windows: list[str] = []
    for idx in range(len(unit_spans)):
        left = max(0, idx - radius)
        right = min(len(unit_spans) - 1, idx + radius)
        windows.append(text[unit_spans[left][0] : unit_spans[right][1]])
    return windows


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _semantic_chunk_spans(
    unit_spans: list[tuple[int, int]],
    break_after: set[int],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start_idx = 0
    while start_idx < len(unit_spans):
        start_char = unit_spans[start_idx][0]
        end_idx = start_idx
        while end_idx < len(unit_spans):
            end_char = unit_spans[end_idx][1]
            current_size = end_char - start_char
            if end_idx == len(unit_spans) - 1:
                break
            if current_size >= chunk_size:
                break
            if end_idx in break_after and current_size >= min_chunk_size:
                break
            end_idx += 1

        end_char = unit_spans[end_idx][1]
        spans.append((start_char, end_char))
        if end_idx == len(unit_spans) - 1:
            break

        next_idx = end_idx + 1
        if overlap > 0:
            overlap_start = max(start_char + 1, end_char - overlap)
            while (
                next_idx > start_idx + 1
                and unit_spans[next_idx - 1][0] >= overlap_start
            ):
                next_idx -= 1
        start_idx = max(start_idx + 1, next_idx)
    return spans


def _recursive_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    spans: list[tuple[int, int]] = []
    start = 0
    min_end = max(1, int(chunk_size * 0.55))
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            best = -1
            for sep in separators:
                pos = text.rfind(sep, start + min_end, end)
                if pos > best:
                    best = pos + len(sep)
            if best > start:
                end = best
        spans.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return spans
