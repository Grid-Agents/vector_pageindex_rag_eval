from __future__ import annotations

import re

from .types import Chunk, Document


HEADING_RE = re.compile(
    r"""^\s*(
        (article|section|clause|schedule|exhibit|appendix)\s+[\w\dIVXLC().-]+
        |\d+(\.\d+)*[.)]?\s+[A-Z][A-Za-z0-9 ,;:'"()/&-]{3,}
        |[A-Z][A-Z0-9 ,;:'"()/&-]{8,}
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def make_chunks(
    document: Document,
    *,
    strategy: str = "hierarchical",
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(document, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return recursive_chunks(document, chunk_size, chunk_overlap)
    if strategy == "hierarchical":
        return hierarchical_chunks(document, chunk_size, chunk_overlap)
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

