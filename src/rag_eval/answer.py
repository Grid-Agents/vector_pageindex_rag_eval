from __future__ import annotations

from typing import Any

from .types import LLMResponse, RetrievedSpan, Usage


ANSWER_SYSTEM = """You answer legal benchmark questions using only the provided retrieved text.
If the retrieved text is insufficient, say that the answer is not supported by the retrieved snippets.
Be concise and do not invent citations."""


def answer_question(
    llm: Any,
    *,
    query: str,
    retrieved_spans: list[RetrievedSpan],
    max_context_chars: int,
) -> LLMResponse:
    if not retrieved_spans:
        return LLMResponse(
            text="The answer is not supported because no snippets were retrieved.",
            usage=Usage(),
        )

    blocks = []
    used = 0
    for idx, span in enumerate(retrieved_spans, start=1):
        text = span.text[: max(0, max_context_chars - used)]
        if not text:
            break
        blocks.append(
            f"[{idx}] {span.document_id} [{span.start_char}:{span.end_char}]\n{text}"
        )
        used += len(text)
        if used >= max_context_chars:
            break

    prompt = (
        f"Question:\n{query}\n\n"
        "Retrieved snippets:\n\n"
        + "\n\n".join(blocks)
        + "\n\nAnswer the question using only these snippets."
    )
    return llm.complete(system=ANSWER_SYSTEM, user=prompt)

