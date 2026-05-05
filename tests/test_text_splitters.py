import numpy as np

from rag_eval.text_splitters import make_chunks
from rag_eval.types import Document


def test_hierarchical_chunks_preserve_offsets():
    text = "SECTION 1\nAlpha beta gamma.\n\nSECTION 2\nDelta epsilon zeta."
    doc = Document("doc.txt", text)

    chunks = make_chunks(doc, strategy="hierarchical", chunk_size=30, chunk_overlap=0)

    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_semantic_chunks_preserve_offsets():
    text = (
        "Alpha beta gamma.\n\n"
        "Alpha beta delta.\n\n"
        "Warranties indemnities liability."
    )
    doc = Document("doc.txt", text)

    def embed(texts):
        vectors = []
        for item in texts:
            if "Warranties" in item:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)

    chunks = make_chunks(
        doc,
        strategy="semantic",
        chunk_size=500,
        chunk_overlap=0,
        semantic_embedding_fn=embed,
        semantic_break_percentile=50,
        semantic_min_chunk_size=1,
    )

    assert len(chunks) == 2
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
