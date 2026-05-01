from rag_eval.text_splitters import make_chunks
from rag_eval.types import Document


def test_hierarchical_chunks_preserve_offsets():
    text = "SECTION 1\nAlpha beta gamma.\n\nSECTION 2\nDelta epsilon zeta."
    doc = Document("doc.txt", text)

    chunks = make_chunks(doc, strategy="hierarchical", chunk_size=30, chunk_overlap=0)

    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text

