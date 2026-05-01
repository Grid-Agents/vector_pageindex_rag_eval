import json

from rag_eval.data import LegalBenchRAGLoader


def test_legalbench_loader_reads_examples_and_sampled_docs(tmp_path):
    data_dir = tmp_path / "data"
    corpus_dir = data_dir / "corpus"
    bench_dir = data_dir / "benchmarks"
    corpus_dir.mkdir(parents=True)
    bench_dir.mkdir(parents=True)
    (corpus_dir / "contract.txt").write_text("abcdefghij", encoding="utf-8")
    (bench_dir / "cuad.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "query": "What letters?",
                        "snippets": [{"file_path": "contract.txt", "span": [2, 5]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loader = LegalBenchRAGLoader(data_dir)
    examples = loader.load_examples(["cuad"], n=1, seed=1)
    documents = loader.load_documents(examples, corpus_scope="sampled")

    assert examples[0].query == "What letters?"
    assert examples[0].gold_spans[0].text == "cde"
    assert documents[0].document_id == "contract.txt"

