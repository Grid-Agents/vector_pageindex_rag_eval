from __future__ import annotations

import json
import random
from pathlib import Path

from .types import Document, Example, GoldSpan


class LegalBenchRAGLoader:
    """Load LegalBench-RAG's `data/corpus` and `data/benchmarks/*.json` layout."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.corpus_dir = self.data_dir / "corpus"
        self.benchmarks_dir = self.data_dir / "benchmarks"

    def validate_layout(self) -> None:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Missing LegalBench-RAG corpus dir: {self.corpus_dir}")
        if not self.benchmarks_dir.exists():
            raise FileNotFoundError(
                f"Missing LegalBench-RAG benchmarks dir: {self.benchmarks_dir}"
            )

    def benchmark_names(self) -> list[str]:
        self.validate_layout()
        return sorted(path.stem for path in self.benchmarks_dir.glob("*.json"))

    def load_examples(
        self,
        benchmarks: list[str] | None = None,
        *,
        n: int | None = None,
        seed: int = 42,
    ) -> list[Example]:
        self.validate_layout()
        names = self._resolve_benchmark_names(benchmarks)
        examples: list[Example] = []
        for name in names:
            path = self.benchmarks_dir / f"{name}.json"
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                tests = raw.get("tests", [])
            elif isinstance(raw, list):
                tests = raw
            else:
                tests = []
            if not isinstance(tests, list):
                raise ValueError(f"Benchmark JSON has no test list: {path}")
            for idx, test in enumerate(tests):
                snippets = test.get("snippets", [])
                gold_spans = []
                for snippet in snippets:
                    start, end = snippet["span"]
                    gold_spans.append(
                        GoldSpan(
                            document_id=snippet["file_path"],
                            start_char=int(start),
                            end_char=int(end),
                        )
                    )
                examples.append(
                    Example(
                        example_id=str(test.get("id", f"{name}:{idx}")),
                        benchmark=name,
                        query=str(test["query"]),
                        gold_spans=gold_spans,
                        tags=list(test.get("tags", [])) or [name],
                    )
                )

        rng = random.Random(seed)
        if n is None:
            return examples
        if n > len(examples):
            raise ValueError(f"requested n={n} but only {len(examples)} examples available")
        return rng.sample(examples, n)

    def load_documents(
        self,
        examples: list[Example],
        *,
        corpus_scope: str = "sampled",
    ) -> list[Document]:
        self.validate_layout()
        if corpus_scope not in {"sampled", "all"}:
            raise ValueError("corpus_scope must be 'sampled' or 'all'")

        if corpus_scope == "all":
            doc_ids = {
                span.document_id for example in examples for span in example.gold_spans
            }
            benchmarks = sorted({example.benchmark for example in examples})
            for benchmark in benchmarks:
                benchmark_dir = self.corpus_dir / benchmark
                if not benchmark_dir.exists():
                    continue
                doc_ids.update(
                    str(path.relative_to(self.corpus_dir))
                    for path in benchmark_dir.rglob("*")
                    if path.is_file()
                )
            doc_ids = sorted(doc_ids)
        else:
            doc_ids = sorted(
                {
                    span.document_id
                    for example in examples
                    for span in example.gold_spans
                }
            )

        documents = [self.load_document(doc_id) for doc_id in doc_ids]
        doc_by_id = {doc.document_id: doc for doc in documents}
        for example in examples:
            hydrated_spans = []
            for span in example.gold_spans:
                doc = doc_by_id.get(span.document_id)
                text = ""
                if doc is not None:
                    text = doc.text[span.start_char : span.end_char]
                hydrated_spans.append(
                    GoldSpan(
                        document_id=span.document_id,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        text=text,
                    )
                )
            example.gold_spans = hydrated_spans
        return documents

    def load_document(self, document_id: str) -> Document:
        path = self.corpus_dir / document_id
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus document: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return Document(document_id=document_id, text=f.read())

    def _resolve_benchmark_names(self, benchmarks: list[str] | None) -> list[str]:
        available = self.benchmark_names()
        if not benchmarks or benchmarks == ["all"]:
            return available
        missing = sorted(set(benchmarks) - set(available))
        if missing:
            raise FileNotFoundError(
                f"Unknown benchmark(s) {missing}; available: {available}"
            )
        return benchmarks
