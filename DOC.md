# RAG Evaluation Overview

This repository evaluates retrieval quality on LegalBench-RAG. The goal is to compare different RAG retrieval approaches on the same legal questions, the same source documents, and the same gold-standard evidence spans.

The evaluation is retrieval-first: it asks, "Did the system find the right source text?" Generated answers can be produced for qualitative review, but they are not part of the core score.

## 1. Benchmark Data

We use LegalBench-RAG, a benchmark for legal retrieval. It provides two pieces:

- `data/corpus/`: the legal source documents, stored as text files.
- `data/benchmarks/`: benchmark question files, one JSON file per task.

Download the data into this repo with:

```bash
mkdir -p data
curl -L 'https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1' -o /tmp/legalbenchrag.zip
unzip -q /tmp/legalbenchrag.zip -d data
```

Expected layout:

```text
data/
  corpus/
    contractnli/
    cuad/
    maud/
    privacy_qa/
  benchmarks/
    contractnli.json
    cuad.json
    maud.json
    privacy_qa.json
```

The benchmark files in this repo currently contain:

| Benchmark | Questions | What it covers |
| --- | ---: | --- |
| `cuad` | 4,042 | Contract clauses and agreement questions |
| `maud` | 1,676 | M&A agreement questions |
| `contractnli` | 977 | Contract natural-language inference questions |
| `privacy_qa` | 194 | Privacy policy questions |

Each benchmark example has:

- a `query`, which is the user question;
- one or more `snippets`, which point to the correct evidence;
- each snippet's `file_path`, which identifies the corpus document;
- each snippet's `[start, end]` character span, which marks the exact gold evidence inside the document.

That character span is important: the benchmark does not only say which document is correct. It says exactly which part of the document should have been retrieved.

## 2. RAG Systems Compared

The repo compares four retrieval approaches. They all return the same output shape: a list of retrieved document spans with document ID, character offsets, text, score, and metadata.

### Vector RAG

Vector RAG breaks documents into chunks, embeds those chunks, searches for chunks similar to the question, and optionally reranks the best candidates.

Configurable dimensions include:

- **Chunking strategy**
  - `hierarchical`: uses legal-style headings when available, then chunks within sections.
  - `semantic`: uses embeddings to find natural topic boundaries.
  - `recursive`: splits near natural text separators such as paragraphs and sentences.
  - `fixed`: simple fixed-size character windows.
- **Search strategy**
  - `vector`: dense embedding similarity.
  - `hybrid`: combines dense similarity with BM25 keyword matching.
- **Model profile**
  - `bge`: local Sentence Transformers embeddings with `BAAI/bge-large-en-v1.5` and reranking with `BAAI/bge-reranker-v2-m3`.
  - `voyage`: Voyage embeddings and reranking. The checked-in default config uses `voyage-law-2` and `rerank-2`; the CLI also supports appending Voyage defaults with `voyage-4-large` and `rerank-2.5`.

The current `configs/default.yaml` is set up to evaluate Voyage-based vector variants with hierarchical and semantic chunking, hybrid search, `top_k: 20`, and reranked output `top_k: 5`.

### PageIndex RAG

PageIndex builds a semantic table-of-contents tree over each document. Instead of searching every chunk directly, an LLM navigates the tree:

1. choose likely relevant documents;
2. inspect each selected document's table of contents;
3. walk down to the most relevant page-level nodes;
4. return those nodes as character spans from the original document.

This approach spends more work during indexing because it can ask Claude to summarize virtual pages and tree nodes. The benefit is that retrieval can use a document-level structure rather than only local chunk similarity.

### Official PageIndex Adapter

`pageindex_official` adapts VectifyAI's official self-hosted PageIndex implementation to LegalBench-RAG's plain-text corpus. The adapter converts each text document into Markdown with section and virtual-page headings, builds the official PageIndex tree, and uses the official LLM tree-search pattern to retrieve nodes.

This gives a closer comparison against the upstream PageIndex approach while keeping the same benchmark and scoring pipeline.

### RLM Adapter

`rlm` adapts the official `alexzhang13/rlm` Recursive Language Model package. It gives the RLM REPL tools to list, search, and read LegalBench source documents, then requires a final JSON response with exact `document_id`, `start_char`, and `end_char` span selections.

When enabled, the runner stores the RLM turn count, final output, and each official RLM iteration's LLM output plus REPL code/stdout/stderr. The dashboard shows this under each RLM question result as a collapsed reasoning trajectory.

## 3. Evaluation Pipeline

The main entrypoint is:

```bash
uv run python run_experiment.py
```

At a high level, each run does this:

1. Load benchmark questions from `data/benchmarks/*.json`.
2. Load the matching source documents from `data/corpus/`.
3. Build the selected retrieval systems once over those documents.
4. For each benchmark question, ask each retrieval system to return relevant spans.
5. Compare the retrieved spans with the benchmark's gold spans.
6. Write per-question results, summary metrics, costs, and an HTML dashboard.

## 4. Evaluation Metrics

The primary score is span-level F1. This measures whether the retrieved text overlaps the exact gold evidence span.

### Span-Level Metrics

For each question, the evaluator compares:

- **Gold characters**: all benchmark-provided evidence characters.
- **Retrieved characters**: all characters returned by the retriever.
- **Overlap characters**: retrieved characters that overlap the gold evidence.

Then it computes:

- **Precision** = overlapping retrieved characters / retrieved characters.
- **Recall** = overlapping retrieved characters / gold characters.
- **F1** = the balance between precision and recall.

Interpretation:

- High precision means the retriever returned mostly relevant text.
- High recall means the retriever found most of the gold evidence.
- High F1 means it did both well.

Example: if the gold answer span is 1,000 characters and the system retrieves 1,500 characters, with 750 characters overlapping the gold span:

- precision = 750 / 1,500 = 0.50
- recall = 750 / 1,000 = 0.75
- F1 = 0.60

This is stricter than document-level retrieval because it rewards finding the right passage, not just the right file.

### Document-Level Metrics

The runner also reports document-level precision, recall, and F1. These only check whether the retrieved document IDs match the gold document IDs.

Document-level metrics are useful for diagnosing routing:

- If document-level recall is high but span-level recall is low, the system usually found the right document but the wrong passage.
- If document-level recall is low, the system is failing earlier by looking in the wrong documents.

The aggregate report treats span-level F1 as the primary metric because the benchmark is about finding the exact supporting evidence.

## 5. Note

The best method is not always the one with the highest raw F1. For a production decision, compare quality, latency, setup cost, query cost, and operational complexity together.
