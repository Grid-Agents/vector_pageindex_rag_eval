# LegalBench-RAG: Vector RAG vs PageIndex

This repo runs small or full LegalBench-RAG experiments comparing:

- `vector`: hierarchical chunking by default, local sentence-transformer embeddings, and the open-source `BAAI/bge-reranker-v2-m3` reranker.
- `pageindex`: a PageIndex-style semantic table-of-contents tree built over virtual pages, with Claude semanticizing every page/span at build time and then traversing the tree to page nodes at query time.
- `pageindex_official`: an adapter around VectifyAI's official self-hosted PageIndex repo. LegalBench-RAG documents are plain text, so the adapter converts each document into Markdown with section and virtual-page headings, calls the official `md_to_tree` implementation, then runs the official LLM tree-search pattern over the generated tree.

The runner records retrieved spans, exact character-overlap retrieval metrics, document-level retrieval metrics, token usage, and estimated Claude cost in USD. Generated answers are optional and are not scored.

## Setup

```bash
cd vector_pageindex_rag_eval
uv sync --dev
cp .env.example .env
export ANTHROPIC_API_KEY=your_claude_api_key_here
```

Download LegalBench-RAG into `data/` with:

```bash
mkdir -p data
curl -L 'https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1' -o /tmp/legalbenchrag.zip
unzip -q /tmp/legalbenchrag.zip -d data
```

This will create:

```text
vector_pageindex_rag_eval/
  data/
    corpus/
      ...
    benchmarks/
      privacy_qa.json
      contractnli.json
      maud.json
      cuad.json
```

LegalBench-RAG benchmark JSON files contain `tests`; each test has a `query` and `snippets` with `file_path` plus `[start, end]` character spans into `data/corpus`.

## Run

By default, `configs/default.yaml` runs the CUAD benchmark (`data/benchmarks/cuad.json`). Override it with `--benchmark` for one benchmark, `--benchmarks` for a comma-separated list, or `all` for every benchmark in `data/benchmarks`.

Cheap smoke run on two sampled CUAD questions:

```bash
uv run python run_experiment.py --n 2 --methods vector,pageindex
```

Run a different benchmark:

```bash
uv run python run_experiment.py --benchmark maud --n 10 --methods vector
```

Useful overrides:

```bash
uv run python run_experiment.py \
  --n 20 \
  --benchmarks cuad,maud \
  --methods vector,pageindex \
  --chunk-strategy hierarchical \
  --top-k 30 \
  --rerank-top-k 5 \
  --corpus-scope sampled
```

You can also use the shell wrapper:

```bash
scripts/run_experiment.sh --benchmark cuad --n 5 --methods vector,pageindex
```

Compare both PageIndex implementations:

```bash
scripts/run_experiment.sh \
  --benchmark cuad \
  --n 5 \
  --methods vector,pageindex,pageindex_official
```

Runs are retrieval-only by default. Pass `--answer-with-llm` only when you want qualitative answers saved alongside the retrieval results; those answers are not part of precision, recall, or F1.

If you only want to warm caches without running retrieval, use the build-only script:

```bash
scripts/build_indexes.sh --methods pageindex --benchmark cuad --corpus-scope all
scripts/build_indexes.sh --methods pageindex_official --benchmark cuad --corpus-scope all
scripts/build_indexes.sh --methods vector --benchmark cuad --corpus-scope all
scripts/build_indexes.sh --methods vector,pageindex --benchmark cuad --corpus-scope all --force-reindex
```

Run tests with:

```bash
uv run pytest
```

Use `--corpus-scope all` when you want retrieval over the full selected benchmark corpus instead of only the documents needed by the sampled gold spans. That is closer to the full benchmark, but slower and more expensive for PageIndex indexing.

## Results

Each run writes:

```text
results/<run_id>/run.json
results/<run_id>/results.jsonl
results/<run_id>/summary.csv
results/visualization.html
```

Open `results/visualization.html` to inspect run-level retrieval metrics, per-question gold snippets, retrieved spans, PageIndex reasoning traces, and PageIndex ToC trees for both the self-implemented and official methods.

## Config

Edit `configs/default.yaml` for model and pipeline settings. Claude defaults to `claude-sonnet-4-6` with the current public API price class of `$3/M` input tokens and `$15/M` output tokens. If Anthropic changes pricing or you use a different Claude model, update `llm.pricing`.

Vector RAG settings:

- `cache_dir`: persisted chunk metadata and embedding matrix cache
- `force_reindex`: rebuild cached vector embeddings
- `chunk_strategy`: `hierarchical`, `recursive`, or `fixed`
- `embedding_model`: default `BAAI/bge-large-en-v1.5`
- `reranker.model`: default `BAAI/bge-reranker-v2-m3`

PageIndex settings:

- `cache_dir`: semantic ToC cache
- `build_with_llm`: use Claude to semanticize virtual pages and parent spans
- `virtual_page_target_tokens`: target size of each synthetic page
- `virtual_page_max_tokens`: hard cap for each synthetic page
- `toc_check_units`: how many virtual pages to scan for an existing ToC or early heading structure
- `max_units_per_node`: max virtual-page span for a non-root ToC node
- `max_tokens_per_node`: max estimated token span for a non-root ToC node
- `selected_documents`: max documents shortlisted before per-document node selection
- `selected_nodes`: max final page nodes selected during tree traversal
- `max_retrieved_chars_per_node`: retrieval span cap for returned page nodes
- `record_reasoning_trajectory`: save the PageIndex document-selection and ToC tree-walk decision path in `run.json`

Official PageIndex settings:

- `cache_dir`: official PageIndex tree cache and generated Markdown files
- `repo_path`: optional local checkout of `https://github.com/VectifyAI/PageIndex`; if empty, the adapter auto-clones the repo into the cache
- `build_with_llm`: patch official summary-generation calls through this repo's configured LLM so setup usage is collected in the normal run output
- `virtual_page_target_tokens` / `virtual_page_max_tokens`: virtual page size used when converting LegalBench text into Markdown for the official Markdown implementation
- `if_add_node_summary` / `if_add_doc_description`: official `md_to_tree` summary and document-description options
- `selected_documents`, `selected_nodes`, `max_tree_chars`, `max_retrieved_chars_per_node`, `record_reasoning_trajectory`: query-time retrieval and trace settings, matching the existing PageIndex method

## Metrics

Retrieval metrics match LegalBench-RAG’s character-overlap spirit:

- Precision = overlapping retrieved characters / retrieved characters
- Recall = overlapping retrieved characters / gold characters
- F1 = harmonic mean of precision and recall

The runner also records document-level precision, recall, and F1 from unique retrieved `document_id` values. The benchmark score is based on retrieved character spans unless you explicitly choose to analyze document-level metrics.
