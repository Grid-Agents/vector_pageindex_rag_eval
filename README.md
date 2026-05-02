# LegalBench-RAG: Vector RAG vs PageIndex

This repo runs small or full LegalBench-RAG experiments comparing:

- `vector`: hierarchical chunking by default, local sentence-transformer embeddings, and the open-source `BAAI/bge-reranker-v2-m3` reranker.
- `pageindex`: a PageIndex-style semantic table-of-contents tree built with Claude, followed by Claude reasoning over the tree at query time.

The runner records retrieved spans, answers, exact character-overlap retrieval metrics, token usage, and estimated Claude cost in USD.

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

Run tests with:

```bash
uv run pytest
```

Use `--corpus-scope all` when you want retrieval over the full corpus instead of only the documents needed by the sampled gold spans. That is closer to the full benchmark, but slower and more expensive for PageIndex indexing.

## Results

Each run writes:

```text
results/<run_id>/run.json
results/<run_id>/results.jsonl
results/<run_id>/summary.csv
results/visulization.html
results/visualization.html
```

Open `results/visulization.html` to inspect run-level metrics, per-question qualitative answers, retrieved spans, and the PageIndex ToC tree tab.

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
- `build_with_llm`: use Claude to summarize/semanticize ToC nodes
- `selected_nodes`: max nodes selected during tree reasoning
- `max_retrieved_chars_per_node`: retrieval span cap for broad nodes

## Metrics

Retrieval metrics match LegalBench-RAG’s character-overlap spirit:

- Precision = overlapping retrieved characters / retrieved characters
- Recall = overlapping retrieved characters / gold characters
- F1 = harmonic mean of precision and recall

Answers are saved for qualitative review; the benchmark score is based on retrieved character spans.
