# LegalBench-RAG: Vector RAG vs PageIndex

This repo runs small or full LegalBench-RAG experiments comparing:

- `vector`: evaluates chunking/search variants by default, using local sentence-transformer embeddings and the open-source `BAAI/bge-reranker-v2-m3` reranker unless another model profile is selected.
- `pageindex`: a PageIndex-style semantic table-of-contents tree built over virtual pages, with Claude semanticizing every page/span at build time and then traversing the tree to page nodes at query time.
- `pageindex_official`: an adapter around VectifyAI's official self-hosted PageIndex repo. LegalBench-RAG documents are plain text, so the adapter converts each document into Markdown with section and virtual-page headings, calls the official `md_to_tree` implementation, then runs the official LLM tree-search pattern over the generated tree.
- `rlm`: an adapter around the official `alexzhang13/rlm` Recursive Language Model package. It now mirrors `rlm-eval`'s contract: RLM receives direct REPL access to the full `{document_id: text}` corpus, a `make_span(document_id, snippet)` helper for exact offsets, optional browse/search helper tools, and tolerant parsing of mixed prose / Python-literal final outputs.

The runner records retrieved spans, exact character-overlap retrieval metrics, document-level retrieval metrics, token usage, and estimated Claude cost in USD. Generated answers are optional and are not scored.

## Setup

```bash
cd vector_pageindex_rag_eval
uv sync --dev
cp .env.example .env
export ANTHROPIC_API_KEY=your_claude_api_key_here
# Optional, only needed when using --include-voyage or Voyage configs.
export VOYAGE_API_KEY=your_voyage_api_key_here
```

RLM support uses the official repo package and currently requires a Python version
compatible with that package:

```bash
uv sync --dev --extra rlm
# or install directly into the active environment:
uv pip install 'rlms @ git+https://github.com/alexzhang13/rlm.git'
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
  --search-strategy hybrid \
  --top-k 30 \
  --rerank-top-k 5 \
  --corpus-scope sampled
```

By default, vector runs expand into all configured vector combinations:

```text
chunk_strategies: hierarchical, recursive, fixed, semantic
search_strategies: vector, hybrid
model_profiles: bge
```

To run one specific new combination plus a Voyage comparison:

```bash
scripts/run_experiment.sh \
  --benchmark cuad \
  --n 5 \
  --methods vector \
  --chunk-strategy semantic \
  --search-strategy hybrid \
  --include-voyage
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

Run RLM as another retrieval method:

```bash
scripts/run_experiment.sh \
  --benchmark cuad \
  --n 5 \
  --methods vector,rlm
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

Open `results/visualization.html` to inspect run-level retrieval metrics, per-question gold snippets, retrieved spans, PageIndex reasoning traces, PageIndex ToC trees, and collapsed RLM turn trajectories under each RLM question result.

## Config

Edit `configs/default.yaml` for model and pipeline settings. The shared PageIndex LLM config currently defaults to `claude-haiku-4-5-20251001`; the RLM adapter defaults to `claude-sonnet-4-6` with `max_depth: 1`, `max_iterations: 20`, and `backend_max_tokens: 2048` to match `rlm-eval`. If Anthropic changes pricing or you use a different Claude model, update `llm.pricing`.

Vector RAG settings:

- `cache_dir`: persisted chunk metadata and embedding matrix cache
- `force_reindex`: rebuild cached vector embeddings
- `evaluate_combinations`: expand `vector` into chunk/search/model-profile variants
- `chunk_strategy`: `hierarchical`, `recursive`, `fixed`, or `semantic`
- `search_strategy`: `vector` or `hybrid`
- `hybrid`: BM25/vector weighting and BM25 constants
- `embedding_model`: default `BAAI/bge-large-en-v1.5`
- `reranker.model`: default `BAAI/bge-reranker-v2-m3`
- `model_profiles`: named embedding/reranker profiles; pass `--include-voyage` to append `voyage-4-large` + `rerank-2.5`

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

RLM settings:

- `backend`, `model_name`, `api_key_env`, `backend_kwargs`: official RLM model backend configuration
- `environment`: official RLM REPL environment, defaulting to `local`
- `max_iterations`, `max_depth`, `max_timeout`: RLM loop controls
- `backend_max_tokens`: backend response cap used for Anthropic runs when `backend_kwargs.max_tokens` is unset
- `selected_spans`, `max_retrieved_chars_per_span`: output span limits for this evaluator
- `record_reasoning_trajectory`: save turn count plus each RLM iteration's LLM output and REPL block output in `run.json` and the dashboard
- RLM sampling now uses the same `random.sample(seed)` behavior as `rlm-eval`, so `seed/n` subsets line up across the two repos

## Metrics

Retrieval metrics match LegalBench-RAG’s character-overlap spirit:

- Precision = overlapping retrieved characters / retrieved characters
- Recall = overlapping retrieved characters / gold characters
- F1 = harmonic mean of precision and recall

The runner also records document-level precision, recall, and F1 from unique retrieved `document_id` values. The benchmark score is based on retrieved character spans unless you explicitly choose to analyze document-level metrics.
