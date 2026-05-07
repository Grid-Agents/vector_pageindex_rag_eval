# RAG Implementations

This repo compares retrieval pipelines over LegalBench-RAG text files:

- `vector`: chunk text, embed chunks, search by dense vector or hybrid BM25+dense scores, then optionally rerank.
- `pageindex`: build a semantic table-of-contents tree per document, then ask an LLM to navigate the tree and select relevant nodes.
- `pageindex_official`: convert LegalBench text into Markdown, build the tree with VectifyAI's official self-hosted PageIndex Markdown implementation, then use the official LLM tree-search pattern to select nodes.
- `rlm`: call the official `alexzhang13/rlm` Recursive Language Model package with REPL tools for searching and reading the loaded LegalBench corpus, then parse the RLM's final JSON span selections.

All implementations return the same `RetrievalOutput` shape: a list of `RetrievedSpan` objects with `document_id`, exact character offsets, retrieved text, score, and retriever metadata. The runner scores retrieval with character-level overlap against LegalBench-RAG gold spans, plus document-level hit metrics. Optional generated answers are saved only for qualitative review.

## Shared Data Flow

The experiment entrypoint is `run_experiment()` in `src/rag_eval/runner.py`.

1. `LegalBenchRAGLoader` reads benchmark examples from `data/benchmarks/*.json`.
2. Each example contains a query and gold snippets with `file_path` plus `[start, end]` character spans.
3. The loader reads source documents from `data/corpus`.
4. `data.corpus_scope` controls which documents are indexed:
   - `sampled`: only documents referenced by the sampled examples' gold spans.
   - `all`: every file under `data/corpus`.
5. The selected retrieval methods are built once over the loaded documents.
6. For each query, every method returns retrieved character spans.
7. If `run.answer_with_llm` is true, `answer_question()` sends the retrieved snippets to the configured Claude model and asks it to answer using only those snippets. This is disabled by default and is not scored.
8. Retrieval quality is scored by exact span overlap in `src/rag_eval/metrics.py`.

The important contract is character offsets. Retrieval is not just returning strings; every returned span maps back to `[start_char:end_char]` in a LegalBench-RAG corpus file.

## Text Chunking

Both pipelines use `make_chunks()` from `src/rag_eval/text_splitters.py`, but for different purposes.

Supported chunk strategies:

- `fixed`: slices every `chunk_size` characters with `chunk_overlap`.
- `recursive`: tries to end chunks on natural separators such as blank lines, newlines, sentence breaks, semicolons, commas, and spaces.
- `hierarchical`: detects legal-style headings first, then chunks within oversized sections.
- `semantic`: embeds paragraph/sentence units, breaks on low-cohesion boundaries, and still enforces `chunk_size` plus `chunk_overlap`.

`hierarchical` is the default. It detects headings with `HEADING_RE`, including patterns such as `Article ...`, `Section ...`, numbered headings, and all-caps headings. If section detection does not find useful structure, it falls back to recursive chunking.

Each `Chunk` keeps:

- `document_id`
- `start_char`
- `end_char`
- `text`
- `title`
- `level`

The vector retriever embeds these chunks directly. PageIndex instead creates synthetic "virtual pages" over the raw text and uses those page nodes as the base unit of its ToC tree.

## Vector RAG

The vector implementation is `VectorRAG` in `src/rag_eval/vector_rag.py`.

### Build Step

`VectorRAG.build(documents)` does the following:

1. Computes a cache key from document content and index-building settings.
2. Reuses a cached vector index if one exists and `vector_rag.force_reindex` is false.
3. Otherwise chunks every loaded document using:
   - `vector_rag.chunk_strategy`
   - `vector_rag.chunk_size`
   - `vector_rag.chunk_overlap`
4. Loads an embedder from `vector_rag.embedding_provider` and `vector_rag.embedding_model`.
5. Embeds each chunk's retrieval text.
6. Stores normalized embeddings in a NumPy `float32` matrix.
7. Writes the chunk metadata and embedding matrix to the vector cache.
8. Optionally builds a BM25 index for `search_strategy: hybrid`.
9. Optionally loads a reranker from `vector_rag.reranker.provider` and `vector_rag.reranker.model`.

The default config uses:

```yaml
vector_rag:
  cache_dir: .cache/vector
  force_reindex: false
  chunk_strategy: hierarchical
  search_strategy: vector
  evaluate_combinations: true
  chunk_strategies: [hierarchical, recursive, fixed, semantic]
  search_strategies: [vector, hybrid]
  chunk_size: 1200
  chunk_overlap: 120
  embedding_provider: sentence_transformers
  embedding_model: BAAI/bge-large-en-v1.5
  query_instruction: "Represent this sentence for searching relevant passages: "
  batch_size: 32
  top_k: 20
  reranker:
    enabled: true
    provider: sentence_transformers
    model: BAAI/bge-reranker-v2-m3
    top_k: 5
```

The embedded text for a chunk is:

```text
<chunk title>
<chunk text>
```

If the chunk has no title, only the chunk text is embedded. Including the title gives the embedding model local section context.

### Vector Index Cache

The vector index is persisted under `vector_rag.cache_dir`, which defaults to:

```text
.cache/vector
```

Each cached index has two files:

```text
.cache/vector/<cache-key>.json
.cache/vector/<cache-key>.npz
```

The JSON file stores index metadata, document fingerprints, and chunk offsets/titles. The NPZ file stores the normalized embedding matrix. Chunk text is reconstructed from the already-loaded source documents using the cached character offsets.

The cache key includes:

- cache format version
- `chunk_strategy`
- `chunk_size`
- `chunk_overlap`
- semantic chunking settings
- `embedding_provider`
- `embedding_model`
- `embedding_output_dimension`
- embedding text format
- each indexed document's `document_id`, text length, and SHA-1 hash

The cache is reused only when all of those inputs match. Changing the corpus, chunking settings, embedding provider, or embedding model creates a different cache key. Search strategy is intentionally not part of the embedding cache key, so vector and hybrid variants can share the same chunk embeddings. Set `vector_rag.force_reindex: true` or pass `--force-reindex` to rebuild the vector cache.

The cache skips the slow document embedding pass on later runs. The query embedder and optional reranker are still loaded because they are needed at query time.

### Query Step

`VectorRAG.query(query)` does the following:

1. Prefixes the query with `vector_rag.query_instruction`.
2. Embeds the query with normalized embeddings.
3. Computes dot product scores with the stored chunk matrix.
4. Because both document and query embeddings are normalized, dot product is cosine similarity.
5. For `search_strategy: hybrid`, also scores chunks with BM25 and combines min-max-normalized dense and BM25 scores.
6. Takes the top `vector_rag.top_k` candidate chunks.
7. If reranking is enabled, scores candidates with the configured reranker.
8. Returns the top `vector_rag.reranker.top_k` reranked chunks as `RetrievedSpan` objects.

Returned vector spans include metadata:

```json
{
  "retriever": "vector",
  "search_strategy": "hybrid",
  "chunk_title": "...",
  "chunk_level": 1,
  "vector_score": 0.72,
  "bm25_score": 4.1,
  "hybrid_score": 0.86
}
```

`run.methods: ["vector"]` expands to named vector variants when `vector_rag.evaluate_combinations` is true. The default config evaluates the cross product of four chunkers (`hierarchical`, `recursive`, `fixed`, `semantic`) and two search strategies (`vector`, `hybrid`) for the `bge` model profile. Pass `--include-voyage` to append a Voyage profile using `voyage-4-large` embeddings and `rerank-2.5` reranking; this requires `VOYAGE_API_KEY`.

Vector RAG has no LLM setup cost. It uses local embedding and reranking models, then incurs answer-generation cost only if `run.answer_with_llm` is enabled.

## PageIndex RAG

The PageIndex-style implementation is `PageIndexRAG` in `src/rag_eval/pageindex_rag.py`.

This is not a full upstream PageIndex PDF/Markdown index. It adapts the PageIndex idea to LegalBench-RAG raw text files:

- build a semantic tree over virtual pages that cover the full document;
- expose that tree as a compact table of contents;
- ask an LLM to choose relevant documents, then select the smallest relevant nodes within those document trees;
- return the selected nodes' original character spans.

### Build Step

`PageIndexRAG.build(documents)` builds one tree per document and indexes every node by `node_id`.

For each document, `_load_or_build_tree()` first checks the cache:

```yaml
pageindex:
  cache_dir: .cache/pageindex
  force_reindex: false
```

The cache filename is based on the document id, document length, and SHA-1 hash of document text. If the document content changes, the cache key changes.

If no valid cache exists, the system creates a virtual-page tree:

1. The document is partitioned into contiguous `virtual pages` under `pageindex.virtual_page_target_tokens` and `pageindex.virtual_page_max_tokens`.
2. Virtual page boundaries are aligned to detected section boundaries when possible, so a page does not usually cross a heading-defined section.
3. The first `pageindex.toc_check_units` virtual pages are scanned for an existing ToC or early heading structure.
4. Each virtual page becomes a leaf node with a stable `node_id`.
5. Contiguous leaf pages are grouped into non-root span nodes with caps from `pageindex.max_units_per_node` and `pageindex.max_tokens_per_node`.
6. A root node spans the whole document.

The default shape is:

```text
root document node
  span node covering up to about 10 virtual pages
    page node covering one virtual page
```

Every node stores:

- `node_id`
- `title`
- `summary`
- `start_char`
- `end_char`
- `unit_start`
- `unit_end`
- `token_count`
- `children`

The heuristic tree is already usable because titles come from detected sections and summaries come from page/span previews.

### LLM Semanticization

If `pageindex.build_with_llm` is true, the builder semanticizes the tree bottom-up:

1. Every virtual page leaf is sent to the LLM with its full page text.
2. The LLM returns a short page title and summary.
3. Each non-root span node is then summarized from its child node titles and summaries.
4. The root node is summarized from its top-level children.

This means the LLM sees the full document content, but never all at once. It reads every page leaf directly, then works upward from child summaries for broader nodes.

Setup token usage from tree semanticization is accumulated in `pageindex.setup_usage` and recorded separately from per-query usage in `run.json`.

If semanticization fails, the fallback tree is still cached and used. The build error is stored in tree metadata.

### Query Step

`PageIndexRAG.query(query)` does the following:

1. Formats a compact document catalog using each document's summary plus representative section titles.
2. Sends the query plus document catalog to the LLM.
3. Asks the LLM to choose up to `pageindex.selected_documents` relevant `document_id` values.
4. For each selected document, starts at the root and shows only that node's immediate children.
5. The LLM selects relevant child nodes.
6. For every selected non-leaf node, the retriever descends another level and repeats the child selection step.
7. Traversal continues until it reaches page leaves.
8. The selected page nodes are returned using their original document character spans.

### Multi-Document Routing

PageIndex does not build one global merged tree. It builds one independent tree per loaded document:

```python
self.documents = {doc.document_id: doc for doc in documents}

for doc in documents:
    tree, usage = self._load_or_build_tree(doc)
    self.trees[doc.document_id] = tree
    self._index_tree(doc.document_id, tree)
```

`self.trees` keeps the per-document roots:

```text
document_id -> tree
```

`self.node_index` is the global routing table:

```text
node_id -> (document_id, node)
```

At query time, PageIndex first builds a compact document catalog:

```text
- cuad/doc-a.txt: summary... | sections: Grant of License; Liability; Insurance
- cuad/doc-b.txt: summary... | sections: Term; Confidentiality; Indemnification
```

The LLM sees the query plus this catalog and chooses `document_id` values. After that, PageIndex traverses each chosen document tree level by level instead of dumping the entire tree into one prompt.

After the LLM returns node selections, the implementation resolves each selected node through `self.node_index`:

```python
doc_id, node = self.node_index[node_id]
doc = self.documents[doc_id]
```

The returned `RetrievedSpan` then uses that `doc_id` and the selected node's original character range:

```python
RetrievedSpan(
    document_id=doc_id,
    start_char=node["start_char"],
    end_char=node["end_char"],
    text=doc.text[start:end],
)
```

This means PageIndex document selection is still prompt-based, but node routing is now a staged tree walk: root children first, then narrower descendants, until page nodes are selected.

One practical consequence is that `pageindex.max_tree_chars` now applies only to the current child list shown at a traversal step, not to the whole document tree. This lets PageIndex handle much longer documents without hiding later sections behind one monolithic prompt.

The selection prompt asks for:

```json
{
  "selections": [
    {
      "node_id": "...",
      "reason": "..."
    }
  ]
}
```

If document selection or node selection fails, PageIndex falls back to keyword scoring. Document fallback matches query terms against document ids, summaries, and representative section titles. Node fallback prefers page leaves and scores titles/summaries within the current document or subtree.

Returned PageIndex spans include metadata:

```json
{
  "retriever": "pageindex",
  "node_id": "...",
  "node_title": "...",
  "reason": "..."
}
```

Broad selected nodes can cover too much text, so each returned node span is capped by `pageindex.max_retrieved_chars_per_node`. The span always starts at the selected node's `start_char`; only `end_char` may be shortened by the cap.

When `pageindex.record_reasoning_trajectory` is true, each PageIndex result also stores a `reasoning_trajectory` object. The trace records the document shortlist response, accepted document selections, each staged ToC child-selection step, fallback decisions, and the final node ids that became retrieved spans. This is an observable decision path from the model's JSON rationales and the tree nodes shown in each prompt, not hidden chain-of-thought.

Default PageIndex config:

```yaml
pageindex:
  cache_dir: .cache/pageindex
  force_reindex: false
  build_with_llm: true
  virtual_page_target_tokens: 900
  virtual_page_max_tokens: 1200
  toc_check_units: 20
  max_units_per_node: 10
  max_tokens_per_node: 20000
  node_summary_max_tokens: 220
  root_summary_max_tokens: 260
  selected_documents: 3
  max_tree_chars: 24000
  selected_nodes: 5
  max_retrieved_chars_per_node: 5000
  record_reasoning_trajectory: true
  reasoning_max_catalog_chars: 12000
  reasoning_max_node_summary_chars: 320
```

## Key Differences

Vector RAG retrieves by embedding similarity at chunk granularity. It is fast after local model loading and does not need an LLM to decide what to retrieve. Its main tuning knobs are chunking, candidate `top_k`, embedding model, and reranker model.

PageIndex retrieves by LLM navigation over document structure. It can choose larger semantic regions and use summaries/titles rather than only dense similarity. It has two LLM-dependent phases: optional build-time semanticization and query-time node selection. Caching is important because build-time semanticization can be expensive on large corpora.

Official PageIndex uses the same runner contract but a different tree builder. `OfficialPageIndexRAG` lives under `src/rag_eval/official_pageindex`, auto-loads or clones `https://github.com/VectifyAI/PageIndex`, and calls the upstream `md_to_tree` function. Since LegalBench-RAG documents are text files rather than PDFs, it first emits Markdown headings for detected sections and virtual pages so the upstream Markdown parser can build a hierarchy. Query-time retrieval follows the official cookbook pattern: send the question plus the PageIndex tree to the LLM and ask for relevant `node_id` values. Returned spans are still mapped back to original LegalBench character offsets.

## RLM RAG

The RLM implementation is `RLMRAG` in `src/rag_eval/rlm_rag.py`.

It uses the official `alexzhang13/rlm` package rather than vendoring RLM code. Install that package from the official repo before selecting `rlm`:

```bash
uv sync --dev --extra rlm
# or:
uv pip install 'rlms @ git+https://github.com/alexzhang13/rlm.git'
```

At build time, RLM stores only the loaded `Document` objects and a compact document catalog. There is no persistent embedding or tree index.

At query time, the adapter creates an official `RLM` with an `RLMLogger` and custom REPL tools:

- `list_documents(limit, offset)`: inspect ids, lengths, and short previews
- `search_documents(query, limit, per_document_limit, window_chars)`: keyword-search full documents and return candidate windows with exact offsets
- `read_document_span(document_id, start_char, end_char)`: read exact source text for a proposed span
- `get_document_length(document_id)`: inspect document length before expanding or clipping spans

The RLM now follows the same access pattern as `rlm-eval`: it receives
`context["query"]` plus the full `context["corpus"] = {document_id: text}`
mapping in the REPL, along with helper tools for browsing/searching windows.
It also receives `make_span(document_id, snippet)`, which computes exact
character offsets from an exact snippet match.

The root prompt instructs it to return JSON:

```json
{
  "spans": [
    {
      "document_id": "...",
      "start_char": 123,
      "end_char": 456,
      "snippet": "exact substring from the source document"
    }
  ]
}
```

The adapter accepts clean JSON, JSON embedded in surrounding prose, and
Python-literal dict output. When a span includes `snippet`, the adapter will
repair bad offsets from that snippet when possible. It still validates
document ids, clamps offsets to the source document, caps spans with
`rlm.max_retrieved_chars_per_span`, and returns normal `RetrievedSpan`
objects. If the RLM returns document-level selections instead of spans, the
adapter maps each selected document to the first configured span-length
window so the evaluator still has a character range.

When `rlm.record_reasoning_trajectory` is true, each result stores a normalized `reasoning_trajectory` with:

- `turn_count`: number of official RLM iterations
- `llm_call_count`: total model calls reported by the official usage summary
- `final_response`: the final JSON or text response from RLM
- `iterations`: each turn's LLM output plus REPL code/stdout/stderr/final value
- `retrieved_spans`: final normalized spans returned to the evaluator

The dashboard renders this as a collapsed "RLM reasoning trajectory" block under each RLM method panel.

## Scoring and Output

The runner writes:

- `results/<run_id>/run.json`
- `results/<run_id>/results.jsonl`
- `results/<run_id>/summary.csv`
- `results/visualization.html`

Retrieval metrics are computed from character ranges:

- precision = overlapping retrieved characters / retrieved characters
- recall = overlapping retrieved characters / gold characters
- F1 = harmonic mean of precision and recall

Document-level metrics are also recorded from unique `document_id` matches:

- document_precision = matched gold documents / retrieved documents
- document_recall = matched gold documents / gold documents
- document_F1 = harmonic mean of document precision and recall

Generated answers are not part of any aggregate score.

For PageIndex runs, `run.json` also includes `toc_trees`, tagged by method, which is useful for inspecting what the LLM navigated during query-time selection.
