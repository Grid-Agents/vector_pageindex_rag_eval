# RAG Implementations

This repo compares two retrieval pipelines over LegalBench-RAG text files:

- `vector`: chunk text, embed chunks locally, search by cosine similarity, then optionally rerank.
- `pageindex`: build a semantic table-of-contents tree per document, then ask an LLM to navigate the tree and select relevant nodes.

Both implementations return the same `RetrievalOutput` shape: a list of `RetrievedSpan` objects with `document_id`, exact character offsets, retrieved text, score, and retriever metadata. The runner can then answer with an LLM and score retrieval with character-level overlap against LegalBench-RAG gold spans.

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
7. If `run.answer_with_llm` is true, `answer_question()` sends the retrieved snippets to the configured Claude model and asks it to answer using only those snippets.
8. Retrieval quality is scored by exact span overlap in `src/rag_eval/metrics.py`.

The important contract is character offsets. Retrieval is not just returning strings; every returned span maps back to `[start_char:end_char]` in a LegalBench-RAG corpus file.

## Text Chunking

Both pipelines use `make_chunks()` from `src/rag_eval/text_splitters.py`, but for different purposes.

Supported chunk strategies:

- `fixed`: slices every `chunk_size` characters with `chunk_overlap`.
- `recursive`: tries to end chunks on natural separators such as blank lines, newlines, sentence breaks, semicolons, commas, and spaces.
- `hierarchical`: detects legal-style headings first, then chunks within oversized sections.

`hierarchical` is the default. It detects headings with `HEADING_RE`, including patterns such as `Article ...`, `Section ...`, numbered headings, and all-caps headings. If section detection does not find useful structure, it falls back to recursive chunking.

Each `Chunk` keeps:

- `document_id`
- `start_char`
- `end_char`
- `text`
- `title`
- `level`

The vector retriever embeds these chunks directly. PageIndex uses larger no-overlap chunks as leaves in its fallback tree.

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
4. Loads a local SentenceTransformers embedder from `vector_rag.embedding_model`.
5. Embeds each chunk's retrieval text.
6. Stores normalized embeddings in a NumPy `float32` matrix.
7. Writes the chunk metadata and embedding matrix to the vector cache.
8. Optionally loads a SentenceTransformers `CrossEncoder` reranker.

The default config uses:

```yaml
vector_rag:
  cache_dir: .cache/vector
  force_reindex: false
  chunk_strategy: hierarchical
  chunk_size: 1200
  chunk_overlap: 120
  embedding_model: BAAI/bge-large-en-v1.5
  query_instruction: "Represent this sentence for searching relevant passages: "
  batch_size: 32
  top_k: 20
  reranker:
    enabled: true
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
- `embedding_model`
- embedding text format
- each indexed document's `document_id`, text length, and SHA-1 hash

The cache is reused only when all of those inputs match. Changing the corpus, chunking settings, or embedding model creates a different cache key. Set `vector_rag.force_reindex: true` or pass `--force-reindex` to rebuild the vector cache.

The cache skips the slow document embedding pass on later runs. The query embedder and optional reranker are still loaded because they are needed at query time.

### Query Step

`VectorRAG.query(query)` does the following:

1. Prefixes the query with `vector_rag.query_instruction`.
2. Embeds the query with normalized embeddings.
3. Computes dot product scores with the stored chunk matrix.
4. Because both document and query embeddings are normalized, dot product is cosine similarity.
5. Takes the top `vector_rag.top_k` candidate chunks.
6. If reranking is enabled, scores `(query, chunk_text)` pairs with the cross-encoder.
7. Returns the top `vector_rag.reranker.top_k` reranked chunks as `RetrievedSpan` objects.

Returned vector spans include metadata:

```json
{
  "retriever": "vector",
  "chunk_title": "...",
  "chunk_level": 1
}
```

Vector RAG has no LLM setup cost. It uses local embedding and reranking models, then optionally incurs answer-generation cost if `run.answer_with_llm` is enabled.

## PageIndex RAG

The PageIndex-style implementation is `PageIndexRAG` in `src/rag_eval/pageindex_rag.py`.

This is not a full upstream PageIndex PDF/Markdown index. It adapts the PageIndex idea to LegalBench-RAG raw text files:

- build a semantic tree of document regions;
- expose that tree as a compact table of contents;
- ask an LLM to select the smallest relevant nodes for a query;
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

If no valid cache exists, the system creates a fallback tree:

1. The document is chunked with `strategy="hierarchical"`.
2. Leaf size is controlled by `pageindex.max_leaf_chars`.
3. Leaf chunks have zero overlap.
4. Each leaf receives a stable node id based on a hash of the document id plus a numeric suffix.
5. Leaves are grouped into parent nodes of size `pageindex.group_size`.
6. A root node spans the whole document.

The default shape is:

```text
root document node
  group node covering about 8 leaves
    leaf node covering a legal section or recursive chunk
```

Every node stores:

- `node_id`
- `title`
- `summary`
- `start_char`
- `end_char`
- `children`

The fallback tree is already usable because titles and previews come from detected sections and text previews.

### LLM Semanticization

If `pageindex.build_with_llm` is true, `_semanticize_tree()` asks the configured LLM to improve the tree metadata.

The prompt sends up to `pageindex.max_build_nodes` flattened nodes. For each node, it includes:

- `node_id`
- current title
- character range
- a short text preview

The LLM must return JSON:

```json
{
  "document_summary": "...",
  "nodes": [
    {
      "node_id": "...",
      "title": "...",
      "summary": "..."
    }
  ]
}
```

The implementation updates only titles and summaries. It does not allow the LLM to change node ids or character ranges. This keeps retrieval grounded in original corpus offsets.

Setup token usage from tree semanticization is accumulated in `pageindex.setup_usage` and recorded separately from per-query usage in `run.json`.

If semanticization fails, the fallback tree is still cached and used. The build error is stored in tree metadata.

### Query Step

`PageIndexRAG.query(query)` does the following:

1. Formats all document trees into a "Semantic ToC forest".
2. Truncates that forest to `pageindex.max_tree_chars`.
3. Sends the query plus tree text to the LLM.
4. Asks the LLM to select up to `pageindex.selected_nodes` relevant `node_id` values.
5. Parses the JSON selection response.
6. Looks up each selected node in `node_index`.
7. Returns the node's original document character span.

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

At query time, `_format_forest()` prints every document tree into one text prompt. The formatted forest is grouped by document:

```text
DOCUMENT cuad/doc-a.txt
- doc-a-root [0:12000] ...
  - doc-a-g000 [0:5000] ...
    - doc-a-0000 [0:1200] ...

DOCUMENT cuad/doc-b.txt
- doc-b-root [0:9000] ...
  - doc-b-g000 [0:4800] ...
    - doc-b-0000 [0:1300] ...
```

The LLM sees the query plus this forest and chooses `node_id` values. There is no separate embedding lookup, metadata filter, or first-stage document classifier for PageIndex. The selected node id is the document decision.

After the LLM returns selections, the implementation resolves each selected node through `self.node_index`:

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

This means PageIndex document selection is prompt-based. The agent chooses the document by inspecting document headings, node titles, summaries, and ranges in the visible forest, then selecting nodes from the document that appears relevant.

One practical consequence is that `pageindex.max_tree_chars` matters a lot for multi-document runs. `_format_forest()` sorts documents by `document_id`, concatenates their trees, and truncates the resulting text if it exceeds the configured limit. If many documents are indexed, later document trees may be partially or fully omitted from the selection prompt. In that case, the LLM cannot choose nodes it never sees. `data.corpus_scope: sampled` keeps this cheap and usually small; `data.corpus_scope: all` is closer to full-corpus retrieval but increases the chance that the forest is truncated.

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

If selection or JSON parsing fails, PageIndex falls back to `_keyword_fallback()`. That fallback scores nodes by matching query terms longer than three characters against node titles and summaries.

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

Default PageIndex config:

```yaml
pageindex:
  cache_dir: .cache/pageindex
  force_reindex: false
  build_with_llm: true
  max_leaf_chars: 2400
  group_size: 8
  max_build_nodes: 80
  max_tree_chars: 24000
  selected_nodes: 5
  max_retrieved_chars_per_node: 5000
```

## Key Differences

Vector RAG retrieves by embedding similarity at chunk granularity. It is fast after local model loading and does not need an LLM to decide what to retrieve. Its main tuning knobs are chunking, candidate `top_k`, embedding model, and reranker model.

PageIndex retrieves by LLM navigation over document structure. It can choose larger semantic regions and use summaries/titles rather than only dense similarity. It has two LLM-dependent phases: optional build-time semanticization and query-time node selection. Caching is important because build-time semanticization can be expensive on large corpora.

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

For PageIndex runs, `run.json` also includes `toc_trees`, which is useful for inspecting what the LLM navigated during query-time selection.
