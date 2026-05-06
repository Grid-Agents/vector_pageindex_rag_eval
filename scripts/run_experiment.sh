#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
BENCHMARKS="cuad"
N="10"
# METHODS="vector,pageindex,pageindex_official"
METHODS="vector"
CORPUS_SCOPE="all"
CHUNK_STRATEGY=""
SEARCH_STRATEGY=""
BATCH_SIZE=""
TOP_K=""
RERANK_TOP_K=""
MODEL_PROFILES=""
EMBEDDING_PROVIDER=""
EMBEDDING_MODEL=""
RERANKER_PROVIDER=""
RERANKER_MODEL=""
VECTOR_COMBINATIONS="false"
NO_VECTOR_COMBINATIONS="false"
INCLUDE_VOYAGE="false"
VOYAGE_EMBEDDING_MODEL=""
VOYAGE_RERANKER_MODEL=""
RUN_ID=""
MERGE_RESULTS="false"
MERGE_INTO_RUN="results/20260505T172718Z"
RETRIEVAL_ONLY="false"
ANSWER_WITH_LLM="false"
FORCE_REINDEX="false"
RECORD_REASONING_TRAJECTORY="true"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_experiment.sh [options] [-- extra run_experiment.py args]

Options:
  --benchmark NAME          Benchmark to run. Default: cuad.
  --benchmarks LIST         Comma-separated benchmarks, or all.
  --n N                     Number of examples. Default: 10.
  --methods LIST            Comma-separated methods: vector,pageindex,pageindex_official. Default: vector,pageindex.
  --corpus-scope SCOPE      sampled or all. Default: all.
  --chunk-strategy NAME     hierarchical, recursive, fixed, or semantic.
  --search-strategy NAME    vector or hybrid.
  --batch-size N            Embedding batch size for vector RAG.
  --top-k N                 Vector retrieval candidate top-k.
  --rerank-top-k N          Reranker output top-k.
  --model-profile NAME      Vector model profile from config to run.
  --model-profiles LIST     Comma-separated vector model profiles from config to run.
  --embedding-provider NAME sentence_transformers or voyage.
  --embedding-model NAME    Embedding model for vector RAG.
  --reranker-provider NAME  sentence_transformers or voyage.
  --reranker-model NAME     Reranker model for vector RAG.
  --vector-combinations     Evaluate vector chunk/search/model-profile combinations.
  --no-vector-combinations  Run one vector configuration only.
  --include-voyage          Add Voyage profile to vector combinations.
  --voyage-embedding-model  Voyage embedding model for --include-voyage.
  --voyage-reranker-model   Voyage reranker model for --include-voyage.
  --answer-with-llm         Generate qualitative answers after retrieval. Saved as diagnostics only.
  --retrieval-only          Skip answer generation. Default and recommended for document metrics.
  --record-reasoning-trajectory
                            Save PageIndex document and ToC node selection traces. Default.
  --no-record-reasoning-trajectory
                            Disable PageIndex reasoning trace recording.
  --force-reindex           Rebuild PageIndex ToC cache.
  --run-id ID               Custom run id.
  --merge-results           Merge this run into an existing --run-id directory.
  --combine-results         Alias for --merge-results.
  --merge-into-run ID|PATH  Merge this run into an existing run id or results path.
  --config PATH             Config file. Default: configs/default.yaml.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark|--benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --corpus-scope)
      CORPUS_SCOPE="$2"
      shift 2
      ;;
    --chunk-strategy)
      CHUNK_STRATEGY="$2"
      shift 2
      ;;
    --search-strategy)
      SEARCH_STRATEGY="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --rerank-top-k)
      RERANK_TOP_K="$2"
      shift 2
      ;;
    --model-profile|--model-profiles)
      MODEL_PROFILES="$2"
      shift 2
      ;;
    --embedding-provider)
      EMBEDDING_PROVIDER="$2"
      shift 2
      ;;
    --embedding-model)
      EMBEDDING_MODEL="$2"
      shift 2
      ;;
    --reranker-provider)
      RERANKER_PROVIDER="$2"
      shift 2
      ;;
    --reranker-model)
      RERANKER_MODEL="$2"
      shift 2
      ;;
    --vector-combinations)
      VECTOR_COMBINATIONS="true"
      shift
      ;;
    --no-vector-combinations)
      NO_VECTOR_COMBINATIONS="true"
      shift
      ;;
    --include-voyage)
      INCLUDE_VOYAGE="true"
      shift
      ;;
    --voyage-embedding-model)
      VOYAGE_EMBEDDING_MODEL="$2"
      shift 2
      ;;
    --voyage-reranker-model)
      VOYAGE_RERANKER_MODEL="$2"
      shift 2
      ;;
    --retrieval-only)
      RETRIEVAL_ONLY="true"
      shift
      ;;
    --answer-with-llm)
      ANSWER_WITH_LLM="true"
      shift
      ;;
    --record-reasoning-trajectory)
      RECORD_REASONING_TRAJECTORY="true"
      shift
      ;;
    --no-record-reasoning-trajectory)
      RECORD_REASONING_TRAJECTORY="false"
      shift
      ;;
    --force-reindex)
      FORCE_REINDEX="true"
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --merge-results|--combine-results)
      MERGE_RESULTS="true"
      shift
      ;;
    --merge-into-run)
      MERGE_INTO_RUN="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cmd=(
  uv run python run_experiment.py
  --config "$CONFIG"
  --benchmark "$BENCHMARKS"
  --n "$N"
  --methods "$METHODS"
  --corpus-scope "$CORPUS_SCOPE"
)

if [[ -n "$CHUNK_STRATEGY" ]]; then
  cmd+=(--chunk-strategy "$CHUNK_STRATEGY")
fi
if [[ -n "$SEARCH_STRATEGY" ]]; then
  cmd+=(--search-strategy "$SEARCH_STRATEGY")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$TOP_K" ]]; then
  cmd+=(--top-k "$TOP_K")
fi
if [[ -n "$RERANK_TOP_K" ]]; then
  cmd+=(--rerank-top-k "$RERANK_TOP_K")
fi
if [[ -n "$MODEL_PROFILES" ]]; then
  cmd+=(--model-profiles "$MODEL_PROFILES")
fi
if [[ -n "$EMBEDDING_PROVIDER" ]]; then
  cmd+=(--embedding-provider "$EMBEDDING_PROVIDER")
fi
if [[ -n "$EMBEDDING_MODEL" ]]; then
  cmd+=(--embedding-model "$EMBEDDING_MODEL")
fi
if [[ -n "$RERANKER_PROVIDER" ]]; then
  cmd+=(--reranker-provider "$RERANKER_PROVIDER")
fi
if [[ -n "$RERANKER_MODEL" ]]; then
  cmd+=(--reranker-model "$RERANKER_MODEL")
fi
if [[ "$VECTOR_COMBINATIONS" == "true" ]]; then
  cmd+=(--vector-combinations)
fi
if [[ "$NO_VECTOR_COMBINATIONS" == "true" ]]; then
  cmd+=(--no-vector-combinations)
fi
if [[ "$INCLUDE_VOYAGE" == "true" ]]; then
  cmd+=(--include-voyage)
fi
if [[ -n "$VOYAGE_EMBEDDING_MODEL" ]]; then
  cmd+=(--voyage-embedding-model "$VOYAGE_EMBEDDING_MODEL")
fi
if [[ -n "$VOYAGE_RERANKER_MODEL" ]]; then
  cmd+=(--voyage-reranker-model "$VOYAGE_RERANKER_MODEL")
fi
if [[ -n "$RUN_ID" ]]; then
  cmd+=(--run-id "$RUN_ID")
fi
if [[ "$MERGE_RESULTS" == "true" ]]; then
  cmd+=(--merge-results)
fi
if [[ -n "$MERGE_INTO_RUN" ]]; then
  cmd+=(--merge-into-run "$MERGE_INTO_RUN")
fi
if [[ "$RETRIEVAL_ONLY" == "true" ]]; then
  cmd+=(--retrieval-only)
fi
if [[ "$ANSWER_WITH_LLM" == "true" ]]; then
  cmd+=(--answer-with-llm)
fi
if [[ "$RECORD_REASONING_TRAJECTORY" == "true" ]]; then
  cmd+=(--record-reasoning-trajectory)
else
  cmd+=(--no-record-reasoning-trajectory)
fi
if [[ "$FORCE_REINDEX" == "true" ]]; then
  cmd+=(--force-reindex)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
