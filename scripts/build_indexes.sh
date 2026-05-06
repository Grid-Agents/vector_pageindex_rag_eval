#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
BENCHMARKS="cuad"
N=""
METHODS="pageindex_official"
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
FORCE_REINDEX="false"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/build_indexes.sh [options] [-- extra build_indexes.py args]

Options:
  --benchmark NAME          Benchmark to build. Default: cuad.
  --benchmarks LIST         Comma-separated benchmarks, or all.
  --n N                     Sample size for sampled scope only.
  --methods LIST            Comma-separated methods: pageindex,vector,pageindex_official. Default: pageindex.
  --corpus-scope SCOPE      sampled or all. Default: all.
  --chunk-strategy NAME     hierarchical, recursive, fixed, or semantic.
  --search-strategy NAME    vector or hybrid.
  --batch-size N            Embedding batch size for vector RAG.
  --top-k N                 Vector retrieval candidate top-k.
  --rerank-top-k N          Reranker output top-k.
  --model-profile NAME      Vector model profile from config to build.
  --model-profiles LIST     Comma-separated vector model profiles from config to build.
  --embedding-provider NAME sentence_transformers or voyage.
  --embedding-model NAME    Embedding model for vector RAG.
  --reranker-provider NAME  sentence_transformers or voyage.
  --reranker-model NAME     Reranker model for vector RAG.
  --vector-combinations     Build vector chunk/search/model-profile combinations.
  --no-vector-combinations  Build one vector configuration only.
  --include-voyage          Add Voyage profile to vector combinations.
  --voyage-embedding-model  Voyage embedding model for --include-voyage.
  --voyage-reranker-model   Voyage reranker model for --include-voyage.
  --force-reindex           Rebuild the selected method caches.
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
    --force-reindex)
      FORCE_REINDEX="true"
      shift
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
  uv run python build_indexes.py
  --config "$CONFIG"
  --benchmark "$BENCHMARKS"
  --methods "$METHODS"
  --corpus-scope "$CORPUS_SCOPE"
)

if [[ -n "$N" ]]; then
  cmd+=(--n "$N")
fi
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
