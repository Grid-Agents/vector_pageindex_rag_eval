#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
BENCHMARKS="cuad"
N=""
METHODS="pageindex_official"
CORPUS_SCOPE="all"
CHUNK_STRATEGY=""
TOP_K=""
RERANK_TOP_K=""
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
  --chunk-strategy NAME     hierarchical, recursive, or fixed.
  --top-k N                 Vector retrieval candidate top-k.
  --rerank-top-k N          Reranker output top-k.
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
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --rerank-top-k)
      RERANK_TOP_K="$2"
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
if [[ -n "$TOP_K" ]]; then
  cmd+=(--top-k "$TOP_K")
fi
if [[ -n "$RERANK_TOP_K" ]]; then
  cmd+=(--rerank-top-k "$RERANK_TOP_K")
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
