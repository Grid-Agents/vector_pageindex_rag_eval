#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/default.yaml"
BENCHMARKS="cuad"
N="10"
METHODS="vector,pageindex"
CORPUS_SCOPE="all"
CHUNK_STRATEGY=""
TOP_K=""
RERANK_TOP_K=""
RUN_ID=""
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
  --methods LIST            Comma-separated methods. Default: vector,pageindex.
  --corpus-scope SCOPE      sampled or all. Default: all.
  --chunk-strategy NAME     hierarchical, recursive, or fixed.
  --top-k N                 Vector retrieval candidate top-k.
  --rerank-top-k N          Reranker output top-k.
  --answer-with-llm         Generate qualitative answers after retrieval. Saved as diagnostics only.
  --retrieval-only          Skip answer generation. Default and recommended for document metrics.
  --record-reasoning-trajectory
                            Save PageIndex document and ToC node selection traces. Default.
  --no-record-reasoning-trajectory
                            Disable PageIndex reasoning trace recording.
  --force-reindex           Rebuild PageIndex ToC cache.
  --run-id ID               Custom run id.
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
if [[ -n "$TOP_K" ]]; then
  cmd+=(--top-k "$TOP_K")
fi
if [[ -n "$RERANK_TOP_K" ]]; then
  cmd+=(--rerank-top-k "$RERANK_TOP_K")
fi
if [[ -n "$RUN_ID" ]]; then
  cmd+=(--run-id "$RUN_ID")
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
