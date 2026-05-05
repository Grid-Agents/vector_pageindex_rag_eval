from __future__ import annotations

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any


CHUNK_STRATEGIES = ("hierarchical", "recursive", "fixed", "semantic")
SEARCH_STRATEGIES = ("vector", "hybrid")
MODEL_PROVIDERS = ("sentence_transformers", "voyage")
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-4-large"
DEFAULT_VOYAGE_RERANKER_MODEL = "rerank-2.5"


def add_vector_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument("--chunk-strategy", choices=CHUNK_STRATEGIES)
    parser.add_argument("--search-strategy", choices=SEARCH_STRATEGIES)
    parser.add_argument("--top-k", type=int, help="Vector retrieval candidate top-k.")
    parser.add_argument("--rerank-top-k", type=int, help="Reranker output top-k.")
    parser.add_argument(
        "--embedding-provider",
        choices=MODEL_PROVIDERS,
        help="Embedding provider for vector RAG.",
    )
    parser.add_argument("--embedding-model", help="Embedding model for vector RAG.")
    parser.add_argument(
        "--reranker-provider",
        choices=MODEL_PROVIDERS,
        help="Reranker provider for vector RAG.",
    )
    parser.add_argument("--reranker-model", help="Reranker model for vector RAG.")
    parser.add_argument(
        "--vector-combinations",
        action="store_true",
        help="Evaluate vector chunk/search/model-profile combinations.",
    )
    parser.add_argument(
        "--no-vector-combinations",
        action="store_true",
        help="Run a single vector configuration instead of vector combinations.",
    )
    parser.add_argument(
        "--include-voyage",
        action="store_true",
        help=(
            "Append a Voyage model profile to vector combinations "
            "(voyage-4-large + rerank-2.5 by default)."
        ),
    )
    parser.add_argument(
        "--voyage-embedding-model",
        default=DEFAULT_VOYAGE_EMBEDDING_MODEL,
        help="Voyage embedding model used by --include-voyage.",
    )
    parser.add_argument(
        "--voyage-reranker-model",
        default=DEFAULT_VOYAGE_RERANKER_MODEL,
        help="Voyage reranker model used by --include-voyage.",
    )


def apply_vector_cli_overrides(vector_cfg: dict[str, Any], args: Namespace) -> None:
    if getattr(args, "vector_combinations", False):
        vector_cfg["evaluate_combinations"] = True
    if getattr(args, "no_vector_combinations", False):
        vector_cfg["evaluate_combinations"] = False

    chunk_strategy = getattr(args, "chunk_strategy", None)
    if chunk_strategy:
        vector_cfg["chunk_strategy"] = chunk_strategy
        if vector_cfg.get("evaluate_combinations", False):
            vector_cfg["chunk_strategies"] = [chunk_strategy]

    search_strategy = getattr(args, "search_strategy", None)
    if search_strategy:
        vector_cfg["search_strategy"] = search_strategy
        if vector_cfg.get("evaluate_combinations", False):
            vector_cfg["search_strategies"] = [search_strategy]

    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        vector_cfg["top_k"] = top_k

    rerank_top_k = getattr(args, "rerank_top_k", None)
    if rerank_top_k is not None:
        vector_cfg.setdefault("reranker", {})["top_k"] = rerank_top_k
        for profile in vector_cfg.get("model_profiles") or []:
            if isinstance(profile, dict):
                profile.setdefault("reranker", {})["top_k"] = rerank_top_k

    _apply_model_override(vector_cfg, args)

    if getattr(args, "include_voyage", False):
        _append_voyage_profile(
            vector_cfg,
            embedding_model=str(
                getattr(args, "voyage_embedding_model", DEFAULT_VOYAGE_EMBEDDING_MODEL)
            ),
            reranker_model=str(
                getattr(args, "voyage_reranker_model", DEFAULT_VOYAGE_RERANKER_MODEL)
            ),
        )


def _apply_model_override(vector_cfg: dict[str, Any], args: Namespace) -> None:
    provider = getattr(args, "embedding_provider", None)
    model = getattr(args, "embedding_model", None)
    reranker_provider = getattr(args, "reranker_provider", None)
    reranker_model = getattr(args, "reranker_model", None)
    if not any((provider, model, reranker_provider, reranker_model)):
        return

    if provider:
        vector_cfg["embedding_provider"] = provider
        if provider == "voyage":
            vector_cfg["query_instruction"] = ""
    if model:
        vector_cfg["embedding_model"] = model
    if reranker_provider:
        vector_cfg.setdefault("reranker", {})["provider"] = reranker_provider
    if reranker_model:
        vector_cfg.setdefault("reranker", {})["model"] = reranker_model

    if vector_cfg.get("evaluate_combinations", False) or vector_cfg.get(
        "model_profiles"
    ):
        vector_cfg["model_profiles"] = [_base_model_profile(vector_cfg)]


def _append_voyage_profile(
    vector_cfg: dict[str, Any], *, embedding_model: str, reranker_model: str
) -> None:
    if not vector_cfg.get("evaluate_combinations", False):
        vector_cfg["evaluate_combinations"] = True
        vector_cfg["chunk_strategies"] = [
            vector_cfg.get("chunk_strategy", "hierarchical")
        ]
        vector_cfg["search_strategies"] = [vector_cfg.get("search_strategy", "vector")]

    profiles = deepcopy(
        vector_cfg.get("model_profiles") or [_base_model_profile(vector_cfg)]
    )
    voyage_profile = {
        "name": "voyage",
        "embedding_provider": "voyage",
        "embedding_model": embedding_model,
        "query_instruction": "",
        "reranker": {
            "enabled": True,
            "provider": "voyage",
            "model": reranker_model,
            "top_k": int(vector_cfg.get("reranker", {}).get("top_k", 5)),
        },
    }

    for idx, profile in enumerate(profiles):
        if isinstance(profile, dict) and profile.get("name") == "voyage":
            profiles[idx] = voyage_profile
            break
    else:
        profiles.append(voyage_profile)
    vector_cfg["model_profiles"] = profiles


def _base_model_profile(vector_cfg: dict[str, Any]) -> dict[str, Any]:
    embedding_provider = vector_cfg.get("embedding_provider", "sentence_transformers")
    embedding_model = vector_cfg.get("embedding_model")
    return {
        "name": _profile_name(str(embedding_provider), str(embedding_model)),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "query_instruction": vector_cfg.get("query_instruction", ""),
        "reranker": deepcopy(vector_cfg.get("reranker", {})),
    }


def _profile_name(provider: str, model: str) -> str:
    lowered = f"{provider} {model}".lower()
    if "voyage" in lowered:
        return "voyage"
    if "bge" in lowered:
        return "bge"
    return model.rsplit("/", 1)[-1] or provider
