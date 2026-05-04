from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/VectifyAI/PageIndex.git"


@dataclass(frozen=True)
class OfficialPageIndexModules:
    md_to_tree: Any
    utils: types.ModuleType
    source: str
    repo_root: Path


def load_official_pageindex_modules(
    cfg: dict[str, Any], *, cache_dir: Path
) -> OfficialPageIndexModules:
    """Load the official PageIndex markdown implementation from a repo checkout.

    The upstream GitHub repo is not currently packaged as an installable Python
    project, while the PyPI `pageindex` package is the cloud SDK. Loading the
    upstream files directly keeps this adapter tied to the official self-hosted
    implementation without requiring users to manually vendor the repo.
    """

    repo_root = _resolve_repo_root(cfg, cache_dir=cache_dir)
    package_dir = repo_root / "pageindex"
    utils_path = package_dir / "utils.py"
    md_path = package_dir / "page_index_md.py"
    if not utils_path.exists() or not md_path.exists():
        raise RuntimeError(
            "Could not find official PageIndex markdown implementation at "
            f"{repo_root}. Expected pageindex/utils.py and pageindex/page_index_md.py."
        )

    _install_optional_dependency_stubs()
    runtime_name = _runtime_package_name(repo_root)
    package = types.ModuleType(runtime_name)
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[runtime_name] = package
    utils = _load_module(f"{runtime_name}.utils", utils_path)
    md_module = _load_module(f"{runtime_name}.page_index_md", md_path)
    return OfficialPageIndexModules(
        md_to_tree=md_module.md_to_tree,
        utils=utils,
        source=str(repo_root),
        repo_root=repo_root,
    )


def _resolve_repo_root(cfg: dict[str, Any], *, cache_dir: Path) -> Path:
    configured = (
        cfg.get("repo_path")
        or os.environ.get("RAG_EVAL_PAGEINDEX_REPO")
        or os.environ.get("PAGEINDEX_REPO_PATH")
    )
    if configured:
        return _normalize_repo_root(Path(str(configured)).expanduser())

    installed_root = _find_installed_self_hosted_package()
    if installed_root is not None:
        return installed_root

    if not bool(cfg.get("auto_clone_repo", True)):
        raise RuntimeError(
            "Official PageIndex repo not found. Set pageindex_official.repo_path "
            "or RAG_EVAL_PAGEINDEX_REPO, or enable auto_clone_repo."
        )

    repo_url = str(cfg.get("repo_url") or DEFAULT_REPO_URL)
    repo_ref = str(cfg.get("repo_ref") or "main")
    target = cache_dir / "_official_repo" / "PageIndex"
    if not (target / "pageindex" / "page_index_md.py").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(target)],
            check=True,
        )
    return target


def _normalize_repo_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "pageindex" / "page_index_md.py").exists():
        return path
    if (path / "page_index_md.py").exists() and path.name == "pageindex":
        return path.parent
    raise RuntimeError(
        f"{path} is not an official PageIndex repo root or pageindex package directory."
    )


def _find_installed_self_hosted_package() -> Path | None:
    spec = importlib.util.find_spec("pageindex")
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locations:
        return None
    package_dir = Path(next(iter(locations))).resolve()
    if (package_dir / "page_index_md.py").exists() and (package_dir / "utils.py").exists():
        return package_dir.parent
    return None


def _runtime_package_name(repo_root: Path) -> str:
    suffix = str(abs(hash(str(repo_root.resolve()))))
    return f"rag_eval.official_pageindex._official_runtime_{suffix}"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official PageIndex module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_optional_dependency_stubs() -> None:
    """Install tiny stubs for upstream imports unused by markdown indexing.

    The official markdown path imports `utils.py`, whose top-level imports include
    PDF and LiteLLM dependencies. This adapter patches the upstream LLM functions
    to use the repo's configured LLM, so the LiteLLM network functions are not
    needed for our benchmark path.
    """

    if not _module_available("litellm"):
        litellm = types.ModuleType("litellm")
        litellm.drop_params = True  # type: ignore[attr-defined]

        def token_counter(*, model: str | None = None, text: str = "", **_: Any) -> int:
            del model
            return max(1, (len(text or "") + 3) // 4)

        def completion(*_: Any, **__: Any) -> Any:
            raise RuntimeError("LiteLLM is not installed; LLM calls are patched by rag_eval.")

        async def acompletion(*_: Any, **__: Any) -> Any:
            raise RuntimeError("LiteLLM is not installed; LLM calls are patched by rag_eval.")

        litellm.token_counter = token_counter  # type: ignore[attr-defined]
        litellm.completion = completion  # type: ignore[attr-defined]
        litellm.acompletion = acompletion  # type: ignore[attr-defined]
        sys.modules["litellm"] = litellm

    if not _module_available("dotenv"):
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["dotenv"] = dotenv

    if not _module_available("PyPDF2"):
        pypdf2 = types.ModuleType("PyPDF2")

        class PdfReader:  # pragma: no cover - only used if caller asks for PDF mode.
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("PyPDF2 is required for official PageIndex PDF mode.")

        pypdf2.PdfReader = PdfReader  # type: ignore[attr-defined]
        sys.modules["PyPDF2"] = pypdf2

    if not _module_available("pymupdf"):
        pymupdf = types.ModuleType("pymupdf")
        pymupdf.open = _missing_pymupdf_open  # type: ignore[attr-defined]
        sys.modules["pymupdf"] = pymupdf


def _missing_pymupdf_open(*_: Any, **__: Any) -> Any:  # pragma: no cover
    raise RuntimeError("pymupdf is required for official PageIndex PDF mode.")


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None
