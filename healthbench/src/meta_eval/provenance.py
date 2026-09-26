"""Deterministic fingerprints for code, data, configuration, and model revisions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .config import ExperimentConfig


DEFAULT_RUNTIME_DISTRIBUTIONS = (
    "PyYAML",
    "adapt",
    "cvxopt",
    "joblib",
    "keras",
    "numpy",
    "pandas",
    "pymanopt",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "tabpfn",
    "tensorflow",
    "torch",
    "transformers",
)


def resolve_experiment_input_files(
    config: ExperimentConfig,
    project_root: Path,
) -> Dict[str, Path]:
    """Resolve the data files that define one experiment's run identity."""

    root = Path(project_root).resolve()
    files: Dict[str, Path] = {}

    def resolved(raw_path: Any) -> Path:
        path = Path(str(raw_path))
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    split_metadata = config.dataset.get("split_metadata_path")
    if split_metadata:
        files["split_metadata"] = resolved(split_metadata)

    rubric_path = config.dataset.get("rubric_path")
    if rubric_path:
        files["rubric"] = resolved(rubric_path)

    prompt_path = config.dataset.get("prompt_path")
    if prompt_path:
        files["prompt"] = resolved(prompt_path)

    input_path = config.dataset.get("input_path")
    if input_path:
        files["input"] = resolved(input_path)

    score_dir_raw = config.dataset.get("score_dir")
    if score_dir_raw:
        score_dir = resolved(score_dir_raw)
        if not score_dir.is_dir():
            raise FileNotFoundError(score_dir)
        score_paths = sorted(score_dir.glob("*.jsonl"))
        if not score_paths:
            raise FileNotFoundError(f"No score JSONL files found in {score_dir}")
        for score_path in score_paths:
            files[f"score_{score_path.name}"] = score_path

    for index, representation in enumerate(config.representations):
        for field in ("embedding_path", "metadata_path"):
            raw_path = representation.get(field)
            if raw_path:
                files[f"representation_{index}_{field}"] = resolved(raw_path)

    for index, method in enumerate(config.methods):
        checkpoint_env = method.get("checkpoint_env")
        if checkpoint_env:
            raw_path = os.environ.get(str(checkpoint_env))
            if not raw_path:
                raise FileNotFoundError(
                    f"Method {method.get('name', index)} requires environment variable {checkpoint_env}."
                )
            files[f"method_{index}_checkpoint"] = resolved(raw_path)

    upstream_filenames = (
        "manifest.json",
        "_SUCCESS",
        "config.resolved.yaml",
        "split.jsonl",
        "split_theme_counts.csv",
        "estimates.csv",
        "weights.csv",
        "diagnostics.csv",
        "balance.csv",
        "missingness.csv",
        "summary.json",
    )
    if config.extra.get("runner") == "repeated_covariate_shift":
        upstream_filenames = (
            *upstream_filenames,
            "representation_summary.csv",
        )
    for index, upstream in enumerate(config.extra.get("upstream_runs", ())):
        run_dir = resolved(upstream["run_dir"])
        selected_filenames = upstream.get("input_files", upstream_filenames)
        if not isinstance(selected_filenames, (tuple, list)):
            raise ValueError("upstream input_files must be a list.")
        for filename in selected_filenames:
            filename = str(filename)
            key = re.sub(r"[^0-9A-Za-z]+", "_", filename).strip("_").lower()
            files[f"upstream_{index}_{key}"] = run_dir / filename

    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n" + "\n".join(missing))
    return files


def _json_value(value: Any) -> Any:
    if isinstance(value, ExperimentConfig):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Fingerprint inputs cannot contain non-finite floats.")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(project_root: Path, relative_paths: Sequence[Path]) -> str:
    """Hash selected files/directories by relative name and content."""

    root = Path(project_root).resolve()
    files = []

    def is_relevant_file(path: Path) -> bool:
        relative_parts = path.relative_to(root).parts
        if any(part == "__pycache__" or part.endswith(".egg-info") for part in relative_parts):
            return False
        return path.suffix not in {".pyc", ".pyo"} and path.name != ".DS_Store"

    for raw_path in relative_paths:
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Code path escapes project root: {raw_path}") from error
        if not candidate.exists():
            continue
        if candidate.is_file():
            if is_relevant_file(candidate):
                files.append(candidate)
        else:
            files.extend(path for path in candidate.rglob("*") if path.is_file() and is_relevant_file(path))

    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root: Path, args: Sequence[str]) -> Optional[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def git_state(project_root: Path) -> Dict[str, Any]:
    commit = _git_value(project_root, ["rev-parse", "HEAD"])
    status = _git_value(project_root, ["status", "--porcelain=v1", "--untracked-files=normal"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": sha256_bytes((status or "").encode("utf-8")),
    }


def relevant_git_diff_state(
    project_root: Path,
    relative_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Hash tracked changes and untracked files under the fingerprinted code paths."""

    root = Path(project_root).resolve()
    path_args = [Path(path).as_posix() for path in relative_paths]
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            "--no-renames",
            "--no-color",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "HEAD",
            "--",
            *path_args,
        ],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *path_args,
        ],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return {
            "dirty": False,
            "diff_sha256": sha256_bytes(b""),
            "untracked_files": [],
        }

    untracked_paths = [Path(raw.decode("utf-8")) for raw in untracked.stdout.split(b"\0") if raw]
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff.stdout)
    digest.update(b"untracked-files\0")
    for relative in sorted(untracked_paths, key=lambda path: path.as_posix()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Untracked code path escapes project root: {relative}") from error
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "dirty": bool(diff.stdout or untracked_paths),
        "diff_sha256": digest.hexdigest(),
        "untracked_files": [path.as_posix() for path in sorted(untracked_paths)],
    }


def hash_named_files(project_root: Path, files: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    root = Path(project_root).resolve()
    output = {}
    for name, raw_path in sorted(files.items()):
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        output[str(name)] = {
            "path": resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    return output


def runtime_dependency_lock(
    dependency_versions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Content-address the exact Python/runtime packages used by an experiment."""

    if dependency_versions is None:
        versions = {}
        for distribution in DEFAULT_RUNTIME_DISTRIBUTIONS:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = "not-installed"
    else:
        versions = {str(name): str(version) for name, version in sorted(dependency_versions.items())}

    snapshot = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": versions,
    }
    return {
        "kind": "resolved-runtime-snapshot",
        "sha256": sha256_bytes(canonical_json(snapshot).encode("utf-8")),
        "snapshot": snapshot,
    }


def _score_model_name(path: Path) -> str:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get("model")
            if not model:
                raise ValueError(f"{path}:{line_number}: missing model identity.")
            return str(model)
    raise ValueError(f"{path}: no score rows.")


def artifact_model_revisions(
    config: ExperimentConfig,
    data_files: Mapping[str, Path],
    hashed_data: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """Resolve model identities to upstream or content-addressed artifact revisions."""

    revisions: Dict[str, str] = {}
    for index, representation in enumerate(config.representations):
        input_name = f"representation_{index}_embedding_path"
        display_name = str(representation["name"])
        model_id = str(representation.get("model_id", display_name))
        revision = representation.get("revision")
        if revision:
            resolved_revision = str(revision)
        elif input_name in hashed_data:
            resolved_revision = f"artifact-sha256:{hashed_data[input_name]['sha256']}"
        else:
            continue
        revisions[f"embedding_model:{display_name}"] = f"{model_id}@{resolved_revision}"

    for input_name, path in sorted(data_files.items()):
        if not input_name.startswith("score_"):
            continue
        model_name = _score_model_name(path)
        revisions[f"generation_score:{model_name}"] = f"artifact-sha256:{hashed_data[input_name]['sha256']}"

    judge_model_id = config.dataset.get("judge_model_id")
    judge_model_revision = config.dataset.get("judge_model_revision")
    if judge_model_id or judge_model_revision:
        if not judge_model_id or not judge_model_revision:
            raise ValueError("judge_model_id and judge_model_revision must be provided together.")
        revisions["judge_model"] = f"{judge_model_id}@{judge_model_revision}"

    for method in config.methods:
        model_id = method.get("model_id")
        model_revision = method.get("model_revision")
        if model_id and model_revision:
            revisions[f"prediction_model:{method['name']}"] = f"{model_id}@{model_revision}"
    return revisions


def compute_run_fingerprint(identity: Mapping[str, Any]) -> str:
    """Hash only deterministic identity fields, never timestamps or host names."""

    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def build_run_provenance(
    config: ExperimentConfig,
    project_root: Path,
    *,
    data_files: Optional[Mapping[str, Path]] = None,
    model_revisions: Optional[Mapping[str, str]] = None,
    code_paths: Optional[Iterable[Path]] = None,
    dependency_versions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build deterministic provenance and the corresponding run identifier."""

    root = Path(project_root).resolve()
    selected_code_paths = (
        tuple(code_paths)
        if code_paths is not None
        else (
            Path("src/meta_eval"),
            Path("pyproject.toml"),
        )
    )
    repository_state = git_state(root)
    relevant_diff_state = relevant_git_diff_state(root, selected_code_paths)
    observed_code_tree_sha256 = tree_digest(root, selected_code_paths)
    configured_source_control = config.extra.get("source_control")
    if configured_source_control:
        configured_code_tree_sha256 = str(configured_source_control["code_tree_sha256"])
        if observed_code_tree_sha256 != configured_code_tree_sha256:
            raise ValueError(
                "Deployed code tree does not match the source_control snapshot: "
                f"expected {configured_code_tree_sha256}, got {observed_code_tree_sha256}."
            )
        identity_git_commit = configured_source_control.get("git_commit")
        identity_git_worktree = {
            "dirty": bool(configured_source_control["dirty"]),
            "diff_sha256": str(configured_source_control["diff_sha256"]),
            "untracked_files": list(configured_source_control.get("untracked_files", ())),
        }
        identity_code_tree_sha256 = configured_code_tree_sha256
    else:
        identity_git_commit = repository_state["commit"]
        identity_git_worktree = relevant_diff_state
        identity_code_tree_sha256 = observed_code_tree_sha256
    hashed_data = hash_named_files(root, data_files or {})
    resolved_model_revisions = (
        dict(model_revisions)
        if model_revisions is not None
        else artifact_model_revisions(config, data_files or {}, hashed_data)
    )
    configured_dependency_versions = config.extra.get("runtime_versions")
    if dependency_versions is None and configured_dependency_versions:
        dependency_versions = {
            str(name): str(version) for name, version in configured_dependency_versions.items()
        }
    configured_score_route = config.extra.get("score_route")
    score_route = (
        dict(configured_score_route)
        if configured_score_route
        else {
            "kind": "precomputed-content-addressed-score-artifacts",
            "judge_invoked_by_this_run": False,
            "upstream_judge_revision_recorded": "judge_model" in resolved_model_revisions,
        }
    )
    identity = {
        "provenance_schema_version": 2,
        "config": config.to_dict(),
        "git_commit": identity_git_commit,
        "git_worktree": identity_git_worktree,
        "code_tree_sha256": identity_code_tree_sha256,
        "dependency_lock": runtime_dependency_lock(dependency_versions),
        "data": hashed_data,
        "model_revisions": dict(sorted(resolved_model_revisions.items())),
        "score_route": score_route,
    }
    fingerprint = compute_run_fingerprint(identity)
    return {
        "fingerprint": fingerprint,
        "run_id": f"{config.experiment_id}-{fingerprint[:12]}",
        "identity": identity,
        "observed_repository_state": {
            **repository_state,
            "relevant_git_worktree": relevant_diff_state,
            "code_tree_sha256": observed_code_tree_sha256,
        },
    }
