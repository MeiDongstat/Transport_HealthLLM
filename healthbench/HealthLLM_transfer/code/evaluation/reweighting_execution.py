"""Shared lifecycle utilities for repeated HealthBench reweighting runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_TIMINGS: ContextVar[list[float] | None] = ContextVar("split_timings", default=None)


@dataclass(frozen=True)
class SplitWeights:
    """Return method-by-source weights and split-level diagnostics."""

    raw_weights: np.ndarray
    weights: np.ndarray
    diagnostics: Mapping[str, np.ndarray | float] = field(default_factory=dict)


@dataclass(frozen=True)
class RepeatedBatch:
    """Store source-aligned outputs for a contiguous set of repeated splits."""

    split_indices: np.ndarray
    raw_weights: np.ndarray
    weights: np.ndarray
    n_source_complete: np.ndarray
    reweighted_scores: np.ndarray
    diagnostics: Mapping[str, np.ndarray]


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive mapping merge with lists replaced as complete values."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    """Return a YAML configuration with an optional relative base configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config
    resolved_base = Path(base_path)
    if not resolved_base.is_absolute():
        resolved_base = config_path.parent / resolved_base
    return _merge_config(load_config(resolved_base), config)


def score_path_keys(config: Mapping[str, Any]) -> list[str]:
    """Return the path keys of score files used by the configured experiments."""
    if "score_sets" in config:
        return list(dict.fromkeys(spec["path_key"] for spec in config["score_sets"].values()))
    return ["scores"]


def select_score_set(
    config: Mapping[str, Any],
    name: str,
    *,
    dataset_keys: tuple[str, str] = ("score_key", "target_score_key"),
) -> dict[str, Any]:
    """Return a single-score experiment using the named source and target fields."""
    selected = dict(config)
    spec = selected.pop("score_sets")[name]
    selected["paths"] = {**config["paths"], "scores": config["paths"][spec["path_key"]]}
    selected["dataset"] = {
        **config["dataset"],
        dataset_keys[0]: spec["score_key"],
        dataset_keys[1]: spec["target_score_key"],
    }
    return selected


def resolve_path(raw_path: str | Path) -> Path:
    """Return a path resolved from the HealthLLM_transfer root."""
    path = Path(raw_path)
    return path if path.is_absolute() else HEALTHLLM_TRANSFER_ROOT / path


def named(config: Mapping[str, Any], section: str, name: str) -> Mapping[str, Any]:
    """Return one named YAML list entry."""
    return {str(entry["name"]): entry for entry in config[section]}[name]


def feature_specs(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return ordered feature specifications with unique names."""
    specifications = tuple(config["feature_sets"])
    names = tuple(str(item["name"]) for item in specifications)
    if len(names) != len(set(names)):
        raise ValueError("Feature-set names must be unique")
    return specifications


def configured_split_count(config: Mapping[str, Any]) -> int:
    """Return the YAML-defined repeated-split count."""
    return int(config["execution"]["n_splits"])


def configured_batch_count(config: Mapping[str, Any]) -> int:
    """Return the number of YAML-defined split batches."""
    n_splits = configured_split_count(config)
    batch_size = int(config["execution"]["batch_size"])
    return (n_splits + batch_size - 1) // batch_size


def batch_bounds(config: Mapping[str, Any], batch_id: int) -> tuple[int, int]:
    """Return zero-based split bounds for a zero-based batch ID."""
    n_batches = configured_batch_count(config)
    if batch_id < 0 or batch_id >= n_batches:
        raise ValueError(f"batch_id must be between 0 and {n_batches - 1}")
    batch_size = int(config["execution"]["batch_size"])
    start = batch_id * batch_size
    return start, min(start + batch_size, configured_split_count(config))


def task_count(config: Mapping[str, Any]) -> int:
    """Return the feature-by-batch array-task count."""
    return len(feature_specs(config)) * configured_batch_count(config)


def task_mapping(config: Mapping[str, Any], task_id: int) -> tuple[str, int]:
    """Return the feature name and zero-based batch for an array task."""
    count = task_count(config)
    if task_id < 0 or task_id >= count:
        raise ValueError(f"task_id must be between 0 and {count - 1}")
    feature_index, batch_id = divmod(task_id, configured_batch_count(config))
    return str(feature_specs(config)[feature_index]["name"]), batch_id


batch_task = task_mapping


def load_common_inputs(
    config: Mapping[str, Any],
    case: str,
    *,
    score_key: str = "final_score",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned IDs, seeds, target masks, model names, and scores."""
    case_spec = named(config, "cases", case)
    with np.load(resolve_path(config["paths"]["split"]), allow_pickle=False) as data:
        prompt_id = data["all_prompt_id"].astype(str)
        seeds = data["seeds"].astype(np.int64)
        target_masks = data[case_spec["target_mask"]].astype(bool)

    with np.load(resolve_path(config["paths"]["scores"]), allow_pickle=False) as data:
        score_prompt_id = data["prompt_id"].astype(str)
        models = data["model"].astype(str)
        scores = data[score_key].astype(np.float64)

    if len(np.unique(models)) != len(models):
        raise ValueError("Duplicate score model names")
    if scores.shape != (len(score_prompt_id), len(models)):
        raise ValueError("Scores do not match prompt and model identifiers")

    if len(np.unique(prompt_id)) != len(prompt_id):
        raise ValueError("Duplicate split prompt_id")
    if len(np.unique(score_prompt_id)) != len(score_prompt_id):
        raise ValueError("Duplicate score prompt_id")
    if set(prompt_id) != set(score_prompt_id):
        raise ValueError("Split and score prompt IDs differ")
    score_row = {value: index for index, value in enumerate(score_prompt_id)}
    aligned_rows = np.asarray([score_row[value] for value in prompt_id])
    if "n_splits" in config.get("execution", {}):
        selected = np.arange(config["execution"]["n_splits"])
        seeds, target_masks = seeds[selected], target_masks[selected]
    return prompt_id, seeds, target_masks, models, scores[aligned_rows]


def load_feature(config: Mapping[str, Any], feature_set: str, prompt_id: np.ndarray) -> np.ndarray:
    """Return one full-population feature matrix selected by YAML."""
    feature_spec = named(config, "feature_sets", feature_set)
    representation = config["representations"][feature_spec["representation"]]
    with np.load(resolve_path(representation["path"]), allow_pickle=False) as data:
        if not np.array_equal(data["prompt_id"].astype(str), prompt_id):
            raise ValueError(f"Prompt order differs for {feature_set}")
        blocks = [
            data[representation["blocks"][block]].astype(np.float64) for block in feature_spec["blocks"]
        ]
    features = np.ascontiguousarray(np.column_stack(blocks))
    expected_shape = (len(prompt_id), int(feature_spec["expected_dimension"]))
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature matrix for {feature_set}: {features.shape}")
    return features


def load_themes(config: Mapping[str, Any], prompt_id: np.ndarray) -> np.ndarray:
    """Return themes aligned to the configured split prompt order."""
    with np.load(resolve_path(config["paths"]["split"]), allow_pickle=False) as data:
        split_prompt_id = data["all_prompt_id"].astype(str)
        themes = data["all_theme"].astype(str)
    if not np.array_equal(split_prompt_id, prompt_id):
        raise ValueError("Theme and score split prompt orders differ")
    return themes


def weighted_model_scores(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted means and complete-case counts for each model."""
    observed = np.isfinite(values)
    numerator = np.nansum(values * weights[:, np.newaxis], axis=0)
    denominator = np.sum(observed * weights[:, np.newaxis], axis=0)
    return numerator / denominator, observed.sum(axis=0).astype(np.int64)


def run_repeated_splits(
    *,
    seeds: np.ndarray,
    target_masks: np.ndarray,
    scores: np.ndarray,
    features: np.ndarray,
    split_indices: Sequence[int],
    fit_method: Callable[[np.ndarray, np.ndarray, np.ndarray, int], SplitWeights],
) -> RepeatedBatch:
    """Fit one method callback across selected splits and score every result."""
    selected = np.asarray(tuple(split_indices), dtype=np.int64)
    raw_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    complete_rows: list[np.ndarray] = []
    diagnostic_rows: dict[str, list[np.ndarray]] = {}
    diagnostic_names: tuple[str, ...] | None = None

    for split_index in selected:
        split_started = time.monotonic()
        target = target_masks[split_index]
        source_indices = np.flatnonzero(~target).astype(np.int64)
        estimate = fit_method(
            features[source_indices],
            features[target],
            source_indices,
            int(seeds[split_index]),
        )
        raw = np.asarray(estimate.raw_weights, dtype=np.float64)
        normalized = np.asarray(estimate.weights, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
            normalized = normalized[np.newaxis, :]
        if raw.shape != normalized.shape or raw.shape[1] != len(source_indices):
            raise ValueError("Method weights do not match the source rows")
        if not np.isfinite(raw).all() or np.any(raw < 0.0):
            raise ValueError("Raw weights must be finite and nonnegative")
        if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
            raise ValueError("Normalized weights must be finite and nonnegative")
        if not np.allclose(normalized.mean(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("Normalized weights must have source mean one")

        method_scores = []
        complete = None
        for method_weights in normalized:
            reweighted, method_complete = weighted_model_scores(scores[source_indices], method_weights)
            method_scores.append(reweighted)
            if complete is None:
                complete = method_complete
            else:
                assert np.array_equal(complete, method_complete)

        names = tuple(estimate.diagnostics)
        if diagnostic_names is None:
            diagnostic_names = names
            diagnostic_rows = {name: [] for name in names}
        elif names != diagnostic_names:
            raise ValueError("Split diagnostics differ across the batch")
        for name, value in estimate.diagnostics.items():
            diagnostic_rows[name].append(np.asarray(value))

        raw_rows.append(raw)
        weight_rows.append(normalized)
        score_rows.append(np.stack(method_scores))
        complete_rows.append(complete)
        timings = _SPLIT_TIMINGS.get()
        if timings is not None:
            elapsed = time.monotonic() - split_started
            timings.append(elapsed)
            print(f"SPLIT {split_index + 1} seconds={elapsed:.3f}", flush=True)

    return RepeatedBatch(
        split_indices=selected,
        raw_weights=np.stack(raw_rows),
        weights=np.stack(weight_rows),
        n_source_complete=np.stack(complete_rows),
        reweighted_scores=np.stack(score_rows),
        diagnostics={name: np.stack(values) for name, values in diagnostic_rows.items()},
    )


def save_npz(path: Path, *, compressed: bool, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically save one compressed or uncompressed NPZ artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    save = np.savez_compressed if compressed else np.savez
    save(temporary, **arrays)
    temporary.replace(path)


def save_batch(path: Path, **arrays: np.ndarray) -> None:
    """Atomically save one compressed batch checkpoint."""
    save_npz(path, compressed=True, arrays=arrays)


def execute_batch(path: Path, build_arrays: Callable[[], Mapping[str, np.ndarray]]) -> Path:
    """Build and save one batch unless the exact checkpoint already exists."""
    if path.exists():
        print(f"EXISTS {path}", flush=True)
        return path
    save_batch(path, **build_arrays())
    print(f"SAVED {path}", flush=True)
    return path


def save_result_pair(
    weight_path: Path,
    score_path: Path,
    *,
    weight_arrays: Mapping[str, np.ndarray],
    score_arrays: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Atomically save separate compressed weights and uncompressed scores."""
    if weight_path.exists() and score_path.exists():
        print(f"EXISTS {weight_path} {score_path}", flush=True)
        return weight_path, score_path
    if weight_path.exists() or score_path.exists():
        raise ValueError(f"Incomplete output pair: {weight_path} {score_path}")
    save_npz(weight_path, compressed=True, arrays=weight_arrays)
    save_npz(score_path, compressed=False, arrays=score_arrays)
    print(f"SAVED {weight_path} {score_path}", flush=True)
    return weight_path, score_path


def load_batches(
    paths: Sequence[Path],
    *,
    constant_fields: Sequence[str],
    concatenated_fields: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load ordered checkpoints, enforcing constant metadata fields."""
    constants: dict[str, np.ndarray] = {}
    parts = {name: [] for name in concatenated_fields}
    expected_fields = set(constant_fields) | set(concatenated_fields)
    for path_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as batch:
            if set(batch.files) != expected_fields:
                raise ValueError(f"Checkpoint schema differs: {path}")
            for name in constant_fields:
                value = batch[name].copy()
                if path_index == 0:
                    constants[name] = value
                elif not np.array_equal(value, constants[name]):
                    raise ValueError(f"Checkpoint field {name!r} differs: {path}")
            for name in concatenated_fields:
                parts[name].append(batch[name].copy())
    return {
        **constants,
        **{name: np.concatenate(values, axis=0) for name, values in parts.items()},
    }


def expected_source_indices(target_masks: np.ndarray, n_splits: int) -> np.ndarray:
    """Return source-row indices for the requested splits."""
    source_counts = np.sum(~target_masks[:n_splits], axis=1)
    if np.unique(source_counts).size != 1:
        raise ValueError("Source sample size must be constant across splits")
    return np.stack([np.flatnonzero(~target).astype(np.int64) for target in target_masks[:n_splits]])


def recompute_scores(
    scores: np.ndarray,
    source_indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scores recomputed from split-aligned one- or multi-method weights."""
    score_rows = []
    complete_rows = []
    for indices, split_weights in zip(source_indices, weights, strict=True):
        method_weights = np.asarray(split_weights)
        if method_weights.ndim == 1:
            method_weights = method_weights[np.newaxis, :]
        method_scores = []
        complete = None
        for values in method_weights:
            reweighted, method_complete = weighted_model_scores(scores[indices], values)
            method_scores.append(reweighted)
            if complete is None:
                complete = method_complete
            else:
                assert np.array_equal(complete, method_complete)
        score_rows.append(np.stack(method_scores))
        complete_rows.append(complete)
    return np.stack(score_rows), np.stack(complete_rows)


def validate_weight_arrays(raw_weights: np.ndarray, weights: np.ndarray) -> None:
    """Validate finite nonnegative raw weights and global mean-one weights."""
    raw = np.asarray(raw_weights, dtype=np.float64)
    normalized = np.asarray(weights, dtype=np.float64)
    if raw.shape != normalized.shape:
        raise ValueError("Raw and normalized weight shapes differ")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("Raw weights must be finite and nonnegative")
    if np.any(raw.mean(axis=-1) <= 0.0):
        raise ValueError("Raw weights must have positive mean")
    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise ValueError("Normalized weights must be finite and nonnegative")
    if not np.allclose(normalized.mean(axis=-1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Normalized weights must have mean one")
