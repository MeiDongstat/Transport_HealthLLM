"""Validate finalized HealthLLM_transfer reweighting artifacts."""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = HEALTHLLM_TRANSFER_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.reweighting_execution import (  # noqa: E402
    configured_split_count,
    expected_source_indices,
    feature_specs,
    load_common_inputs,
    load_config,
    recompute_scores,
    validate_weight_arrays,
)


def verify_npz_compression(path: Path, *, compressed: bool) -> None:
    """Verify every NPZ member uses the requested ZIP storage mode."""
    expected = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or any(member.compress_type != expected for member in members):
            raise ValueError(f"Unexpected NPZ compression mode: {path}")


def _read_npz(path: Path, fields: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as result:
        if set(result.files) != fields:
            raise ValueError(f"Final artifact schema differs: {path}")
        return {name: result[name].copy() for name in result.files}


def _verify_common_metadata(
    arrays: Mapping[str, np.ndarray],
    *,
    methods: np.ndarray,
    features: np.ndarray,
    dimensions: np.ndarray,
    seeds: np.ndarray,
) -> None:
    if not np.array_equal(arrays["method"].astype(str), methods.astype(str)):
        raise ValueError("Final method order differs")
    if not np.array_equal(arrays["feature_set"].astype(str), features.astype(str)):
        raise ValueError("Final feature order differs")
    if not np.array_equal(arrays["expected_dimension"], dimensions):
        raise ValueError("Final feature dimensions differ")
    if not np.array_equal(arrays["split_id"], np.arange(1, len(seeds) + 1, dtype=np.int64)):
        raise ValueError("Final split IDs differ")
    if not np.array_equal(arrays["seed"], seeds):
        raise ValueError("Final split seeds differ")


def verify_kernel_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Verify finalized kernel-method weights and scores."""
    from evaluation import kernel_reweighting as kernel

    weight_path = kernel.final_weight_path(config, case)
    score_path = kernel.final_score_path(config, case)
    weight_fields = {
        "method",
        "feature_set",
        "expected_dimension",
        "split_id",
        "seed",
        "raw_weights",
        "weights",
        "sigma",
        "selected_lambda",
        "selected_gamma_multiplier",
        "selected_gamma",
    }
    score_fields = {
        "method",
        "feature_set",
        "expected_dimension",
        "split_id",
        "seed",
        "model",
        "n_source_complete",
        "reweighted_scores",
    }
    weight_arrays = _read_npz(weight_path, weight_fields)
    score_arrays = _read_npz(score_path, score_fields)
    _, seeds, target_masks, models, scores = load_common_inputs(config, case)
    n_splits = configured_split_count(config)
    seeds = seeds[:n_splits]
    source_indices = expected_source_indices(target_masks, n_splits)
    methods = np.asarray([str(method["name"]) for method in config["methods"]])
    features = np.asarray([str(feature["name"]) for feature in config["feature_sets"]])
    dimensions = np.asarray(
        [int(feature["expected_dimension"]) for feature in config["feature_sets"]],
        dtype=np.int64,
    )
    for arrays in (weight_arrays, score_arrays):
        _verify_common_metadata(
            arrays,
            methods=methods,
            features=features,
            dimensions=dimensions,
            seeds=seeds,
        )
    if not np.array_equal(score_arrays["model"].astype(str), models):
        raise ValueError("Final model order differs")
    validate_weight_arrays(weight_arrays["raw_weights"], weight_arrays["weights"])
    for feature_index, feature_weights in enumerate(weight_arrays["weights"]):
        for method_index, method_weights in enumerate(feature_weights):
            expected_scores, expected_complete = recompute_scores(scores, source_indices, method_weights)
            if not np.array_equal(score_arrays["n_source_complete"], expected_complete):
                raise ValueError("Final complete-case counts differ")
            if not np.allclose(
                score_arrays["reweighted_scores"][feature_index, method_index],
                expected_scores[:, 0],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError("Final reweighted scores differ")
    verify_npz_compression(weight_path, compressed=True)
    verify_npz_compression(score_path, compressed=False)
    print(f"VERIFIED {weight_path} {score_path}", flush=True)
    return weight_path, score_path


def verify_predicted_theme_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Verify finalized predicted-theme weights and scores."""
    from evaluation import predicted_theme_reweighting as method

    score_path, weight_path = method.final_paths(config, case)
    weight_arrays = _read_npz(weight_path, set(method.FINAL_WEIGHT_FIELDS))
    score_arrays = _read_npz(score_path, set(method.FINAL_SCORE_FIELDS))
    _, seeds, target_masks, models, scores = load_common_inputs(config, case)
    n_splits = configured_split_count(config)
    seeds = seeds[:n_splits]
    source_indices = expected_source_indices(target_masks, n_splits)
    specifications = feature_specs(config)
    feature_names = np.asarray([str(item["name"]) for item in specifications])
    dimensions = np.asarray([int(item["expected_dimension"]) for item in specifications], dtype=np.int64)
    for arrays in (weight_arrays, score_arrays):
        _verify_common_metadata(
            arrays,
            methods=method.output_methods(config),
            features=feature_names,
            dimensions=dimensions,
            seeds=seeds,
        )
    if not np.array_equal(weight_arrays["theme"].astype(str), method.theme_labels(config)):
        raise ValueError("Final theme labels differ")
    if not np.array_equal(score_arrays["model"].astype(str), models):
        raise ValueError("Final model order differs")
    validate_weight_arrays(weight_arrays["raw_weights"], weight_arrays["weights"])
    candidates = np.asarray(
        method.named(config, "methods", "predicted_probability_kmm")["gamma_multipliers"],
        dtype=np.float64,
    )
    multipliers = weight_arrays["selected_gamma_multiplier"]
    if not np.isclose(multipliers[..., np.newaxis], candidates).any(axis=-1).all():
        raise ValueError("Final KMM gamma multiplier differs from YAML")
    sigma = weight_arrays["kmm_sigma"]
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        raise ValueError("Final KMM sigma is invalid")
    selected_gamma = weight_arrays["selected_gamma"]
    expected_gamma = multipliers / (2.0 * sigma**2)
    if not np.allclose(selected_gamma, expected_gamma, rtol=1e-12, atol=1e-12):
        raise ValueError("Final selected KMM gamma differs from sigma and multiplier")
    for feature_index, feature_weights in enumerate(weight_arrays["weights"]):
        expected_scores, expected_complete = recompute_scores(
            scores, source_indices, feature_weights.transpose(1, 0, 2)
        )
        if not np.array_equal(score_arrays["n_source_complete"], expected_complete):
            raise ValueError("Final complete-case counts differ")
        if not np.allclose(
            score_arrays["reweighted_scores"][feature_index],
            expected_scores.transpose(1, 0, 2),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("Final reweighted scores differ")
    verify_npz_compression(weight_path, compressed=True)
    verify_npz_compression(score_path, compressed=False)
    print(f"VERIFIED {score_path} {weight_path}", flush=True)
    return score_path, weight_path


def verify_plus_embedding_case(
    config: Mapping[str, Any],
    *,
    case: str,
    n_folds: int | None = None,
) -> tuple[Path, Path]:
    """Verify finalized theme-probability plus embedding outputs."""
    from evaluation import predicted_theme_plus_embedding_reweighting as method

    weight_path = method.final_weights_path(config, case)
    score_path = method.final_scores_path(config, case)
    shared = {
        "method",
        "feature_set",
        "expected_dimension",
        "hybrid_dimension",
        "n_folds",
        "split_id",
        "seed",
    }
    weight_arrays = _read_npz(
        weight_path,
        shared | {"raw_weights", "weights"},
    )
    score_arrays = _read_npz(score_path, shared | {"model", "n_source_complete", "reweighted_scores"})
    _, seeds, target_masks, models, scores = load_common_inputs(config, case)
    labels = np.asarray(config["theme_labels"]).astype(str)
    n_splits = configured_split_count(config)
    seeds = seeds[:n_splits]
    fold_count = int(n_folds) if n_folds is not None else int(config["cross_fitting"]["n_folds"])
    source_indices = expected_source_indices(target_masks, n_splits)
    specifications = feature_specs(config)
    feature_names = np.asarray([str(item["name"]) for item in specifications])
    dimensions = np.asarray([int(item["expected_dimension"]) for item in specifications], dtype=np.int64)
    for arrays in (weight_arrays, score_arrays):
        _verify_common_metadata(
            arrays,
            methods=np.asarray(method.METHOD_NAMES),
            features=feature_names,
            dimensions=dimensions,
            seeds=seeds,
        )
        if not np.array_equal(arrays["hybrid_dimension"], dimensions + len(labels)):
            raise ValueError("Final hybrid dimensions differ")
        if int(arrays["n_folds"]) != fold_count:
            raise ValueError("Final fold count differs")
    if not np.array_equal(score_arrays["model"].astype(str), models):
        raise ValueError("Final model order differs")
    validate_weight_arrays(weight_arrays["raw_weights"], weight_arrays["weights"])
    for feature_index, feature_weights in enumerate(weight_arrays["weights"]):
        expected_scores, expected_complete = recompute_scores(
            scores, source_indices, feature_weights.transpose(1, 0, 2)
        )
        if not np.array_equal(score_arrays["n_source_complete"], expected_complete):
            raise ValueError("Final complete-case counts differ")
        if not np.allclose(
            score_arrays["reweighted_scores"][feature_index],
            expected_scores.transpose(1, 0, 2),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("Final reweighted scores differ")
    verify_npz_compression(weight_path, compressed=True)
    verify_npz_compression(score_path, compressed=False)
    print(f"VERIFIED {weight_path} {score_path}", flush=True)
    return weight_path, score_path


def verify_domain_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Verify finalized domain-classifier weights and scores."""
    from evaluation import domain_classifier_reweighting as method

    weight_path = method.final_weights_path(config, case)
    score_path = method.final_scores_path(config, case)
    shared = {
        "method",
        "feature_set",
        "expected_dimension",
        "n_folds",
        "split_id",
        "seed",
    }
    weight_arrays = _read_npz(
        weight_path,
        shared
        | {
            "raw_weights",
            "weights",
            "source_probability",
            "target_probability",
        },
    )
    score_arrays = _read_npz(score_path, shared | {"model", "n_source_complete", "reweighted_scores"})
    _, seeds, target_masks, models, scores = load_common_inputs(config, case)
    n_splits = configured_split_count(config)
    seeds = seeds[:n_splits]
    n_folds = int(config["cross_fitting"]["n_folds"])
    source_indices = expected_source_indices(target_masks, n_splits)
    specifications = feature_specs(config)
    feature_names = np.asarray([str(item["name"]) for item in specifications])
    dimensions = np.asarray([int(item["expected_dimension"]) for item in specifications], dtype=np.int64)
    for arrays in (weight_arrays, score_arrays):
        _verify_common_metadata(
            arrays,
            methods=method.OUTPUT_METHODS,
            features=feature_names,
            dimensions=dimensions,
            seeds=seeds,
        )
        if int(arrays["n_folds"]) != n_folds:
            raise ValueError("Final fold count differs")
    if not np.array_equal(score_arrays["model"].astype(str), models):
        raise ValueError("Final model order differs")
    source_probability = weight_arrays["source_probability"]
    target_probability = weight_arrays["target_probability"]
    if source_probability.shape != weight_arrays["weights"].shape:
        raise ValueError("Final source domain probability shape differs")
    expected_target_shape = (
        len(specifications),
        len(method.OUTPUT_METHODS),
        n_splits,
        int(target_masks[0].sum()),
    )
    if target_probability.shape != expected_target_shape:
        raise ValueError("Final target domain probability shape differs")
    for name, probabilities in (
        ("source", source_probability),
        ("target", target_probability),
    ):
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"Final {name} domain probabilities are invalid")
    validate_weight_arrays(weight_arrays["raw_weights"], weight_arrays["weights"])
    for feature_index, feature_weights in enumerate(weight_arrays["weights"]):
        for method_index, method_weights in enumerate(feature_weights):
            expected_scores, expected_complete = recompute_scores(scores, source_indices, method_weights)
            if not np.array_equal(score_arrays["n_source_complete"], expected_complete):
                raise ValueError("Final complete-case counts differ")
            if not np.allclose(
                score_arrays["reweighted_scores"][feature_index, method_index],
                expected_scores[:, 0],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError("Final reweighted scores differ")
    verify_npz_compression(weight_path, compressed=True)
    verify_npz_compression(score_path, compressed=False)
    print(f"VERIFIED {weight_path} {score_path}", flush=True)
    return weight_path, score_path


WORKFLOWS = {
    "kernel": (
        HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_kernel_methods.yaml",
        verify_kernel_case,
    ),
    "predicted-theme": (
        HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_predicted_theme_reweighting.yaml",
        verify_predicted_theme_case,
    ),
    "predicted-theme-plus-embedding": (
        HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_predicted_theme_plus_embedding_reweighting.yaml",
        verify_plus_embedding_case,
    ),
    "domain-classifier": (
        HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_domain_classifier.yaml",
        verify_domain_case,
    ),
}


def main() -> None:
    """Validate one finalized workflow case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, choices=tuple(WORKFLOWS))
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    default_config, verify = WORKFLOWS[args.workflow]
    config = load_config(args.config if args.config is not None else default_config)
    verify(config, case=args.case)


if __name__ == "__main__":
    main()
