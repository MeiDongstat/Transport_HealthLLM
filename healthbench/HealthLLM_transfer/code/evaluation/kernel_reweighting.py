"""Execute YAML-defined kernel reweighting over repeated splits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = HEALTHLLM_TRANSFER_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from covariateshift.estimators import median_sigma  # noqa: E402
from covariateshift.kernel_based_methods import fit_kernel_weights  # noqa: E402
from evaluation.reweighting_execution import (  # noqa: E402
    SplitWeights,
    batch_bounds,
    load_batches,
    load_common_inputs,
    load_config,
    load_feature,
    named,
    resolve_path,
    run_repeated_splits,
    save_result_pair,
)
from evaluation.reweighting_execution import (  # noqa: E402
    weighted_model_scores as _weighted_model_scores,
)

CONFIG_PATH = HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_kernel_methods.yaml"


def weighted_model_scores(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return model-specific complete-case weighted means."""
    return _weighted_model_scores(values, weights)


def score_batch_path(
    config: Mapping[str, Any], case: str, method: str, feature_set: str, batch_id: int
) -> Path:
    """Return the score-checkpoint path for one batch."""
    start, stop = batch_bounds(config, batch_id)
    return (
        resolve_path(config["paths"]["checkpoint_scores_dir"])
        / case
        / method
        / feature_set
        / f"scores_{start + 1:04d}_{stop:04d}.npz"
    )


def weight_batch_path(
    config: Mapping[str, Any], case: str, method: str, feature_set: str, batch_id: int
) -> Path:
    """Return the compressed weight-checkpoint path for one batch."""
    start, stop = batch_bounds(config, batch_id)
    return (
        resolve_path(config["paths"]["checkpoint_weights_dir"])
        / case
        / method
        / feature_set
        / f"weights_{start + 1:04d}_{stop:04d}.npz"
    )


def final_score_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the case-level reweighted-score artifact path."""
    return (
        resolve_path(config["paths"]["reweighted_scores_dir"])
        / case
        / "kernel_based_methods_reweighted_scores.npz"
    )


def final_weight_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the case-level compressed-weight artifact path."""
    return resolve_path(config["paths"]["weights_dir"]) / case / "kernel_based_methods_weights.npz"


def _optional_float(value: float | None) -> float:
    return np.nan if value is None else float(value)


def build_result_arrays(
    config: Mapping[str, Any],
    *,
    case: str,
    method: str,
    feature_set: str,
    split_indices: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return separate score and weight checkpoint arrays."""
    feature_spec = named(config, "feature_sets", feature_set)
    method_spec = named(config, "methods", method)
    prompt_id, seeds, target_masks, models, scores = load_common_inputs(config, case)
    features = load_feature(config, feature_set, prompt_id)

    def fit_method(
        source_features: np.ndarray,
        target_features: np.ndarray,
        _source_indices: np.ndarray,
        seed: int,
    ) -> SplitWeights:
        sigma = median_sigma(
            np.vstack((source_features, target_features)),
            max_points=int(config["execution"]["median_max_points"]),
            seed=seed,
        )
        estimate = fit_kernel_weights(method_spec, source_features, target_features, sigma=sigma, seed=seed)
        return SplitWeights(
            raw_weights=np.asarray(estimate.raw_weight, dtype=np.float64),
            weights=np.asarray(estimate.normalized_weight, dtype=np.float64),
            diagnostics={
                "sigma": sigma,
                "selected_lambda": _optional_float(estimate.selected_lambda),
                "selected_gamma_multiplier": _optional_float(estimate.selected_gamma_multiplier),
                "selected_gamma": _optional_float(estimate.selected_gamma),
            },
        )

    batch = run_repeated_splits(
        seeds=seeds,
        target_masks=target_masks,
        scores=scores,
        features=features,
        split_indices=split_indices,
        fit_method=fit_method,
    )
    common = {
        "method": np.asarray(method),
        "feature_set": np.asarray(feature_set),
        "expected_dimension": np.asarray(int(feature_spec["expected_dimension"]), dtype=np.int64),
        "split_id": batch.split_indices + 1,
        "seed": seeds[batch.split_indices],
    }
    return (
        {
            **common,
            "model": models,
            "n_source_complete": batch.n_source_complete,
            "reweighted_scores": batch.reweighted_scores[:, 0],
        },
        {
            **common,
            "raw_weights": batch.raw_weights[:, 0],
            "weights": batch.weights[:, 0],
            **batch.diagnostics,
        },
    )


def run_batch(
    config: Mapping[str, Any],
    *,
    case: str,
    method: str,
    feature_set: str,
    batch_id: int,
) -> tuple[Path, Path]:
    """Fit and separately save score and weight checkpoints for one batch."""
    score_path = score_batch_path(config, case, method, feature_set, batch_id)
    weight_path = weight_batch_path(config, case, method, feature_set, batch_id)
    if score_path.exists() and weight_path.exists():
        print(f"EXISTS {score_path} {weight_path}")
        return score_path, weight_path
    if score_path.exists() or weight_path.exists():
        raise ValueError(f"Incomplete output pair for {case}/{method}/{feature_set}/batch {batch_id}")
    start, stop = batch_bounds(config, batch_id)
    score_arrays, weight_arrays = build_result_arrays(
        config,
        case=case,
        method=method,
        feature_set=feature_set,
        split_indices=range(start, stop),
    )
    save_result_pair(
        weight_path,
        score_path,
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )
    return score_path, weight_path


def load_configuration_batches(
    config: Mapping[str, Any], *, case: str, method: str, feature_set: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return one method-feature combination assembled from checkpoints."""
    n_splits = int(config["execution"]["n_splits"])
    batch_size = int(config["execution"]["batch_size"])
    batch_ids = range((n_splits + batch_size - 1) // batch_size)
    constant = ("method", "feature_set", "expected_dimension")
    shared = ("split_id", "seed")
    weights = load_batches(
        [weight_batch_path(config, case, method, feature_set, batch) for batch in batch_ids],
        constant_fields=constant,
        concatenated_fields=shared
        + (
            "raw_weights",
            "weights",
            "sigma",
            "selected_lambda",
            "selected_gamma_multiplier",
            "selected_gamma",
        ),
    )
    scores = load_batches(
        [score_batch_path(config, case, method, feature_set, batch) for batch in batch_ids],
        constant_fields=constant + ("model",),
        concatenated_fields=shared + ("n_source_complete", "reweighted_scores"),
    )
    return scores, weights


def finalize_case(
    config: Mapping[str, Any],
    *,
    case: str,
    methods: Sequence[str] | None = None,
    feature_sets: Sequence[str] | None = None,
    n_splits: int | None = None,
) -> tuple[Path, Path]:
    """Merge checkpoints into the two case-level output artifacts."""
    named(config, "cases", case)
    selected_methods = (
        list(methods) if methods is not None else [str(item["name"]) for item in config["methods"]]
    )
    selected_features = (
        list(feature_sets)
        if feature_sets is not None
        else [str(item["name"]) for item in config["feature_sets"]]
    )
    selected_n_splits = int(n_splits) if n_splits is not None else int(config["execution"]["n_splits"])
    results = [
        [
            load_configuration_batches(config, case=case, method=method, feature_set=feature)
            for method in selected_methods
        ]
        for feature in selected_features
    ]
    score_results = [[result[0] for result in row] for row in results]
    weight_results = [[result[1] for result in row] for row in results]
    _, seeds, _, models, _ = load_common_inputs(config, case)
    common = {
        "method": np.asarray(selected_methods),
        "feature_set": np.asarray(selected_features),
        "expected_dimension": np.asarray(
            [
                int(named(config, "feature_sets", feature)["expected_dimension"])
                for feature in selected_features
            ],
            dtype=np.int64,
        ),
        "split_id": np.arange(1, selected_n_splits + 1, dtype=np.int64),
        "seed": seeds[:selected_n_splits],
    }
    weight_arrays = {
        **common,
        **{
            name: np.stack(
                [
                    np.stack([result[name][:selected_n_splits] for result in feature_row])
                    for feature_row in weight_results
                ]
            )
            for name in (
                "raw_weights",
                "weights",
                "sigma",
                "selected_lambda",
                "selected_gamma_multiplier",
                "selected_gamma",
            )
        },
    }
    score_arrays = {
        **common,
        "model": models,
        "n_source_complete": score_results[0][0]["n_source_complete"][:selected_n_splits],
        "reweighted_scores": np.stack(
            [
                np.stack([result["reweighted_scores"][:selected_n_splits] for result in feature_row])
                for feature_row in score_results
            ]
        ),
    }
    weight_path, score_path = save_result_pair(
        final_weight_path(config, case),
        final_score_path(config, case),
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )
    return score_path, weight_path


def build_parser() -> argparse.ArgumentParser:
    """Return the batch-runner command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method")
    parser.add_argument("--feature-set")
    parser.add_argument("--batch-id", type=int)
    parser.add_argument("--finalize-case", action="store_true")
    return parser


def main() -> None:
    """Run one kernel batch or finalize one case."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.finalize_case:
        finalize_case(config, case=args.case)
        return
    if args.method is None or args.feature_set is None or args.batch_id is None:
        raise ValueError("--method, --feature-set, and --batch-id are required")
    run_batch(
        config,
        case=args.case,
        method=args.method,
        feature_set=args.feature_set,
        batch_id=args.batch_id,
    )


if __name__ == "__main__":
    main()
