"""Pooled-label DR with direct pooled KMM weights and joint outcome cross-fitting.

Density ratios estimate target density divided by pooled source-plus-labelled-
target density, ordered as source rows followed by labelled target rows.
Source and target predictions follow the supplied source and target folds.
KMM fits these ratios directly on pooled labelled rows against all target
covariates. Its mean-one sample weights enter the weighted residual correction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import common_mean_regression as common
from . import kernel_based_methods as kernel
from .estimators import median_sigma


def fit_pooled_label_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    labeled_indices: np.ndarray,
    *,
    density_config: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Return direct KMM weights for source rows followed by labelled target rows."""
    method = density_config["method"]
    if method["name"] != "kmm":
        raise ValueError("Pooled-label DR requires direct KMM weights")
    pooled_features = np.vstack((source_features, target_features[labeled_indices]))
    sigma = median_sigma(
        np.vstack((pooled_features, target_features)),
        max_points=density_config["median_max_points"],
        seed=seed,
    )
    estimate = kernel.fit_kernel_weights(
        method,
        pooled_features,
        target_features,
        sigma=sigma,
        seed=seed,
    )
    return estimate.normalized_weight


def fit_pooled_label_outcome_crossfit(
    source_features: np.ndarray,
    source_outcome: np.ndarray,
    target_features: np.ndarray,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
    *,
    source_fold_ids: np.ndarray,
    target_fold_ids: np.ndarray,
    model_factory: Callable[[int], Any],
    model_seed: int,
) -> dict[str, np.ndarray]:
    """Return pooled-regression predictions on supplied source and target folds."""

    assert np.array_equal(
        np.sort(np.r_[labeled_indices, unlabeled_indices]),
        np.arange(len(target_features)),
    )
    fitted = common.fit_common_mean_crossfit(
        source_features,
        source_outcome,
        target_features,
        labeled_indices=labeled_indices,
        labeled_outcome=labeled_outcome,
        source_fold_ids=source_fold_ids,
        target_fold_ids=target_fold_ids,
        regressor_factory=model_factory,
        seed=model_seed,
    )
    return {
        "source_oof_prediction": fitted.source_prediction,
        "target_crossfit_prediction": fitted.target_prediction,
    }


def estimate_pooled_label_dr(
    source_outcome: np.ndarray,
    source_oof_prediction: np.ndarray,
    labeled_outcome: np.ndarray,
    labeled_indices: np.ndarray,
    target_crossfit_prediction: np.ndarray,
    weights_source_labeled_to_target: np.ndarray,
) -> dict[str, float]:
    """Return the pooled DR estimate and its two-sample normalized-ratio standard error."""

    target_prediction = np.asarray(target_crossfit_prediction, dtype=float)
    source_residual = np.asarray(source_outcome) - np.asarray(source_oof_prediction)
    labeled_residual = np.asarray(labeled_outcome) - target_prediction[labeled_indices]
    plugin = float(target_prediction.mean())
    correction = float(
        np.average(
            np.concatenate((source_residual, labeled_residual)),
            weights=weights_source_labeled_to_target,
        )
    )
    weights = np.asarray(weights_source_labeled_to_target, dtype=float)
    weights = weights / weights.sum()
    n_source, n_target = len(source_residual), len(target_prediction)
    # Center residuals by the ratio estimate to include its random denominator.
    source_score = n_source * weights[:n_source] * (source_residual - correction)
    target_score = target_prediction.copy()
    target_score[labeled_indices] += n_target * weights[n_source:] * (labeled_residual - correction)
    variance = (
        np.var(source_score, ddof=1) / n_source + np.var(target_score, ddof=1) / n_target
        if min(n_source, n_target) >= 2
        else np.nan
    )
    return {
        "estimate": plugin + correction,
        "plugin_mean": plugin,
        "residual_correction": correction,
        "standard_error": float(np.sqrt(variance)),
    }


def fit_pooled_label_dr(
    source_features: np.ndarray,
    source_outcome: np.ndarray,
    target_features: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
    *,
    density_config: Mapping[str, Any],
    density_seed: int,
    source_fold_ids: np.ndarray,
    target_fold_ids: np.ndarray,
    model_factory: Callable[[int], Any],
    model_seed: int,
) -> dict[str, float | np.ndarray]:
    """Return pooled-label DR components, predictions, and direct pooled KMM weights."""

    weights = fit_pooled_label_weights(
        source_features,
        target_features,
        labeled_indices,
        density_config=density_config,
        seed=density_seed,
    )
    unlabeled_indices = np.setdiff1d(
        np.arange(len(target_features)),
        labeled_indices,
    )
    predictions = fit_pooled_label_outcome_crossfit(
        source_features,
        source_outcome,
        target_features,
        labeled_indices,
        unlabeled_indices,
        labeled_outcome,
        source_fold_ids=source_fold_ids,
        target_fold_ids=target_fold_ids,
        model_factory=model_factory,
        model_seed=model_seed,
    )
    estimate = estimate_pooled_label_dr(
        source_outcome,
        predictions["source_oof_prediction"],
        labeled_outcome,
        labeled_indices,
        predictions["target_crossfit_prediction"],
        weights,
    )
    return {
        **estimate,
        **predictions,
        "weights_source_labeled_to_target": weights,
    }


def main() -> None:
    """Fit pooled-label DR from an NPZ sample and print its estimate components."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Input NPZ keys: source_features, source_outcome, target_features, "
            "labeled_indices, labeled_outcome, "
            "source_fold_ids, target_fold_ids. Outcomes and indices are vectors; "
            "feature arrays are matrices. KMM estimates target density divided "
            "by pooled labelled density directly from covariates. Labelled outcomes follow "
            "labeled_indices order. Fold IDs follow the complete source and target "
            "row order and are contiguous and zero-based."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Single-sample NPZ.")
    parser.add_argument(
        "--kernel-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/healthbench_kernel_methods.yaml",
        help="Kernel configuration containing the KMM method and median_max_points.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local TabPFN regressor checkpoint.")
    parser.add_argument("--device", default="cuda", help="TabPFN device (default: cuda).")
    parser.add_argument("--model-version", default="V3", help="TabPFN model version (default: V3).")
    parser.add_argument("--n-estimators", type=int, default=4, help="TabPFN ensemble size (default: 4).")
    parser.add_argument("--seed", type=int, default=123, help="Outcome-model seed (default: 123).")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as data:
        inputs = {
            name: data[name]
            for name in (
                "source_features",
                "source_outcome",
                "target_features",
                "labeled_indices",
                "labeled_outcome",
                "source_fold_ids",
                "target_fold_ids",
            )
        }

    kernel_config = yaml.safe_load(args.kernel_config.read_text(encoding="utf-8"))
    density_config = {
        "method": next(method for method in kernel_config["methods"] if method["name"] == "kmm"),
        "median_max_points": kernel_config["execution"]["median_max_points"],
    }
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_version = getattr(ModelVersion, args.model_version.upper())

    def model_factory(seed: int) -> Any:
        return TabPFNRegressor.create_default_for_version(
            model_version,
            model_path=str(args.checkpoint),
            device=args.device,
            n_estimators=args.n_estimators,
            ignore_pretraining_limits=True,
            random_state=seed,
        )

    result = fit_pooled_label_dr(
        **inputs,
        density_config=density_config,
        density_seed=args.seed,
        model_factory=model_factory,
        model_seed=args.seed,
    )
    print(json.dumps({name: result[name] for name in ("estimate", "plugin_mean", "residual_correction")}))


if __name__ == "__main__":
    main()
