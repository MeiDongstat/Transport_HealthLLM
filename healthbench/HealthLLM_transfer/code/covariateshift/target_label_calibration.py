"""PPI++ and AIPW target-mean estimation with labelled target outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from covariateshift.common_mean_regression import RegressorFactory, predict_mean


def half_up_count(n_rows: int, fraction: float) -> int:
    """Return a half-up rounded sample count."""

    return int((Decimal(n_rows) * Decimal(str(fraction))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def make_nested_label_splits(
    n_target: int,
    fractions: Sequence[float],
    *,
    seed: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Create nested labeled target samples from one random permutation."""

    permutation = np.random.default_rng(seed).permutation(n_target)
    all_indices = np.arange(n_target)
    splits = {}
    for fraction in sorted(float(value) for value in fractions):
        labeled = permutation[: half_up_count(n_target, fraction)].copy()
        unlabeled = np.setdiff1d(all_indices, labeled, assume_unique=True)
        splits[fraction] = {
            "labeled_indices": labeled,
            "unlabeled_indices": unlabeled,
        }
    return splits


def estimate_ppi(
    target_source_prediction: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
) -> dict[str, float]:
    """Return PPI++ components, the power coefficient, and its two-sample standard error."""

    prediction = np.asarray(target_source_prediction, dtype=float)
    labeled_outcome = np.asarray(labeled_outcome, dtype=float)
    n_labeled = len(labeled_indices)
    if not 2 <= n_labeled < len(prediction):
        raise ValueError("PPI++ requires at least two labeled and one unlabeled target observations.")

    unlabeled_mask = np.ones(len(prediction), dtype=bool)
    unlabeled_mask[labeled_indices] = False
    labeled_prediction = prediction[labeled_indices]
    unlabeled_prediction = prediction[unlabeled_mask]
    if np.ptp(prediction) == 0:
        # Constant predictions have zero contrast for every coefficient.
        lambda_hat = 0.0
    else:
        # Angelopoulos et al. (2023), PPI++, Example 6.1: L covariance, pooled T variance.
        outcome_prediction_covariance = float(np.cov(labeled_outcome, labeled_prediction, ddof=1)[0, 1])
        prediction_variance = float(np.var(prediction, ddof=1))
        sample_size_ratio = n_labeled / len(unlabeled_prediction)
        lambda_hat = outcome_prediction_covariance / ((1 + sample_size_ratio) * prediction_variance)

    plugin = float(lambda_hat * unlabeled_prediction.mean())
    labeled_residual = labeled_outcome - lambda_hat * labeled_prediction
    correction = float(np.mean(labeled_residual))
    variance = np.var(labeled_residual, ddof=1) / n_labeled
    if lambda_hat != 0:
        variance += (
            lambda_hat**2 * np.var(unlabeled_prediction, ddof=1) / len(unlabeled_prediction)
            if len(unlabeled_prediction) >= 2
            else np.nan
        )
    return {
        "estimate": plugin + correction,
        "plugin_mean": plugin,
        "residual_correction": correction,
        "lambda_hat": lambda_hat,
        "standard_error": float(np.sqrt(variance)),
    }


def estimate_aipw(
    target_source_prediction: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
    *,
    label_probability: float | None = None,
) -> dict[str, float]:
    """Return AIPW with a supplied or empirical label probability and fixed predictions."""

    if len(labeled_indices) == 0:
        raise ValueError("AIPW requires a nonempty labeled target sample.")

    prediction = np.asarray(target_source_prediction, dtype=float)
    residual = np.asarray(labeled_outcome, dtype=float) - prediction[labeled_indices]
    plugin = float(prediction.mean())
    if label_probability is None:
        correction = float(residual.mean())
    else:
        if not 0 < label_probability <= 1:
            raise ValueError("label_probability must lie in (0, 1]")
        correction = float(residual.sum() / (label_probability * len(prediction)))
    return {
        "estimate": plugin + correction,
        "plugin_mean": plugin,
        "residual_correction": correction,
    }


def estimate_crossfit_aipw(
    target_prediction: np.ndarray,
    target_labeled: np.ndarray,
    target_outcome: np.ndarray,
    *,
    target_fold_ids: np.ndarray,
    label_probability: float,
    empirical_label_probability: bool = False,
) -> dict[str, float]:
    """Return equally averaged fold AIPW components and their linearized standard error."""
    prediction = np.asarray(target_prediction, dtype=float)
    labeled = np.asarray(target_labeled, dtype=bool)
    outcome = np.asarray(target_outcome, dtype=float)
    folds = np.asarray(target_fold_ids)
    if not 0 < label_probability <= 1:
        raise ValueError("label_probability must lie in (0, 1]")
    correction = np.zeros(len(prediction))
    correction[labeled] = (outcome[labeled] - prediction[labeled]) / label_probability
    fold_values = np.unique(folds)
    components = np.array(
        [(prediction[folds == fold].mean(), correction[folds == fold].mean()) for fold in fold_values]
    )
    plugin, residual = components.mean(axis=0)
    fold_variances = np.empty(len(fold_values))
    for index, fold in enumerate(fold_values):
        heldout = folds == fold
        fold_size = heldout.sum()
        score = prediction[heldout] + correction[heldout]
        if empirical_label_probability:
            # Global alpha estimation couples the folds through their sample-size proportions.
            score -= (
                len(fold_values)
                * fold_size
                / len(prediction)
                * residual
                * labeled[heldout]
                / label_probability
            )
        fold_variances[index] = np.var(score, ddof=1) / fold_size if fold_size >= 2 else np.nan
    return {
        "estimate": float(plugin + residual),
        "plugin_mean": float(plugin),
        "residual_correction": float(residual),
        "standard_error": float(np.sqrt(fold_variances.sum()) / len(fold_values)),
    }


def fit_target_aipw(
    target_features: np.ndarray,
    target_labeled: np.ndarray,
    target_outcome: np.ndarray,
    *,
    target_fold_ids: np.ndarray,
    label_probability: float,
    seed: int,
    regressor_factory: RegressorFactory,
    empirical_label_probability: bool = False,
    model_seeds: Sequence[int] | None = None,
) -> dict[str, float]:
    """Return target-only cross-fitted AIPW and its equally weighted fold standard error."""
    features = np.asarray(target_features, dtype=float)
    labeled = np.asarray(target_labeled, dtype=bool)
    outcome = np.asarray(target_outcome, dtype=float)
    folds = np.asarray(target_fold_ids)
    if labeled.shape != (len(features),) or outcome.shape != labeled.shape or folds.shape != labeled.shape:
        raise ValueError("Target labels, outcomes, and fold IDs must align with target features")
    if not np.isfinite(outcome[labeled]).all():
        raise ValueError("Labeled target outcomes must be finite")
    n_folds = len(np.unique(folds))
    if (
        n_folds < 2
        or not np.issubdtype(folds.dtype, np.integer)
        or not np.array_equal(
            np.unique(folds),
            np.arange(n_folds),
        )
    ):
        raise ValueError("Target fold IDs must span at least two contiguous folds starting at zero")

    prediction = np.empty(len(features))
    if model_seeds is None:
        model_seeds = [
            int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(n_folds)
        ]
    elif len(model_seeds) != n_folds:
        raise ValueError("AIPW requires one model seed per target fold")
    for fold in range(n_folds):
        heldout = folds == fold
        training = labeled & ~heldout
        if not training.any():
            raise ValueError(f"AIPW training fold {fold} has no labeled target observations")
        model = regressor_factory(model_seeds[fold])
        model.fit(features[training], outcome[training])
        prediction[heldout] = predict_mean(model, features[heldout])
        del model
    return estimate_crossfit_aipw(
        prediction,
        labeled,
        outcome,
        target_fold_ids=folds,
        label_probability=label_probability,
        empirical_label_probability=empirical_label_probability,
    )
