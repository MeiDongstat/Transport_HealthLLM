"""Three-fold recalibrated prediction-powered estimation of a target mean.

Target labels are sampled randomly from the target population. Fixed source predictions must be learned independently of these labels. Recalibration uses target covariates together with the fixed prediction; each supplied regressor fits its preprocessing on its training fold only.

The scalar mean specialization follows Ji, Lei and Zrnic (2025), RePPI:
https://github.com/Wenlong2000/RePPI/blob/main/reppi.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Regressor(Protocol):
    """Describe the recalibration learner's fit and predict interface."""

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> Any:
        """Return the regressor fitted to the supplied labeled observations."""
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return one conditional-mean prediction per supplied row."""
        ...


RegressorFactory = Callable[[int], Regressor]


@dataclass(frozen=True)
class RePPIEstimate:
    """Store the target mean, its two components, and three ordered fold results."""

    estimate: float
    plugin_mean: float
    residual_correction: float
    fold_estimate: np.ndarray
    fold_coefficient: np.ndarray
    standard_error: float


@dataclass(frozen=True)
class RePPIResult(RePPIEstimate):
    """Store labeled-order predictions and fold-by-unlabeled-row predictions."""

    labeled_prediction: np.ndarray
    unlabeled_prediction: np.ndarray
    labeled_fold_id: np.ndarray
    unlabeled_indices: np.ndarray


def _vector(values: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if size is not None and len(values) != size:
        raise ValueError(f"{name} must have length {size}")
    return values


def _estimate(
    labeled_outcome: np.ndarray,
    labeled_prediction: np.ndarray,
    unlabeled_prediction: np.ndarray,
    labeled_fold_id: np.ndarray,
) -> RePPIEstimate:
    n_labeled = len(labeled_outcome)
    n_unlabeled = unlabeled_prediction.shape[1]
    fold_coefficient = np.empty(3, dtype=np.float64)
    fold_estimate = np.empty(3, dtype=np.float64)
    plugin_mean = 0.0
    residual_correction = 0.0
    labeled_variance = 0.0
    unlabeled_contributions = np.zeros(n_unlabeled)
    for fold_id in range(3):
        heldout = labeled_fold_id == fold_id
        outcome = labeled_outcome[heldout]
        prediction = labeled_prediction[heldout]
        covariance = np.cov(outcome, prediction, ddof=1)
        prediction_variance = float(covariance[1, 1])
        constant_prediction = np.ptp(prediction) == 0
        if constant_prediction or prediction_variance == 0:
            # Undefined recalibration leaves this fold's labeled mean and sampling variance.
            coefficient = 0.0
        else:
            # The initial mean and gradient scaling cancel; the ratio uses this estimation fold.
            coefficient = float(covariance[0, 1] / ((1 + len(outcome) / n_unlabeled) * prediction_variance))
        plugin = float(coefficient * unlabeled_prediction[fold_id].mean())
        residual = outcome - coefficient * prediction
        correction = float(residual.mean())
        fold_coefficient[fold_id] = coefficient
        fold_estimate[fold_id] = plugin + correction
        fold_fraction = len(outcome) / n_labeled
        plugin_mean += fold_fraction * plugin
        residual_correction += fold_fraction * correction
        labeled_variance += fold_fraction**2 * np.var(residual, ddof=1) / len(outcome)
        unlabeled_contributions += fold_fraction * coefficient * unlabeled_prediction[fold_id]
    # All rotations reuse U; their prediction covariance belongs in its variance.
    unlabeled_variance = 0.0
    if np.any(fold_coefficient != 0):
        unlabeled_variance = (
            np.var(unlabeled_contributions, ddof=1) / n_unlabeled if n_unlabeled >= 2 else np.nan
        )
    return RePPIEstimate(
        estimate=plugin_mean + residual_correction,
        plugin_mean=plugin_mean,
        residual_correction=residual_correction,
        fold_estimate=fold_estimate,
        fold_coefficient=fold_coefficient,
        standard_error=float(np.sqrt(labeled_variance + unlabeled_variance)),
    )


def estimate_reppi(
    labeled_outcome: np.ndarray,
    *,
    labeled_prediction: np.ndarray,
    unlabeled_prediction: np.ndarray,
    labeled_fold_id: np.ndarray,
) -> RePPIEstimate:
    """Return a RePPI mean from aligned labels and three-fold predictions.

    Labeled arrays share the input label order; fold IDs are zero-based.
    unlabeled_prediction has shape (3, n_unlabeled), with row k produced by
    the same model as labeled_prediction on estimation fold k.
    The plug-in SE treats fitted regressors and recalibration coefficients as fixed.
    A fold with zero labeled prediction variance contributes its labeled mean
    and sampling variance, with recalibration coefficient zero.
    """
    labeled_outcome = _vector(labeled_outcome, "labeled_outcome")
    labeled_prediction = _vector(labeled_prediction, "labeled_prediction", len(labeled_outcome))
    labeled_fold_id = np.asarray(labeled_fold_id)
    if (
        labeled_fold_id.shape != labeled_outcome.shape
        or not np.issubdtype(labeled_fold_id.dtype, np.integer)
        or not np.isin(labeled_fold_id, [0, 1, 2]).all()
    ):
        raise ValueError("labeled_fold_id must contain one integer fold ID in {0, 1, 2} per label")
    if any(np.count_nonzero(labeled_fold_id == fold_id) < 2 for fold_id in range(3)):
        raise ValueError("RePPI requires at least two labeled observations in each of three folds")
    unlabeled_prediction = np.asarray(unlabeled_prediction, dtype=np.float64)
    if (
        unlabeled_prediction.ndim != 2
        or unlabeled_prediction.shape[0] != 3
        or unlabeled_prediction.shape[1] == 0
        or not np.isfinite(unlabeled_prediction).all()
    ):
        raise ValueError(
            "unlabeled_prediction must be a finite matrix of shape (3, n_unlabeled), n_unlabeled >= 1"
        )
    return _estimate(labeled_outcome, labeled_prediction, unlabeled_prediction, labeled_fold_id)


def fit_reppi(
    target_features: np.ndarray,
    target_source_prediction: np.ndarray,
    labeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
    *,
    model_factory: RegressorFactory,
    seed: int,
) -> RePPIResult:
    """Return a three-fold recalibrated target mean and ordered predictions.

    labeled_outcome follows labeled_indices; each factory call creates a fresh
    regressor. Unlabeled columns follow the returned ascending target indices.
    Folds have sizes n_labeled // 3, n_labeled // 3, and the remaining labels.
    """
    target_features = np.asarray(target_features, dtype=np.float64)
    if target_features.ndim != 2 or min(target_features.shape) == 0 or not np.isfinite(target_features).all():
        raise ValueError("target_features must be a nonempty finite matrix")
    n_target = len(target_features)
    target_source_prediction = _vector(target_source_prediction, "target_source_prediction", n_target)
    labeled_indices = np.asarray(labeled_indices)
    if (
        labeled_indices.ndim != 1
        or not np.issubdtype(labeled_indices.dtype, np.integer)
        or len(np.unique(labeled_indices)) != len(labeled_indices)
        or np.any((labeled_indices < 0) | (labeled_indices >= n_target))
    ):
        raise ValueError("labeled_indices must be a vector of unique integer target indices in range")
    n_labeled = len(labeled_indices)
    if not 6 <= n_labeled < n_target:
        raise ValueError("RePPI requires at least six labeled and one unlabeled target observations")
    labeled_outcome = _vector(labeled_outcome, "labeled_outcome", n_labeled)

    split_seed, *model_seeds = np.random.SeedSequence(seed).spawn(4)
    permutation = np.random.default_rng(split_seed).permutation(n_labeled)
    fold_size = n_labeled // 3
    labeled_fold_id = np.empty(n_labeled, dtype=np.int64)
    for fold_id, indices in enumerate(np.split(permutation, [fold_size, 2 * fold_size])):
        labeled_fold_id[indices] = fold_id

    unlabeled_mask = np.ones(n_target, dtype=bool)
    unlabeled_mask[labeled_indices] = False
    unlabeled_indices = np.flatnonzero(unlabeled_mask)
    recalibration_features = np.column_stack((target_features, target_source_prediction))
    labeled_features = recalibration_features[labeled_indices]
    unlabeled_features = recalibration_features[unlabeled_indices]
    labeled_prediction = np.empty(n_labeled, dtype=np.float64)
    unlabeled_prediction = np.empty((3, len(unlabeled_indices)), dtype=np.float64)
    for fold_id in range(3):
        training = labeled_fold_id == (fold_id + 2) % 3
        heldout = labeled_fold_id == fold_id
        model = model_factory(int(model_seeds[fold_id].generate_state(1)[0]))
        model.fit(labeled_features[training], labeled_outcome[training])
        labeled_prediction[heldout] = _vector(
            model.predict(labeled_features[heldout]),
            "labeled_prediction",
            int(heldout.sum()),
        )
        unlabeled_prediction[fold_id] = _vector(
            model.predict(unlabeled_features),
            "unlabeled_prediction",
            len(unlabeled_indices),
        )

    estimate = _estimate(labeled_outcome, labeled_prediction, unlabeled_prediction, labeled_fold_id)
    return RePPIResult(
        estimate=estimate.estimate,
        plugin_mean=estimate.plugin_mean,
        residual_correction=estimate.residual_correction,
        fold_estimate=estimate.fold_estimate,
        fold_coefficient=estimate.fold_coefficient,
        standard_error=estimate.standard_error,
        labeled_prediction=labeled_prediction,
        unlabeled_prediction=unlabeled_prediction,
        labeled_fold_id=labeled_fold_id,
        unlabeled_indices=unlabeled_indices,
    )
