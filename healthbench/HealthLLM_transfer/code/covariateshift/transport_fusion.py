"""Shared nuisance fitting and cross-fitting for CAFE."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from . import common_mean_regression as common
from .common_mean_regression import RegressorFactory
from .paired_fusion import fit_source_q
from .cafe_transport_inference import CAFEResult as TransportResult
from .cafe_transport_inference import _label_mask, _observed_target_scores, _vector, estimate_cafe, make_cafe_fold_ids


@dataclass(frozen=True)
class CAFEResult(TransportResult):
    """Store fusion inference, outcome predictions, density ratios, and coefficients."""

    source_m_prediction: np.ndarray
    target_m_prediction: np.ndarray
    source_w: np.ndarray
    target_a: np.ndarray
    source_q: np.ndarray
    source_a: np.ndarray


@dataclass(frozen=True)
class _CoefficientTrainingData:
    """Expose raw outer-training data for coefficient penalty selection."""

    outer_fold: int
    seed: int
    source_y: np.ndarray
    target_features: np.ndarray
    target_labeled: np.ndarray
    target_y: np.ndarray
    source_q: np.ndarray


def _fit_cafe_with_coefficient(
    source_features: np.ndarray,
    source_y: np.ndarray,
    target_features: np.ndarray,
    target_labeled: np.ndarray,
    target_y: np.ndarray,
    *,
    n_folds: int,
    n_inner_folds: int,
    seed: int,
    density_ratio: Mapping[str, Any],
    regressor_factory: RegressorFactory,
    source_q: np.ndarray | None,
    source_fold_id: np.ndarray | None,
    target_fold_id: np.ndarray | None,
    coefficient_fitter: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, _CoefficientTrainingData],
        Callable[[np.ndarray], np.ndarray],
    ],
) -> CAFEResult:
    """Fit coefficient functions on inner OOF residuals and evaluate outer held-out rows."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("Source and target features must be matrices")
    n_source, n_target = (len(source_features), len(target_features))
    source_y = _vector(source_y, n_source, "source_y")
    labeled = _label_mask(target_labeled, n_target)
    target_y = _observed_target_scores(target_y, labeled, "target_y")
    if (source_fold_id is None) != (target_fold_id is None):
        raise ValueError("Source and target fold IDs must be provided together")
    if source_fold_id is None:
        source_fold_id, target_fold_id = make_cafe_fold_ids(n_source, n_target, n_folds=n_folds, seed=seed)
    else:
        source_fold_id, target_fold_id = (np.asarray(source_fold_id), np.asarray(target_fold_id))
        for fold_ids, size in ((source_fold_id, n_source), (target_fold_id, n_target)):
            if (
                fold_ids.shape != (size,)
                or not np.issubdtype(fold_ids.dtype, np.integer)
                or (not np.array_equal(np.unique(fold_ids), np.arange(n_folds)))
                or (n_folds < 2)
            ):
                raise ValueError(
                    "Supplied fold IDs must align with samples and span n_folds starting at zero"
                )
    for fold_id in range(n_folds):
        if not np.any(labeled & (target_fold_id != fold_id)):
            raise ValueError(f"Fold {fold_id} training requires labeled target observations")
    if source_q is None:
        source_q = fit_source_q(source_features, target_features, density_ratio=density_ratio, seed=seed)
    else:
        source_q = _vector(source_q, n_source, "source_q")
        if np.any(source_q < 0) or not np.isclose(source_q.mean(), 1.0):
            raise ValueError("source_q must be nonnegative and normalized to full-source mean one")
    regression = common.fit_common_mean_crossfit(
        source_features,
        source_y,
        target_features,
        labeled_indices=np.flatnonzero(labeled),
        labeled_outcome=target_y[labeled],
        source_fold_ids=source_fold_id,
        target_fold_ids=target_fold_id,
        regressor_factory=regressor_factory,
        seed=seed,
    )
    source_a = np.empty(n_source)
    target_a = np.empty(n_target)
    for fold_id in range(n_folds):
        source_training = source_fold_id != fold_id
        target_training = target_fold_id != fold_id
        training_labeled = labeled[target_training]
        inner_seed = int(np.random.SeedSequence([seed, 701, fold_id]).generate_state(1)[0])
        inner_source_fold, inner_target_fold = make_cafe_fold_ids(
            int(source_training.sum()), int(target_training.sum()), n_folds=n_inner_folds, seed=inner_seed
        )
        inner_regression = common.fit_common_mean_crossfit(
            source_features[source_training],
            source_y[source_training],
            target_features[target_training],
            labeled_indices=np.flatnonzero(training_labeled),
            labeled_outcome=target_y[target_training][training_labeled],
            source_fold_ids=inner_source_fold,
            target_fold_ids=inner_target_fold,
            regressor_factory=regressor_factory,
            seed=inner_seed,
        )
        predict_coefficient = coefficient_fitter(
            source_features[source_training],
            target_features[target_training][training_labeled],
            target_y[target_training][training_labeled]
            - inner_regression.target_prediction[training_labeled],
            source_q[source_training] * (source_y[source_training] - inner_regression.source_prediction),
            _CoefficientTrainingData(
                fold_id,
                inner_seed,
                source_y[source_training],
                target_features[target_training],
                training_labeled,
                target_y[target_training],
                source_q[source_training],
            ),
        )
        source_a[~source_training] = predict_coefficient(source_features[~source_training])
        target_a[~target_training] = predict_coefficient(target_features[~target_training])
    source_w = source_a * source_q
    inference = estimate_cafe(
        source_y,
        labeled,
        target_y,
        source_m_prediction=regression.source_prediction,
        target_m_prediction=regression.target_prediction,
        source_w=source_w,
        target_a=target_a,
    )
    return CAFEResult(
        inference.alpha,
        inference.estimate,
        inference.asymptotic_variance,
        inference.standard_error,
        inference.source_contributions,
        inference.target_contributions,
        source_fold_id,
        target_fold_id,
        regression.source_prediction,
        regression.target_prediction,
        source_w,
        target_a,
        source_q,
        source_a,
    )
