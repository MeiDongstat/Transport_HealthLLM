"""Shared nuisance fitting and cross-fitting for CAFE."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from covariateshift import kernel_based_methods
from covariateshift.common_mean_regression import RegressorFactory
from covariateshift.cafe_pairedscore_inference import CAFEResult as PairedScoreResult
from covariateshift.cafe_pairedscore_inference import (
    _label_mask,
    _observed_target_scores,
    _predict,
    _vector,
    estimate_cafe,
    make_cafe_fold_ids,
    make_cafe_model_seeds,
)
from covariateshift.estimators import median_sigma


@dataclass(frozen=True)
class CAFEResult(PairedScoreResult):
    """Store fusion inference and source-aligned density ratios and a."""

    source_q: np.ndarray
    source_a: np.ndarray


@dataclass(frozen=True)
class _CoefficientTrainingData:
    outer_fold: int
    seed: int
    source_b: np.ndarray
    target_features: np.ndarray
    target_labeled: np.ndarray
    target_b: np.ndarray
    target_y: np.ndarray
    source_q: np.ndarray


def _nuisance_predictions(
    source_features: np.ndarray,
    source_b: np.ndarray,
    target_features: np.ndarray,
    labeled: np.ndarray,
    target_b: np.ndarray,
    target_y: np.ndarray,
    *,
    source_fold_id: np.ndarray,
    target_fold_id: np.ndarray,
    n_folds: int,
    seed: int,
    regressor_factory: RegressorFactory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for fold_id in range(n_folds):
        if not np.any(labeled & (target_fold_id != fold_id)):
            raise ValueError(f"Fold {fold_id} training requires labeled target observations")
    model_seeds = make_cafe_model_seeds(n_folds=n_folds, seed=seed)
    source_b_prediction = np.empty(len(source_b))
    target_b_prediction, target_m_prediction = np.empty((2, len(target_features)))
    for fold_id in range(n_folds):
        source_heldout = source_fold_id == fold_id
        target_heldout = target_fold_id == fold_id
        labeled_training = labeled & ~target_heldout
        auxiliary_regressor = regressor_factory(model_seeds[2 * fold_id])
        auxiliary_regressor.fit(
            np.vstack((source_features[~source_heldout], target_features[labeled_training])),
            np.concatenate((source_b[~source_heldout], target_b[labeled_training])),
        )
        outcome_training = target_y[labeled_training]
        source_b_prediction[source_heldout] = _predict(auxiliary_regressor, source_features[source_heldout])
        target_b_prediction[target_heldout] = _predict(auxiliary_regressor, target_features[target_heldout])
        del auxiliary_regressor
        outcome_regressor = regressor_factory(model_seeds[2 * fold_id + 1])
        outcome_regressor.fit(target_features[labeled_training], outcome_training)
        target_m_prediction[target_heldout] = _predict(outcome_regressor, target_features[target_heldout])
        del outcome_regressor
    return (source_b_prediction, target_b_prediction, target_m_prediction)


def fit_source_q(
    source_features: np.ndarray, target_features: np.ndarray, *, density_ratio: Mapping[str, Any], seed: int
) -> np.ndarray:
    """Return mean-one KMM weights fitted to all source and target covariates."""
    if density_ratio["method"]["name"] != "kmm":
        raise ValueError("CAFE density ratios require KMM")
    density_seed = int(np.random.SeedSequence([seed, 700]).generate_state(1)[0])
    sigma = median_sigma(
        np.vstack((source_features, target_features)),
        max_points=int(density_ratio["median_max_points"]),
        seed=density_seed,
    )
    return kernel_based_methods.fit_kernel_weights(
        density_ratio["method"], source_features, target_features, sigma=sigma, seed=density_seed
    ).normalized_weight


def _fit_cafe_with_coefficient(
    source_features: np.ndarray,
    source_b: np.ndarray,
    target_features: np.ndarray,
    target_labeled: np.ndarray,
    target_b: np.ndarray,
    target_y: np.ndarray,
    *,
    n_folds: int,
    n_inner_folds: int,
    seed: int,
    density_ratio: Mapping[str, Any],
    regressor_factory: RegressorFactory,
    source_q: np.ndarray | None,
    coefficient_fitter: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, _CoefficientTrainingData],
        Callable[[np.ndarray], np.ndarray],
    ],
) -> CAFEResult:
    """Fit coefficient functions on inner OOF residuals and evaluate outer held-out rows."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("Source and target features must be matrices")
    n_source, n_target = (len(source_features), len(target_features))
    source_b = _vector(source_b, n_source, "source_b")
    labeled = _label_mask(target_labeled, n_target)
    target_b = _observed_target_scores(target_b, labeled, "target_b")
    target_y = _observed_target_scores(target_y, labeled, "target_y")
    source_fold_id, target_fold_id = make_cafe_fold_ids(
        n_source, n_target, n_folds=n_folds, seed=seed, target_labeled=labeled
    )
    if source_q is None:
        source_q = fit_source_q(source_features, target_features, density_ratio=density_ratio, seed=seed)
    else:
        source_q = _vector(source_q, n_source, "source_q")
        if np.any(source_q < 0) or not np.isclose(source_q.mean(), 1.0):
            raise ValueError("source_q must be nonnegative and normalized to full-source mean one")
    source_b_prediction, target_b_prediction, target_m_prediction = _nuisance_predictions(
        source_features,
        source_b,
        target_features,
        labeled,
        target_b,
        target_y,
        source_fold_id=source_fold_id,
        target_fold_id=target_fold_id,
        n_folds=n_folds,
        seed=seed,
        regressor_factory=regressor_factory,
    )
    source_a = np.empty(n_source)
    target_a = np.empty(n_target)
    for fold_id in range(n_folds):
        source_training = source_fold_id != fold_id
        target_training = target_fold_id != fold_id
        training_labeled = labeled[target_training]
        inner_seed = int(np.random.SeedSequence([seed, 701, fold_id]).generate_state(1)[0])
        inner_source_fold, inner_target_fold = make_cafe_fold_ids(
            int(source_training.sum()),
            int(target_training.sum()),
            n_folds=n_inner_folds,
            seed=inner_seed,
            target_labeled=training_labeled,
        )
        inner_source_b, inner_target_b, inner_target_m = _nuisance_predictions(
            source_features[source_training],
            source_b[source_training],
            target_features[target_training],
            training_labeled,
            target_b[target_training],
            target_y[target_training],
            source_fold_id=inner_source_fold,
            target_fold_id=inner_target_fold,
            n_folds=n_inner_folds,
            seed=inner_seed,
            regressor_factory=regressor_factory,
        )
        predict_coefficient = coefficient_fitter(
            source_features[source_training],
            target_features[target_training][training_labeled],
            target_b[target_training][training_labeled] - inner_target_b[training_labeled],
            target_y[target_training][training_labeled] - inner_target_m[training_labeled],
            source_q[source_training] * (source_b[source_training] - inner_source_b),
            _CoefficientTrainingData(
                fold_id,
                inner_seed,
                source_b[source_training],
                target_features[target_training],
                training_labeled,
                target_b[target_training],
                target_y[target_training],
                source_q[source_training],
            ),
        )
        source_a[~source_training] = predict_coefficient(source_features[~source_training])
        target_a[~target_training] = predict_coefficient(target_features[~target_training])
    source_w = source_a * source_q
    inference = estimate_cafe(
        source_b,
        labeled,
        target_b,
        target_y,
        source_b_prediction=source_b_prediction,
        target_b_prediction=target_b_prediction,
        target_m_prediction=target_m_prediction,
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
        source_b_prediction,
        target_b_prediction,
        target_m_prediction,
        source_w,
        target_a,
        source_q,
        source_a,
    )
