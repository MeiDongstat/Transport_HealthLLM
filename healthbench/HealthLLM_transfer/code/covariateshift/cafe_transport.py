"""Cross-fitted common-mean fusion with RKHS a(X) and source weights w = a q.

Each outer fold uses inner out-of-fold residuals to minimize
mean_L(((1 - a(X)) rY)^2) + (nL/nS) mean_S((a(X) q rS)^2)
+ lambda_a ||a||_H^2. Outcome regressions pool source and labeled target data.
Density ratios are direct KMM weights fitted to all source and target
covariates; every outer, inner and validation fold retains their full-source scale.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any
import numpy as np
from scipy.spatial.distance import cdist
from . import common_mean_regression as common
from .common_mean_regression import RegressorFactory
from .cafe_pairedscore_inference import Kernel, RKHSFunction, make_cafe_fold_ids
from .cafe_pairedscore import fit_rkhs_coefficient as fit_paired_rkhs_coefficient
from .cafe_pairedscore import prepare_rkhs_coefficient, solve_rkhs_coefficient
from .transport_fusion import CAFEResult, _CoefficientTrainingData, _fit_cafe_with_coefficient
from .estimators import median_sigma


def fit_rkhs_coefficient(
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_y_residual: np.ndarray,
    source_weighted_residual: np.ndarray,
    *,
    coefficient_kernel: Kernel,
    lambda_a: float,
) -> RKHSFunction:
    """Return the common-mean RKHS minimizer using labeled-only target rows.

    source_weighted_residual already includes q with full-source normalization.
    Setting both paired-score target residuals to rY gives the (1-a) rY loss.
    """
    return fit_paired_rkhs_coefficient(
        source_features,
        target_features,
        target_y_residual,
        target_y_residual,
        source_weighted_residual,
        coefficient_kernel=coefficient_kernel,
        lambda_a=lambda_a,
    )


def fit_cafe(
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
    coefficient_kernel: Kernel,
    lambda_a: float,
    source_q: np.ndarray | None = None,
    source_fold_id: np.ndarray | None = None,
    target_fold_id: np.ndarray | None = None,
) -> CAFEResult:
    """Return common-mean CAFE with full-sample KMM weights and cross-fitted RKHS a(X).

    q is estimated separately from a using all source and target covariates,
    including evaluation-fold covariates.
    coefficient_kernel must be symmetric positive semidefinite; lambda_a is
    its fixed positive regularization strength. Pooled outcome regressions,
    full-covariate KMM weights, and fold seeds follow transport_fusion.
    Optional source_q must have full-source mean one and match source row order.
    Supplied source_fold_id and target_fold_id must be provided together
    with n_folds nonempty folds.
    """

    def fit_coefficient(
        source: np.ndarray,
        target: np.ndarray,
        target_y_residual: np.ndarray,
        source_weighted_residual: np.ndarray,
        training: _CoefficientTrainingData,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return fit_rkhs_coefficient(
            source,
            target,
            target_y_residual,
            source_weighted_residual,
            coefficient_kernel=coefficient_kernel,
            lambda_a=lambda_a,
        ).predict

    return _fit_cafe_with_coefficient(
        source_features,
        source_y,
        target_features,
        target_labeled,
        target_y,
        n_folds=n_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        density_ratio=density_ratio,
        regressor_factory=regressor_factory,
        source_q=source_q,
        source_fold_id=source_fold_id,
        target_fold_id=target_fold_id,
        coefficient_fitter=fit_coefficient,
    )


def _select_penalty(
    source: np.ndarray,
    training: _CoefficientTrainingData,
    *,
    coefficient_kernel: Kernel,
    candidates: np.ndarray,
    n_folds: int,
    regressor_factory: RegressorFactory,
) -> tuple[float, np.ndarray]:
    """Return the penalty minimizing mean held-out, unpenalized residual-square loss."""
    target, labeled = (training.target_features, training.target_labeled)
    if labeled.sum() < n_folds:
        raise ValueError("Coefficient cross-validation requires at least one target label per inner fold")
    source_folds, target_folds = make_cafe_fold_ids(
        len(source), len(target), n_folds=n_folds, seed=training.seed, target_labeled=labeled
    )
    losses = np.empty((n_folds, len(candidates)))

    def evaluate_fold(fold: common.CommonMeanFold) -> None:
        target_train = labeled & fold.target_training
        target_eval = labeled & fold.target_heldout
        train_source_residual = training.source_q[fold.source_training] * (
            training.source_y[fold.source_training]
            - common.predict_mean(fold.model, source[fold.source_training])
        )
        train_target_residual = training.target_y[target_train] - common.predict_mean(
            fold.model, target[target_train]
        )
        prepared = prepare_rkhs_coefficient(
            source[fold.source_training],
            target[target_train],
            train_target_residual,
            train_target_residual,
            train_source_residual,
            coefficient_kernel=coefficient_kernel,
        )
        validation_source_residual = training.source_q[fold.source_heldout] * (
            training.source_y[fold.source_heldout] - fold.source_prediction
        )
        validation_target_residual = (
            training.target_y[target_eval] - fold.target_prediction[labeled[fold.target_heldout]]
        )
        source_scale = target_train.sum() / fold.source_training.sum()
        for index, penalty in enumerate(candidates):
            coefficient = solve_rkhs_coefficient(prepared, lambda_a=float(penalty))
            target_correction = (1 - coefficient.predict(target[target_eval])) * validation_target_residual
            source_correction = coefficient.predict(source[fold.source_heldout]) * validation_source_residual
            losses[fold.fold_id, index] = np.mean(target_correction**2) + source_scale * np.mean(
                source_correction**2
            )

    common.fit_common_mean_crossfit(
        source,
        training.source_y,
        target,
        labeled_indices=np.flatnonzero(labeled),
        labeled_outcome=training.target_y[labeled],
        source_fold_ids=source_folds,
        target_fold_ids=target_folds,
        regressor_factory=regressor_factory,
        seed=training.seed,
        on_fold=evaluate_fold,
    )
    if not np.isfinite(losses).all():
        raise ValueError("RKHS coefficient cross-validation losses must be finite")
    selected = int(np.argmin(losses.mean(axis=0)))
    return (float(candidates[selected]), losses)


def fit_cafe_cv(
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
    coefficient_spec: Mapping[str, Any],
    source_q: np.ndarray | None = None,
    source_fold_id: np.ndarray | None = None,
    target_fold_id: np.ndarray | None = None,
) -> tuple[CAFEResult, tuple[dict[str, Any], ...]]:
    """Return Gaussian-RKHS CAFE and outer-fold diagnostics after inner penalty selection.

    coefficient_spec requires median_max_points, n_inner_folds and a positive
    lambda_a grid. Each outer fold sets sigma to the median distance of its
    source and all target training covariates. Inner target folds are label
    stratified. Each inner learner excludes its validation labels; coefficient
    training uses that learner's training residuals, and validation uses its
    held-out residuals. Training and validation use the corresponding slices
    of the fixed full-source KMM weights. Validation retains the training nL/nS
    domain weight, excludes the RKHS penalty, and averages
    losses equally across inner folds.
    Ties select the first grid value. The final coefficient uses the complete
    inner out-of-fold residuals with the constant method's folds and seeds.
    """
    candidates = np.asarray(coefficient_spec["lambda_a"], dtype=np.float64)
    if (
        candidates.ndim != 1
        or len(candidates) == 0
        or (not np.all(np.isfinite(candidates) & (candidates > 0)))
    ):
        raise ValueError("lambda_a grid must be nonempty, finite, and positive")
    cv_folds = coefficient_spec["n_inner_folds"]
    if not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
        raise ValueError("Coefficient n_inner_folds must be an integer of at least two")
    diagnostics = []

    def fit_coefficient(
        source: np.ndarray,
        target: np.ndarray,
        target_y_residual: np.ndarray,
        source_weighted_residual: np.ndarray,
        training: _CoefficientTrainingData,
    ) -> Callable[[np.ndarray], np.ndarray]:
        sigma = median_sigma(
            np.vstack((source, training.target_features)),
            max_points=int(coefficient_spec["median_max_points"]),
            seed=training.seed,
        )

        def coefficient_kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            return np.exp(-cdist(first, second, "sqeuclidean") / (2 * sigma**2))

        penalty, losses = _select_penalty(
            source,
            training,
            coefficient_kernel=coefficient_kernel,
            candidates=candidates,
            n_folds=cv_folds,
            regressor_factory=regressor_factory,
        )
        diagnostics.append(
            {
                "outer_fold": training.outer_fold,
                "selected_lambda_a": penalty,
                "sigma": sigma,
                "lambda_grid": candidates.tolist(),
                "validation_losses": losses.mean(axis=0).tolist(),
                "inner_validation_losses": losses.tolist(),
                "inner_folds": int(cv_folds),
            }
        )
        return fit_rkhs_coefficient(
            source,
            target,
            target_y_residual,
            source_weighted_residual,
            coefficient_kernel=coefficient_kernel,
            lambda_a=penalty,
        ).predict

    result = _fit_cafe_with_coefficient(
        source_features,
        source_y,
        target_features,
        target_labeled,
        target_y,
        n_folds=n_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        density_ratio=density_ratio,
        regressor_factory=regressor_factory,
        source_q=source_q,
        source_fold_id=source_fold_id,
        target_fold_id=target_fold_id,
        coefficient_fitter=fit_coefficient,
    )
    return (result, tuple(diagnostics))
