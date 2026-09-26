"""Cross-fitted paired-score fusion with RKHS a(X) and source weights w = a q.

Each outer fold uses inner out-of-fold residuals to minimize
mean_L((rY - a(X) rB)^2) + (nL/nS) mean_S((a(X) q rS)^2)
+ lambda_a ||a||_H^2. The coefficient kernel and positive penalty are supplied
by the caller. Density ratios are direct KMM weights fitted to all source and
target covariates. Every fold retains their full-source normalization.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from .common_mean_regression import RegressorFactory
from .cafe_pairedscore_inference import (
    Kernel,
    RKHSFunction,
    _kernel_matrix,
    _predict,
    _rkhs_basis,
    _vector,
    make_cafe_fold_ids,
    make_cafe_model_seeds,
)
from .paired_fusion import CAFEResult, _CoefficientTrainingData, _fit_cafe_with_coefficient
from .estimators import median_sigma


@dataclass(frozen=True)
class PreparedRKHSCoefficient:
    """Store one residual-square system in orthonormal RKHS coordinates."""

    hessian: np.ndarray
    right_hand_side: np.ndarray
    coefficient_map: np.ndarray
    centers: np.ndarray
    kernel: Kernel


def prepare_rkhs_coefficient(
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_b_residual: np.ndarray,
    target_y_residual: np.ndarray,
    source_weighted_residual: np.ndarray,
    *,
    coefficient_kernel: Kernel,
) -> PreparedRKHSCoefficient:
    """Return the penalty-independent residual-square system for an RKHS fit.

    target_features contains labeled target training rows only. The source
    residual vector already includes q with full-source normalization.
    The RKHS norm penalizes every direction, including a constant direction
    if the supplied kernel contains one.
    """
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if (
        source_features.ndim != 2
        or target_features.ndim != 2
        or source_features.shape[1] != target_features.shape[1]
    ):
        raise ValueError("Source and labeled target features must have matching covariate columns")
    n_source, n_labeled = (len(source_features), len(target_features))
    if min(n_source, n_labeled) == 0:
        raise ValueError("RKHS coefficient requires source and labeled target residuals")
    target_b_residual = _vector(target_b_residual, n_labeled, "target_b_residual")
    target_y_residual = _vector(target_y_residual, n_labeled, "target_y_residual")
    source_weighted_residual = _vector(source_weighted_residual, n_source, "source_weighted_residual")
    centers = np.vstack((source_features, target_features))
    basis, coefficient_map = _rkhs_basis(_kernel_matrix(coefficient_kernel, centers, centers))
    source_design = source_weighted_residual[:, None] * basis[:n_source]
    target_design = target_b_residual[:, None] * basis[n_source:]
    hessian = target_design.T @ target_design / n_labeled
    hessian += n_labeled / n_source**2 * (source_design.T @ source_design)
    right_hand_side = target_design.T @ target_y_residual / n_labeled
    return PreparedRKHSCoefficient(hessian, right_hand_side, coefficient_map, centers, coefficient_kernel)


def solve_rkhs_coefficient(prepared: PreparedRKHSCoefficient, *, lambda_a: float) -> RKHSFunction:
    """Return the regularized minimizer of a prepared residual-square system."""
    if not np.isfinite(lambda_a) or lambda_a <= 0:
        raise ValueError("lambda_a must be finite and positive")
    hessian = prepared.hessian.copy()
    hessian[np.diag_indices_from(hessian)] += lambda_a
    coordinates = solve(hessian, prepared.right_hand_side, assume_a="pos")
    return RKHSFunction(prepared.centers, prepared.coefficient_map @ coordinates, prepared.kernel)


def fit_rkhs_coefficient(
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_b_residual: np.ndarray,
    target_y_residual: np.ndarray,
    source_weighted_residual: np.ndarray,
    *,
    coefficient_kernel: Kernel,
    lambda_a: float,
) -> RKHSFunction:
    """Return the penalized RKHS fit using labeled target rows and source residuals multiplied by q."""
    prepared = prepare_rkhs_coefficient(
        source_features,
        target_features,
        target_b_residual,
        target_y_residual,
        source_weighted_residual,
        coefficient_kernel=coefficient_kernel,
    )
    return solve_rkhs_coefficient(prepared, lambda_a=lambda_a)


def fit_cafe(
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
    coefficient_kernel: Kernel,
    lambda_a: float,
    source_q: np.ndarray | None = None,
) -> CAFEResult:
    """Return paired-score CAFE with full-sample KMM weights and cross-fitted RKHS a(X).

    q is estimated separately from a using all source and target covariates,
    including evaluation-fold covariates.
    coefficient_kernel must be symmetric positive semidefinite; lambda_a is
    its fixed positive regularization strength. Nuisance predictions and fold
    seeds follow paired_fusion. Optional source_q contains full-source
    mean-one KMM weights in source row order.
    """

    def fit_coefficient(
        source: np.ndarray,
        target: np.ndarray,
        target_b_residual: np.ndarray,
        target_y_residual: np.ndarray,
        source_weighted_residual: np.ndarray,
        training: _CoefficientTrainingData,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return fit_rkhs_coefficient(
            source,
            target,
            target_b_residual,
            target_y_residual,
            source_weighted_residual,
            coefficient_kernel=coefficient_kernel,
            lambda_a=lambda_a,
        ).predict

    return _fit_cafe_with_coefficient(
        source_features,
        source_b,
        target_features,
        target_labeled,
        target_b,
        target_y,
        n_folds=n_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        density_ratio=density_ratio,
        regressor_factory=regressor_factory,
        source_q=source_q,
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
    """Return the penalty minimizing held-out paired residual-square loss."""
    target, labeled = (training.target_features, training.target_labeled)
    if labeled.sum() < n_folds:
        raise ValueError("Coefficient cross-validation requires a target label in each inner fold")
    source_folds, target_folds = make_cafe_fold_ids(
        len(source), len(target), n_folds=n_folds, seed=training.seed, target_labeled=labeled
    )
    seeds = make_cafe_model_seeds(n_folds=n_folds, seed=training.seed)
    losses = np.empty((n_folds, len(candidates)))
    for fold in range(n_folds):
        source_eval, target_eval = (source_folds == fold, target_folds == fold)
        training_labels, validation_labels = (labeled & ~target_eval, labeled & target_eval)
        auxiliary = regressor_factory(seeds[2 * fold])
        auxiliary.fit(
            np.vstack((source[~source_eval], target[training_labels])),
            np.r_[training.source_b[~source_eval], training.target_b[training_labels]],
        )
        outcome = regressor_factory(seeds[2 * fold + 1])
        outcome.fit(target[training_labels], training.target_y[training_labels])
        source_residual = training.source_q * (training.source_b - _predict(auxiliary, source))
        target_b_residual = training.target_b[labeled] - _predict(auxiliary, target[labeled])
        target_y_residual = training.target_y[labeled] - _predict(outcome, target[labeled])
        del auxiliary, outcome
        label_eval = target_eval[labeled]
        prepared = prepare_rkhs_coefficient(
            source[~source_eval],
            target[training_labels],
            target_b_residual[~label_eval],
            target_y_residual[~label_eval],
            source_residual[~source_eval],
            coefficient_kernel=coefficient_kernel,
        )
        source_scale = training_labels.sum() / (~source_eval).sum()
        for index, penalty in enumerate(candidates):
            fitted = solve_rkhs_coefficient(prepared, lambda_a=float(penalty))
            target_correction = (
                target_y_residual[label_eval]
                - fitted.predict(target[validation_labels]) * target_b_residual[label_eval]
            )
            source_correction = fitted.predict(source[source_eval]) * source_residual[source_eval]
            losses[fold, index] = np.mean(target_correction**2) + source_scale * np.mean(source_correction**2)
    if not np.isfinite(losses).all():
        raise ValueError("RKHS coefficient cross-validation losses must be finite")
    selected = int(np.argmin(losses.mean(axis=0)))
    return (float(candidates[selected]), losses)


def fit_cafe_cv(
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
    coefficient_spec: Mapping[str, Any],
    source_q: np.ndarray | None = None,
) -> tuple[CAFEResult, tuple[dict[str, Any], ...]]:
    """Return Gaussian-RKHS paired CAFE and outer-fold penalty-selection diagnostics.

    The Gaussian bandwidth uses outer-training source and all target X.
    Inner learners use training labels only, and validation uses held-out
    residuals with the training nL/nS weight and fixed full-source KMM q.
    Mean unpenalized validation loss selects lambda_a; ties use the first
    candidate. Final coefficients use the complete inner OOF residuals.
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
        target_b_residual: np.ndarray,
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
            target_b_residual,
            target_y_residual,
            source_weighted_residual,
            coefficient_kernel=coefficient_kernel,
            lambda_a=penalty,
        ).predict

    result = _fit_cafe_with_coefficient(
        source_features,
        source_b,
        target_features,
        target_labeled,
        target_b,
        target_y,
        n_folds=n_folds,
        n_inner_folds=n_inner_folds,
        seed=seed,
        density_ratio=density_ratio,
        regressor_factory=regressor_factory,
        source_q=source_q,
        coefficient_fitter=fit_coefficient,
    )
    return (result, tuple(diagnostics))
