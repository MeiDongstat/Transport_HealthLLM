"""Paired-score CAFE inference, RKHS utilities, and cross-fitting partitions."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np
from scipy.linalg import eigh

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Regressor(Protocol):
    """Describe the fit/predict interface of a configured TabPFN regressor."""

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> Any:
        """Return the regressor fitted to the supplied observations."""
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return one predicted conditional mean per row."""
        ...


TabPFNFactory = Callable[[int], Regressor]


@dataclass(frozen=True)
class RKHSFunction:
    """Store a function expanded in kernel sections at its training centers."""

    centers: np.ndarray
    coefficients: np.ndarray
    kernel: Kernel

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return function values at the supplied covariates."""
        return _kernel_matrix(self.kernel, features, self.centers) @ self.coefficients


@dataclass(frozen=True)
class CAFEEstimate:
    """Store the empirical labeling probability, inference, and ordered contributions."""

    alpha: float
    estimate: float
    asymptotic_variance: float
    standard_error: float
    source_contributions: np.ndarray
    target_contributions: np.ndarray


@dataclass(frozen=True)
class CAFEResult(CAFEEstimate):
    """Store CAFE inference, held-out predictions, coefficients, and fold IDs."""

    source_fold_id: np.ndarray
    target_fold_id: np.ndarray
    source_b_prediction: np.ndarray
    target_b_prediction: np.ndarray
    target_m_prediction: np.ndarray
    source_w: np.ndarray
    target_a: np.ndarray


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return values


def _label_mask(target_labeled: np.ndarray, n_target: int) -> np.ndarray:
    target_labeled = np.asarray(target_labeled)
    if target_labeled.shape != (n_target,) or not np.isin(target_labeled, [0, 1]).all():
        raise ValueError("target_labeled must contain one Boolean or 0/1 value per target row")
    if not target_labeled.any():
        raise ValueError("CAFE requires labeled target observations")
    return target_labeled.astype(bool)


def _observed_target_scores(scores: np.ndarray, labeled: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores[labeled]).all():
        raise ValueError(f"{name} must be finite at labeled positions")
    return scores


def _kernel_matrix(kernel: Kernel, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    gram = np.asarray(kernel(first, second), dtype=np.float64)
    if gram.shape != (len(first), len(second)):
        raise ValueError("Kernel must return a matrix with one entry per row pair")
    if not np.isfinite(gram).all():
        raise ValueError("Kernel matrix must not contain infs or NaNs")
    return gram


def _rkhs_basis(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tolerance = np.finfo(np.float64).eps * len(gram) * np.max(np.abs(gram))
    if np.max(np.abs(gram - gram.T)) > tolerance:
        raise ValueError("Kernel Gram matrix must be symmetric")
    eigenvalues, eigenvectors = eigh(gram)
    tolerance = np.finfo(np.float64).eps * len(gram) * np.max(np.abs(eigenvalues))
    if eigenvalues[0] < -tolerance:
        raise ValueError("Kernel Gram matrix must be positive semidefinite")
    positive = eigenvalues > tolerance
    roots = np.sqrt(eigenvalues[positive])
    return (eigenvectors[:, positive] * roots, eigenvectors[:, positive] / roots)


def _stratified_fold_ids(
    size: int, *, n_folds: int, seed: np.random.SeedSequence, labeled: np.ndarray | None
) -> np.ndarray:
    if labeled is None:
        strata, stratum_seeds = ((np.arange(size),), (seed,))
    else:
        strata = (np.flatnonzero(labeled), np.flatnonzero(~labeled))
        stratum_seeds = seed.spawn(2)
    fold_ids = np.empty(size, dtype=np.int64)
    offset = 0
    for stratum, stratum_seed in zip(strata, stratum_seeds, strict=True):
        permutation = np.random.default_rng(stratum_seed).permutation(stratum)
        for fold_id, indices in enumerate(np.array_split(permutation, n_folds)):
            fold_ids[indices] = (fold_id + offset) % n_folds
        offset = (offset + len(stratum)) % n_folds
    return fold_ids


def make_target_fold_ids(
    n_target: int, *, n_folds: int, seed: int, target_labeled: np.ndarray | None = None
) -> np.ndarray:
    """Return paired-score target folds independent of the source sample size."""
    if not 2 <= n_folds <= n_target:
        raise ValueError("n_folds must be between 2 and the target sample size")
    labeled = None if target_labeled is None else _label_mask(target_labeled, n_target)
    target_seed = np.random.SeedSequence(seed).spawn(2)[1]
    return _stratified_fold_ids(n_target, n_folds=n_folds, seed=target_seed, labeled=labeled)


def make_cafe_fold_ids(
    n_source: int, n_target: int, *, n_folds: int, seed: int, target_labeled: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced folds, stratifying target by its label mask when supplied.

    Source, labeled target, and unlabeled target are shuffled independently.
    Within each stratum, fold sizes differ by at most one. Returned arrays
    retain input order. Omitting target_labeled partitions source and target only.
    """
    if not 2 <= n_folds <= min(n_source, n_target):
        raise ValueError("n_folds must be between 2 and both sample sizes")
    source_seed = np.random.SeedSequence(seed).spawn(2)[0]
    source_folds = _stratified_fold_ids(n_source, n_folds=n_folds, seed=source_seed, labeled=None)
    target_folds = make_target_fold_ids(n_target, n_folds=n_folds, seed=seed, target_labeled=target_labeled)
    return (source_folds, target_folds)


def make_cafe_model_seeds(*, n_folds: int, seed: int) -> list[int]:
    """Return alternating B and Y regression seeds for each paired-score outer fold."""
    nuisance_seed = np.random.SeedSequence(seed).spawn(3)[2]
    return [int(child.generate_state(1)[0]) for child in nuisance_seed.spawn(2 * n_folds)]


def estimate_cafe(
    source_b: np.ndarray,
    target_labeled: np.ndarray,
    target_b: np.ndarray,
    target_y: np.ndarray,
    *,
    source_b_prediction: np.ndarray,
    target_b_prediction: np.ndarray,
    target_m_prediction: np.ndarray,
    source_w: np.ndarray,
    target_a: np.ndarray,
) -> CAFEEstimate:
    """Return CAFE inference from ordered out-of-fold predictions and coefficients."""
    n_source, n_target = (len(source_b), len(target_labeled))
    if min(n_source, n_target) == 0:
        raise ValueError("CAFE requires nonempty source and target samples")
    source_b = _vector(source_b, n_source, "source_b")
    labeled = _label_mask(target_labeled, n_target)
    target_b = _observed_target_scores(target_b, labeled, "target_b")
    target_y = _observed_target_scores(target_y, labeled, "target_y")
    source_b_prediction = _vector(source_b_prediction, n_source, "source_b_prediction")
    target_b_prediction = _vector(target_b_prediction, n_target, "target_b_prediction")
    target_m_prediction = _vector(target_m_prediction, n_target, "target_m_prediction")
    source_w = _vector(source_w, n_source, "source_w")
    target_a = _vector(target_a, n_target, "target_a")
    alpha = float(labeled.mean())
    source_contributions = source_w * (source_b - source_b_prediction)
    target_contributions = target_m_prediction.copy()
    target_contributions[labeled] += (
        target_y[labeled]
        - target_m_prediction[labeled]
        - target_a[labeled] * (target_b[labeled] - target_b_prediction[labeled])
    ) / alpha
    estimate = float(target_contributions.mean() + source_contributions.mean())
    sample_size = n_source + n_target
    asymptotic_variance = float(
        np.mean((target_contributions - estimate) ** 2) / (n_target / sample_size)
        + np.mean(source_contributions**2) / (n_source / sample_size)
    )
    return CAFEEstimate(
        alpha,
        estimate,
        asymptotic_variance,
        np.sqrt(asymptotic_variance / sample_size),
        source_contributions,
        target_contributions,
    )


def _predict(regressor: Regressor, features: np.ndarray) -> np.ndarray:
    return _vector(regressor.predict(features), len(features), "TabPFN prediction")
