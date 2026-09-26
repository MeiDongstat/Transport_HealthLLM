"""Common-outcome CAFE inference and observed-score validation."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from . import common_mean_regression as common
from .common_mean_regression import RegressorFactory as TabPFNFactory
from .common_mean_regression import make_fold_ids as make_cafe_fold_ids

Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]
Regressor = common.Regressor


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
    """Store CAFE inference and the zero-based source and target fold IDs."""

    source_fold_id: np.ndarray
    target_fold_id: np.ndarray


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


def estimate_cafe(
    source_y: np.ndarray,
    target_labeled: np.ndarray,
    target_y: np.ndarray,
    *,
    source_m_prediction: np.ndarray,
    target_m_prediction: np.ndarray,
    source_w: np.ndarray,
    target_a: np.ndarray,
) -> CAFEEstimate:
    """Return CAFE inference from ordered out-of-fold predictions and coefficients."""
    n_source, n_target = (len(source_y), len(target_labeled))
    if min(n_source, n_target) == 0:
        raise ValueError("CAFE requires nonempty source and target samples")
    source_y = _vector(source_y, n_source, "source_y")
    labeled = _label_mask(target_labeled, n_target)
    target_y = _observed_target_scores(target_y, labeled, "target_y")
    source_m_prediction = _vector(source_m_prediction, n_source, "source_m_prediction")
    target_m_prediction = _vector(target_m_prediction, n_target, "target_m_prediction")
    source_w = _vector(source_w, n_source, "source_w")
    target_a = _vector(target_a, n_target, "target_a")
    alpha = float(labeled.mean())
    source_contributions = source_w * (source_y - source_m_prediction)
    target_contributions = target_m_prediction.copy()
    target_contributions[labeled] += (
        (1 - target_a[labeled]) * (target_y[labeled] - target_m_prediction[labeled]) / alpha
    )
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
