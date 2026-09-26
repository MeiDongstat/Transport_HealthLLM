"""Kernel density-ratio methods for HealthBench covariate shift."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from covariateshift.estimators import (
    ImportanceWeightEstimate,
    fit_kernel_density_ratio,
    fit_kmm_density_ratio_cv,
)

METHOD_NAMES = ("ulsif", "rulsif", "kliep", "kmm")


@dataclass(frozen=True)
class KMMRatioPredictor:
    """Store a nonnegative kernel extension calibrated on training source rows."""

    source_features: np.ndarray
    source_weights: np.ndarray
    gamma: float
    normalizer: float
    weight_estimate: ImportanceWeightEstimate

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return density ratios at new rows on one shared source/target scale."""
        log_kernel = -self.gamma * cdist(features, self.source_features, "sqeuclidean")
        # Row shifts preserve kernel averages while avoiding exponential underflow.
        kernel = np.exp(log_kernel - log_kernel.max(axis=1, keepdims=True))
        return (kernel @ self.source_weights) / kernel.sum(axis=1) / self.normalizer


def fit_kmm_ratio_predictor(
    method_spec: Mapping[str, Any],
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> KMMRatioPredictor:
    """Return KMM ratios extended with the selected RBF kernel bandwidth."""
    if method_spec["name"] != "kmm":
        raise ValueError("The kernel extension requires KMM")
    estimate = fit_kernel_weights(
        method_spec,
        source_features,
        target_features,
        sigma=sigma,
        seed=seed,
    )
    predictor = KMMRatioPredictor(
        np.asarray(source_features, dtype=float),
        estimate.normalized_weight,
        float(estimate.selected_gamma),
        1.0,
        estimate,
    )
    return replace(predictor, normalizer=float(predictor.predict(source_features).mean()))


def fit_kernel_weights(
    method_spec: Mapping[str, Any],
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> ImportanceWeightEstimate:
    """Return source weights and selected parameters for one kernel method."""
    method_name = str(method_spec["name"])
    gamma_multipliers = [float(value) for value in method_spec["gamma_multipliers"]]
    if method_name == "kmm":
        estimate = fit_kmm_density_ratio_cv(
            source_features,
            target_features,
            random_state=seed,
            sigma=sigma,
            gamma_multipliers=gamma_multipliers,
            cv=int(method_spec["cv"]),
            B=float(method_spec["B"]),
            eps=method_spec["eps"],
            max_size=method_spec["max_size"],
            max_iter=int(method_spec["max_iter"]),
            scorer_bandwidth=str(method_spec.get("scorer_bandwidth", "median_gamma")),
        )
    else:
        fit_options: dict[str, Any] = {
            "random_state": seed,
            "n_centers": int(method_spec["n_centers"]),
            "sigma": sigma,
            "gamma_multipliers": gamma_multipliers,
        }
        if method_name == "ulsif":
            fit_options["lam"] = [float(value) for value in method_spec["lambdas"]]
        elif method_name == "rulsif":
            fit_options["lam"] = [float(value) for value in method_spec["lambda"]]
            fit_options["rulsif_alpha"] = float(method_spec["rulsif_alpha"])
        elif method_name == "kliep":
            fit_options["kliep_cv"] = int(method_spec["cv"])
            fit_options["kliep_algo"] = str(method_spec["algo"])
            fit_options["kliep_max_iter"] = int(method_spec["max_iter"])
        else:
            raise ValueError(f"Unknown kernel method: {method_name!r}")
        estimate = fit_kernel_density_ratio(method_name, source_features, target_features, **fit_options)
    weights = np.asarray(estimate.normalized_weight, dtype=np.float64)
    if (
        weights.shape != (len(source_features),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(weights.mean(), 1.0, rtol=1e-6, atol=1e-6)
    ):
        raise ValueError(f"Invalid {method_name} weights for seed {seed}")
    return estimate
