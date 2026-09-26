"""Predicted-theme-probability plus embedding kernel methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from covariateshift.estimators import median_sigma
from covariateshift.kernel_based_methods import fit_kernel_weights
from covariateshift.predicted_theme import (
    aligned_theme_probabilities,
    theme_fold_ids,
)


def fit_predicted_theme_plus_embedding_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_themes: np.ndarray,
    theme_labels: Sequence[str],
    methods: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    classifier_factory: Any,
    n_folds: int,
    median_max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return full-source OOF/target-average kernel weights, folds, and bandwidth."""
    source = np.ascontiguousarray(source_features, dtype=np.float32)
    target = np.ascontiguousarray(target_features, dtype=np.float32)
    themes = np.asarray(source_themes).astype(str)
    labels = np.asarray(theme_labels).astype(str)
    folds = theme_fold_ids(themes, seed=seed, n_folds=n_folds, one_based=True)
    source_probability_oof = np.full((len(source), len(labels)), np.nan, dtype=np.float64)
    target_probability_by_fold = np.full((n_folds, len(target), len(labels)), np.nan, dtype=np.float64)

    for fold_id in range(1, n_folds + 1):
        training = folds != fold_id
        heldout = folds == fold_id
        fold_seed = seed + fold_id - 1
        classifier = classifier_factory(fold_seed)
        classifier.fit(source[training], themes[training])
        source_probability_oof[heldout] = aligned_theme_probabilities(classifier, source[heldout], labels)
        target_probability_by_fold[fold_id - 1] = aligned_theme_probabilities(classifier, target, labels)

    target_probability_average = target_probability_by_fold.mean(axis=0)
    source_hybrid = np.column_stack((source, source_probability_oof))
    target_hybrid = np.column_stack((target, target_probability_average))
    sigma = median_sigma(
        np.vstack((source_hybrid, target_hybrid)),
        max_points=median_max_points,
        seed=seed,
    )
    estimates = [
        fit_kernel_weights(
            method,
            source_hybrid,
            target_hybrid,
            sigma=sigma,
            seed=seed,
        )
        for method in methods
    ]
    raw_weights = np.stack([np.asarray(estimate.raw_weight, dtype=np.float64) for estimate in estimates])
    weights = np.stack([np.asarray(estimate.normalized_weight, dtype=np.float64) for estimate in estimates])
    if not np.allclose(weights.mean(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Kernel weights do not have source mean one")
    return raw_weights, weights, folds, float(sigma)
