"""Cross-fitted classifier density-ratio method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from covariateshift.estimators import (
    fit_tabpfn_domain_fold,
    make_domain_fold_ids,
    tabpfn_classifier_density_ratio,
)


@dataclass(frozen=True)
class DomainClassifierWeights:
    """Return weights, fold IDs, and pooled OOF domain probabilities."""

    raw_weights: np.ndarray
    weights: np.ndarray
    source_probability: np.ndarray
    target_probability: np.ndarray


def fit_domain_classifier_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    seed: int,
    classifier_factory: Any,
    n_folds: int,
    probability_clip: float | None,
) -> DomainClassifierWeights:
    """Return weights and source/target OOF domain probabilities."""
    pooled = np.vstack((source_features, target_features)).astype(np.float32)
    n_source = len(source_features)
    n_target = len(target_features)
    domains, fold_ids = make_domain_fold_ids(n_source, n_target, n_splits=n_folds, random_state=seed)
    probabilities = np.full(len(pooled), np.nan, dtype=np.float64)
    for fold_id in range(n_folds):
        heldout = fold_ids == fold_id
        probabilities[heldout] = fit_tabpfn_domain_fold(
            pooled,
            domains,
            fold_ids,
            fold_id,
            classifier_factory=classifier_factory,
            random_state=seed,
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("TabPFN did not predict every pooled row")
    estimate = tabpfn_classifier_density_ratio(
        probabilities[:n_source],
        n_source=n_source,
        n_target=n_target,
        probability_clip=probability_clip,
    )
    return DomainClassifierWeights(
        raw_weights=np.asarray(estimate.raw_weight, dtype=np.float64),
        weights=np.asarray(estimate.normalized_weight, dtype=np.float64),
        source_probability=probabilities[:n_source],
        target_probability=probabilities[n_source:],
    )
