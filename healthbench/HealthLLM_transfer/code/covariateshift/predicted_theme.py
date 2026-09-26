"""Predicted-theme Hájek and probability-KMM methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from covariateshift.estimators import fit_kmm_density_ratio_cv, median_sigma
from sklearn.model_selection import StratifiedKFold

METHOD_NAMES = ("predicted_theme_hajek", "predicted_probability_kmm")


@dataclass(frozen=True)
class PredictedThemeWeights:
    """Return source-aligned predicted-theme outputs for one split."""

    source_fold_id: np.ndarray
    raw_weights: np.ndarray
    weights: np.ndarray
    kmm_sigma: float
    selected_gamma_multiplier: float
    selected_gamma: float


def aligned_theme_probabilities(classifier: Any, features: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return classifier probabilities in the configured theme order."""
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    classes = np.asarray(classifier.classes_).astype(str)
    label_array = np.asarray(labels).astype(str)
    if probabilities.shape != (len(features), len(label_array)):
        raise ValueError("TabPFN theme probability shape is invalid")
    if set(classes) != set(label_array):
        raise ValueError("TabPFN theme classes differ from configured classes")
    class_column = {label: index for index, label in enumerate(classes)}
    aligned = probabilities[:, [class_column[label] for label in label_array]]
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0):
        raise ValueError("TabPFN returned invalid theme probabilities")
    if not np.allclose(aligned.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("TabPFN theme probabilities do not sum to one")
    return aligned


def predicted_theme_hajek_weights(
    source_themes: np.ndarray,
    target_probabilities: np.ndarray,
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and mean-one Hájek weights from target argmax themes."""
    source = np.asarray(source_themes).astype(str)
    probabilities = np.asarray(target_probabilities, dtype=np.float64)
    label_array = np.asarray(labels).astype(str)
    source_proportions = np.asarray([np.mean(source == label) for label in label_array], dtype=np.float64)
    if np.any(source_proportions == 0.0):
        raise ValueError("Every configured theme must occur in the source")
    target_codes = probabilities.argmax(axis=1)
    target_proportions = np.bincount(target_codes, minlength=len(label_array)).astype(np.float64) / len(
        target_codes
    )
    label_code = {label: code for code, label in enumerate(label_array)}
    source_codes = np.asarray([label_code[label] for label in source], dtype=np.int64)
    raw_weights = (target_proportions / source_proportions)[source_codes]
    return raw_weights, raw_weights / raw_weights.mean()


def theme_fold_ids(
    source_themes: np.ndarray, *, seed: int, n_folds: int, one_based: bool = False
) -> np.ndarray:
    """Return deterministic source-theme-stratified held-out fold IDs."""
    source = np.asarray(source_themes).astype(str)
    offset = 1 if one_based else 0
    fold_ids = np.full(len(source), -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (_, heldout) in enumerate(splitter.split(np.zeros((len(source), 1)), source), start=offset):
        fold_ids[heldout] = fold_id
    assert np.all(fold_ids >= offset)
    return fold_ids


def fit_predicted_theme_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_themes: np.ndarray,
    labels: Sequence[str],
    *,
    seed: int,
    classifier_factory: Any,
    n_folds: int,
    median_max_points: int,
    kmm_spec: Mapping[str, Any],
) -> PredictedThemeWeights:
    """Return Hájek and source-OOF/target-average probability-KMM weights."""
    source_features = np.asarray(source_features, dtype=np.float32)
    target_features = np.asarray(target_features, dtype=np.float32)
    full_classifier = classifier_factory(seed)
    full_classifier.fit(source_features, source_themes)
    target_probability = aligned_theme_probabilities(full_classifier, target_features, labels)
    hajek_raw, hajek_weights = predicted_theme_hajek_weights(source_themes, target_probability, labels)

    source_fold_id = theme_fold_ids(source_themes, seed=seed, n_folds=n_folds)
    source_probability_oof = np.full((len(source_features), len(labels)), np.nan, dtype=np.float64)
    target_probability_by_fold = np.full(
        (n_folds, len(target_features), len(labels)), np.nan, dtype=np.float64
    )
    gamma_multipliers = [float(value) for value in kmm_spec["gamma_multipliers"]]
    for fold_id in range(n_folds):
        heldout = source_fold_id == fold_id
        training = ~heldout
        fold_seed = seed + fold_id
        classifier = classifier_factory(fold_seed)
        classifier.fit(source_features[training], source_themes[training])
        source_probability_oof[heldout] = aligned_theme_probabilities(
            classifier, source_features[heldout], labels
        )
        target_probability_by_fold[fold_id] = aligned_theme_probabilities(classifier, target_features, labels)

    target_probability_average = target_probability_by_fold.mean(axis=0)
    sigma = median_sigma(
        np.vstack((source_probability_oof, target_probability_average)),
        max_points=median_max_points,
        seed=seed,
    )
    estimate = fit_kmm_density_ratio_cv(
        source_probability_oof,
        target_probability_average,
        random_state=seed,
        sigma=sigma,
        gamma_multipliers=gamma_multipliers,
        cv=int(kmm_spec["cv"]),
        B=float(kmm_spec["B"]),
        eps=kmm_spec["eps"],
        max_size=len(source_features),
        max_iter=int(kmm_spec["max_iter"]),
    )
    raw_weights = np.stack((hajek_raw, estimate.raw_weight))
    weights = np.stack((hajek_weights, estimate.normalized_weight))
    if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0.0):
        raise ValueError(f"Invalid predicted-theme raw weights for seed {seed}")
    if not np.allclose(weights.mean(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"Predicted-theme weights are not mean one for seed {seed}")
    return PredictedThemeWeights(
        source_fold_id=source_fold_id,
        raw_weights=raw_weights,
        weights=weights,
        kmm_sigma=float(sigma),
        selected_gamma_multiplier=float(estimate.selected_gamma_multiplier),
        selected_gamma=float(estimate.selected_gamma),
    )
