"""Known-selection oracle weights for stratified target sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def theme_language_strata(
    themes: Sequence[object],
    languages: Sequence[object],
) -> np.ndarray:
    """Return joint theme and English/non-English stratum labels."""
    language_group = np.where(np.asarray(languages, dtype=str) == "en", "en", "non-en")
    return np.char.add(np.char.add(np.asarray(themes, dtype=str), "|"), language_group)


def target_fractions_from_splits(
    strata: Sequence[object],
    target_masks: np.ndarray,
) -> dict[str, float]:
    """Return actual target fractions for fixed-quota repeated splits."""
    strata = np.asarray(strata, dtype=str)
    fractions = {}
    for stratum in np.unique(strata):
        members = strata == stratum
        target_counts = target_masks[:, members].sum(axis=1)
        if not np.all(target_counts == target_counts[0]):
            raise ValueError(f"Target quotas vary across splits for {stratum}")
        fractions[str(stratum)] = float(target_counts[0] / members.sum())
    return fractions


def selection_design_raw_weights(
    source_strata: Sequence[object],
    target_fraction_by_stratum: Mapping[str, float],
) -> np.ndarray:
    """Return source selection odds implied by target fractions."""

    fractions = {str(stratum): float(fraction) for stratum, fraction in target_fraction_by_stratum.items()}
    invalid = {stratum: fraction for stratum, fraction in fractions.items() if not 0.0 <= fraction < 1.0}
    if invalid:
        raise ValueError(f"Invalid design target fractions: {invalid}.")

    strata = np.asarray(source_strata, dtype=str)
    uncovered = set(strata) - set(fractions)
    if uncovered:
        raise ValueError(f"Design probabilities do not cover source strata: {uncovered}.")

    target_odds = {stratum: fraction / (1.0 - fraction) for stratum, fraction in fractions.items()}
    raw_weights = np.asarray(
        [target_odds[stratum] for stratum in strata],
        dtype=np.float64,
    )
    if not np.isfinite(raw_weights).all() or raw_weights.mean() <= 0.0:
        raise ValueError("Design probabilities provide no positive source weight.")
    return raw_weights


def selection_design_weights(
    source_strata: Sequence[object],
    target_fraction_by_stratum: Mapping[str, float],
) -> np.ndarray:
    """Return mean-one source weights implied by target selection fractions."""
    raw_weights = selection_design_raw_weights(
        source_strata,
        target_fraction_by_stratum,
    )
    return raw_weights / raw_weights.mean()
