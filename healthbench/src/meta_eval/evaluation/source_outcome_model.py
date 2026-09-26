"""Source-trained outcome regression for target-mean calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from ..provenance import sha256_file


PC_FEATURE_SET = "pc50_prompt_pc50_rubric_100d"
FULL_FEATURE_SET = "full_prompt_rubric_1536d"
OUTCOME_FEATURE_SETS = (PC_FEATURE_SET, FULL_FEATURE_SET)


class Regressor(Protocol):
    """Describe the regression interface required by calibration models."""

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> Any: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


RegressorFactory = Callable[[int], Regressor]


def make_tabpfn_regressor_factory(method: Mapping[str, Any], checkpoint: Path) -> RegressorFactory:
    """Return the configured local TabPFN regressor factory."""

    if sha256_file(checkpoint) != str(method["checkpoint_sha256"]):
        raise ValueError("TabPFN regressor checkpoint hash mismatch.")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    model_version = getattr(ModelVersion, str(method["model_version"]).upper())
    inference_options = {
        name: method[name]
        for name in ("fit_mode", "keep_cache_on_device", "kv_cache_precision")
        if name in method
    }

    def build(seed: int) -> Regressor:
        return TabPFNRegressor.create_default_for_version(
            model_version,
            model_path=str(checkpoint),
            device=str(method["device"]),
            n_estimators=int(method["n_estimators"]),
            ignore_pretraining_limits=bool(method["ignore_pretraining_limits"]),
            random_state=int(seed),
            **inference_options,
        )

    return build


@dataclass(frozen=True)
class FeatureRepresentations:
    """Store aligned PC100 and Full1536 source/target features."""

    source_pc: np.ndarray
    target_pc: np.ndarray
    source_full: np.ndarray
    target_full: np.ndarray

    def source(self, feature_set: str) -> np.ndarray:
        """Return source features for one configured representation."""

        return {
            PC_FEATURE_SET: self.source_pc,
            FULL_FEATURE_SET: self.source_full,
        }[feature_set]

    def target(self, feature_set: str) -> np.ndarray:
        """Return target features for one configured representation."""

        return {
            PC_FEATURE_SET: self.target_pc,
            FULL_FEATURE_SET: self.target_full,
        }[feature_set]


@dataclass(frozen=True)
class SourceOutcomeFit:
    """Store source OOF and full-source target predictions."""

    source_oof: np.ndarray
    target_full_source_fit: np.ndarray
    source_fold_ids: np.ndarray


def build_feature_representations(
    source_prompt: np.ndarray,
    source_rubric: np.ndarray,
    target_prompt: np.ndarray,
    target_rubric: np.ndarray,
    *,
    n_components: int,
) -> FeatureRepresentations:
    """Build pooled-PC and direct-concatenation embedding features."""

    source_prompt = np.asarray(source_prompt, dtype=np.float32)
    source_rubric = np.asarray(source_rubric, dtype=np.float32)
    target_prompt = np.asarray(target_prompt, dtype=np.float32)
    target_rubric = np.asarray(target_rubric, dtype=np.float32)
    assert source_prompt.shape[1:] == source_rubric.shape[1:] == (768,)
    assert target_prompt.shape[1:] == target_rubric.shape[1:] == (768,)

    # Each PCA basis is fit once on the model-specific complete S union T population.
    prompt_pca = PCA(n_components=n_components, svd_solver="full")
    rubric_pca = PCA(n_components=n_components, svd_solver="full")
    pooled_prompt = np.vstack((source_prompt, target_prompt))
    pooled_rubric = np.vstack((source_rubric, target_rubric))
    prompt_pca.fit(pooled_prompt)
    rubric_pca.fit(pooled_rubric)

    source_pc = np.column_stack(
        (prompt_pca.transform(source_prompt), rubric_pca.transform(source_rubric))
    ).astype(np.float32)
    target_pc = np.column_stack(
        (prompt_pca.transform(target_prompt), rubric_pca.transform(target_rubric))
    ).astype(np.float32)
    source_full = np.column_stack((source_prompt, source_rubric)).astype(np.float32)
    target_full = np.column_stack((target_prompt, target_rubric)).astype(np.float32)

    assert source_pc.shape[1] == target_pc.shape[1] == 2 * n_components
    assert source_full.shape[1] == target_full.shape[1] == 1536
    return FeatureRepresentations(source_pc, target_pc, source_full, target_full)


def make_fold_ids(n_rows: int, n_splits: int, seed: int) -> np.ndarray:
    """Return one deterministic shuffled fold assignment."""

    fold_ids = np.empty(n_rows, dtype=np.int8)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, heldout) in enumerate(splitter.split(np.arange(n_rows))):
        fold_ids[heldout] = fold
    return fold_ids


def make_two_fold_ids(n_rows: int, seed: int) -> np.ndarray:
    """Return one deterministic shuffled two-fold assignment."""

    return make_fold_ids(n_rows, 2, seed)


def fit_source_outcome_model(
    source_features: np.ndarray,
    source_outcome: np.ndarray,
    target_features: np.ndarray,
    *,
    fold_ids: np.ndarray,
    model_factory: RegressorFactory,
    model_seed: int,
) -> SourceOutcomeFit:
    """Fit the source model and return two-fold OOF plus target predictions."""

    source_features = np.asarray(source_features, dtype=np.float32)
    target_features = np.asarray(target_features, dtype=np.float32)
    source_outcome = np.asarray(source_outcome, dtype=np.float32)
    fold_ids = np.asarray(fold_ids, dtype=np.int8)
    assert source_outcome.shape == fold_ids.shape == (len(source_features),)

    source_oof = np.empty(len(source_outcome), dtype=float)
    for fold in (0, 1):
        heldout = fold_ids == fold
        model = model_factory(model_seed + fold)
        model.fit(source_features[~heldout], source_outcome[~heldout])
        source_oof[heldout] = model.predict(source_features[heldout])

    full_model = model_factory(model_seed + 2)
    full_model.fit(source_features, source_outcome)
    target_prediction = np.asarray(full_model.predict(target_features), dtype=float)
    return SourceOutcomeFit(source_oof, target_prediction, fold_ids.copy())
