"""Shared source-plus-labelled-target outcome regression and cross-fitting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Regressor(Protocol):
    """Describe the outcome learner's fit and predict interface."""

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> Any:
        """Return the learner fitted to the supplied observations."""
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return one conditional-mean prediction per row."""
        ...


RegressorFactory = Callable[[int], Regressor]


def lasso_regression_config(regression: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Lasso settings recorded in the resolved run configuration."""
    if regression["name"] != "lasso":
        raise ValueError("regression.name must be lasso")
    return {key: regression[key] for key in ("name", "alphas", "n_folds", "max_iter", "tol")}


def make_lasso_regressor_factory(regression: Mapping[str, Any]) -> RegressorFactory:
    """Return fresh Lasso learners with training-only scaling and penalty selection."""
    settings = lasso_regression_config(regression)

    def factory(seed: int) -> GridSearchCV:
        return GridSearchCV(
            make_pipeline(
                StandardScaler(),
                Lasso(max_iter=settings["max_iter"], tol=settings["tol"], random_state=seed),
            ),
            param_grid={"lasso__alpha": settings["alphas"]},
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=settings["n_folds"], shuffle=True, random_state=seed),
            n_jobs=1,
            error_score="raise",
        )

    return factory


def regression_config(regression: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the active learner's settings for resolved run provenance."""
    if regression["name"] == "lasso":
        return lasso_regression_config(regression)
    if regression["name"] == "ridge":
        return {key: regression[key] for key in ("name", "alphas", "n_folds")}
    if regression["name"] == "tabpfn":
        keys = (
            "name",
            "package_version",
            "model_version",
            "n_estimators",
            "device",
            "ignore_pretraining_limits",
            "checkpoint_env",
            "checkpoint_repo",
            "checkpoint_revision",
            "checkpoint_filename",
            "checkpoint_sha256",
        )
        settings = {key: regression[key] for key in keys}
        for key in ("fit_mode", "keep_cache_on_device", "kv_cache_precision"):
            if key in regression:
                settings[key] = regression[key]
        if "checkpoint" in regression:
            settings["checkpoint"] = regression["checkpoint"]
        return settings
    if regression["name"] != "krr":
        raise ValueError("regression.name must be lasso, ridge, krr, or tabpfn")
    if regression["kernel"] != "rbf":
        raise ValueError("KRR regression.kernel must be rbf")
    return {key: regression[key] for key in ("name", "kernel", "alphas", "gammas", "n_folds")}


def make_regressor_factory(regression: Mapping[str, Any]) -> RegressorFactory:
    """Return fresh configured learners with training-only preprocessing and tuning."""
    if regression["name"] == "lasso":
        return make_lasso_regressor_factory(regression)
    settings = regression_config(regression)
    if settings["name"] == "ridge":

        def ridge_factory(seed: int) -> GridSearchCV:
            return GridSearchCV(
                # SVD remains stable when a target training fold has fewer labels than features.
                make_pipeline(StandardScaler(), Ridge(solver="svd")),
                param_grid={"ridge__alpha": settings["alphas"]},
                scoring="neg_mean_squared_error",
                cv=KFold(n_splits=settings["n_folds"], shuffle=True, random_state=seed),
                refit=True,
                n_jobs=1,
                error_score="raise",
            )

        return ridge_factory
    if settings["name"] == "tabpfn":
        from meta_eval.evaluation.source_outcome_model import make_tabpfn_regressor_factory

        if version("tabpfn") != settings["package_version"]:
            raise ValueError(f"TabPFN requires version {settings['package_version']}")
        return make_tabpfn_regressor_factory(settings, Path(settings["checkpoint"]))

    def factory(seed: int) -> GridSearchCV:
        return GridSearchCV(
            make_pipeline(
                StandardScaler(),
                TransformedTargetRegressor(
                    regressor=KernelRidge(kernel=settings["kernel"]),
                    # Center within each CV fit to avoid shrinking the score baseline toward zero.
                    transformer=StandardScaler(with_std=False),
                ),
            ),
            param_grid={
                "transformedtargetregressor__regressor__alpha": settings["alphas"],
                "transformedtargetregressor__regressor__gamma": settings["gammas"],
            },
            scoring="neg_mean_squared_error",
            cv=KFold(n_splits=settings["n_folds"], shuffle=True, random_state=seed),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )

    return factory


@dataclass(frozen=True)
class CommonMeanFold:
    """Expose one fitted learner and its training and held-out row masks."""

    fold_id: int
    model: Regressor
    source_training: np.ndarray
    target_training: np.ndarray
    source_heldout: np.ndarray
    target_heldout: np.ndarray
    source_prediction: np.ndarray
    target_prediction: np.ndarray


@dataclass(frozen=True)
class CommonMeanRegressionFit:
    """Store predictions and supplied fold assignments in original row order."""

    source_prediction: np.ndarray
    target_prediction: np.ndarray
    source_fold_ids: np.ndarray
    target_fold_ids: np.ndarray


def predict_mean(model: Regressor, features: np.ndarray, *, name: str = "prediction") -> np.ndarray:
    """Return one finite scalar prediction per supplied row."""
    prediction = np.asarray(model.predict(features), dtype=np.float64)
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError(f"{name} must contain one finite prediction per row")
    return prediction


def make_fold_ids(
    n_source: int,
    n_target: int,
    *,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent balanced source and target partitions with zero-based IDs."""
    if not isinstance(n_folds, (int, np.integer)) or not 2 <= n_folds <= min(n_source, n_target):
        raise ValueError("n_folds must be an integer between 2 and both sample sizes")
    assignments = []
    for size, group_seed in zip((n_source, n_target), np.random.SeedSequence(seed).spawn(2)):
        permutation = np.random.default_rng(group_seed).permutation(size)
        fold_ids = np.empty(size, dtype=np.int64)
        for fold, indices in enumerate(np.array_split(permutation, n_folds)):
            fold_ids[indices] = fold
        assignments.append(fold_ids)
    return assignments[0], assignments[1]


def fit_common_mean_crossfit(
    source_features: np.ndarray,
    source_outcome: np.ndarray,
    target_features: np.ndarray,
    *,
    labeled_indices: np.ndarray,
    labeled_outcome: np.ndarray,
    source_fold_ids: np.ndarray,
    target_fold_ids: np.ndarray,
    regressor_factory: RegressorFactory,
    seed: int,
    on_fold: Callable[[CommonMeanFold], None] | None = None,
) -> CommonMeanRegressionFit:
    """Return pooled-regression predictions and invoke an optional per-fold callback.

    Target fold -1 requests a fold-average prediction and is allowed only for
    unlabelled rows. The callback runs before the next learner is constructed.
    """
    source = np.asarray(source_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target features must be matrices with matching columns")
    if min(len(source), len(target)) == 0 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Feature matrices must be nonempty and finite")
    source_y = np.asarray(source_outcome, dtype=np.float64)
    labeled_y = np.asarray(labeled_outcome, dtype=np.float64)
    labeled_indices = np.asarray(labeled_indices)
    if source_y.shape != (len(source),) or not np.isfinite(source_y).all():
        raise ValueError("source_outcome must be a finite source-aligned vector")
    if labeled_indices.ndim != 1 or not np.issubdtype(labeled_indices.dtype, np.integer):
        raise ValueError("labeled_indices must be an integer vector")
    if len(np.unique(labeled_indices)) != len(labeled_indices) or np.any(
        (labeled_indices < 0) | (labeled_indices >= len(target))
    ):
        raise ValueError("labeled_indices must be unique target row indices")
    if labeled_y.shape != labeled_indices.shape or not np.isfinite(labeled_y).all():
        raise ValueError("labeled_outcome must be finite and aligned with labeled_indices")
    source_folds, target_folds = np.asarray(source_fold_ids), np.asarray(target_fold_ids)
    for folds, size in ((source_folds, len(source)), (target_folds, len(target))):
        if folds.shape != (size,) or not np.issubdtype(folds.dtype, np.integer):
            raise ValueError("Fold IDs must be integer vectors aligned with their samples")
    n_folds = len(np.unique(source_folds))
    if n_folds < 2 or not np.array_equal(np.unique(source_folds), np.arange(n_folds)):
        raise ValueError("Source fold IDs must span at least two contiguous folds starting at zero")
    if np.any((target_folds < -1) | (target_folds >= n_folds)) or np.any(target_folds[labeled_indices] < 0):
        raise ValueError("Target fold IDs must match source folds; labelled rows require a held-out fold")

    labeled = np.zeros(len(target), dtype=bool)
    labeled[labeled_indices] = True
    target_y = np.full(len(target), np.nan)
    target_y[labeled_indices] = labeled_y
    source_prediction = np.empty(len(source))
    target_prediction = np.zeros(len(target))
    averaged = target_folds == -1
    model_seeds = np.random.SeedSequence(seed).spawn(3)[2].spawn(n_folds)
    for fold in range(n_folds):
        source_eval, target_eval = source_folds == fold, target_folds == fold
        source_train, target_train = ~source_eval, ~target_eval
        labeled_train = labeled & target_train
        model = regressor_factory(int(model_seeds[fold].generate_state(1)[0]))
        model.fit(
            np.vstack((source[source_train], target[labeled_train])),
            np.r_[source_y[source_train], target_y[labeled_train]],
        )
        source_prediction[source_eval] = predict_mean(model, source[source_eval], name="source_prediction")
        if target_eval.any():
            target_prediction[target_eval] = predict_mean(
                model, target[target_eval], name="target_prediction"
            )
        if averaged.any():
            target_prediction[averaged] += (
                predict_mean(model, target[averaged], name="target_prediction") / n_folds
            )
        if on_fold is not None:
            on_fold(
                CommonMeanFold(
                    fold,
                    model,
                    source_train,
                    target_train,
                    source_eval,
                    target_eval,
                    source_prediction[source_eval],
                    target_prediction[target_eval],
                )
            )
        del model
    return CommonMeanRegressionFit(
        source_prediction, target_prediction, source_folds.copy(), target_folds.copy()
    )
