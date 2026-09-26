"""Estimate common-outcome target means from supplied sample arrays."""

from __future__ import annotations
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from covariateshift import cafe_transport_inference as inference
from covariateshift import transport_fusion as fusion
from covariateshift import cafe_transport as cafe
from covariateshift import common_mean_regression as common
from covariateshift import pooled_label_DR as pooled_dr
from covariateshift.common_mean_regression import RegressorFactory, fit_common_mean_crossfit
from covariateshift.common_mean_regression import predict_mean as _predict
from covariateshift.estimators import median_sigma
from covariateshift.kernel_based_methods import KMMRatioPredictor, fit_kmm_ratio_predictor
from covariateshift.pooled_label_DR import estimate_pooled_label_dr
from covariateshift.target_label_calibration import estimate_crossfit_aipw, estimate_ppi, fit_target_aipw
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold

__all__ = ["evaluate_sample"]
METHODS = ("labelled_only", "source_reweighting", "ppi", "aipw", "pooled_aipw", "pooled_dr", "cafe")
RatioFactory = Callable[..., KMMRatioPredictor]


@dataclass(frozen=True)
class SharedNuisance:
    """Store label-budget-independent predictions, ratios, and folds."""

    source_fold_ids: np.ndarray
    target_fold_ids: np.ndarray
    fhat_target: np.ndarray
    qhat_source: np.ndarray
    qhat_target: np.ndarray
    kmm_diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PooledNuisance:
    """Store shared source-plus-label predictions and CAFE coefficient evaluations."""

    mhat_source_oof: np.ndarray
    mhat_target: np.ndarray
    source_w: np.ndarray
    target_a: np.ndarray
    tuning: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TransportEvaluation:
    """Store the configured label budget, estimates, and aligned nuisance arrays."""

    alpha: float
    labeled: np.ndarray
    shared: SharedNuisance
    pooled: PooledNuisance
    estimates: Mapping[str, Mapping[str, float]]


def _seed(seed: int, *coordinates: int) -> int:
    return int(np.random.SeedSequence([seed, *coordinates]).generate_state(1)[0])


def _rbf(gamma: float) -> inference.Kernel:
    return lambda first, second: np.exp(-gamma * cdist(first, second, "sqeuclidean"))


def _training_kernel(
    source: np.ndarray, target: np.ndarray, *, max_points: int, seed: int
) -> tuple[inference.Kernel, float]:
    sigma = median_sigma(np.vstack((source, target)), max_points=max_points, seed=seed)
    return (_rbf(1 / (2 * sigma**2)), sigma)


def fit_shared_nuisance(
    source_features: np.ndarray,
    source_y: np.ndarray,
    target_features: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    regressor_factory: inference.TabPFNFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> SharedNuisance:
    """Return source-only predictions and density ratios shared across label budgets."""
    methods = {entry["name"] for entry in config["methods"]}
    source_folds, target_folds = inference.make_cafe_fold_ids(
        len(source_y), len(target_features), n_folds=config["split"]["n_folds"], seed=_seed(seed, 0)
    )
    fhat = np.full(len(target_features), np.nan)
    q_source, q_target = (np.full(len(source_y), np.nan), np.full(len(target_features), np.nan))
    diagnostics = []
    if methods & {"ppi", "aipw"}:
        model = regressor_factory(_seed(seed, 1))
        model.fit(source_features, source_y)
        fhat = _predict(model, target_features)
        del model
    for fold in range(config["split"]["n_folds"]):
        source_eval, target_eval = (source_folds == fold, target_folds == fold)
        source_train, target_train = (source_features[~source_eval], target_features[~target_eval])
        if methods & {"source_reweighting"}:
            density = config["density_ratio"]
            density_seed = _seed(seed, 2, fold)
            sigma = median_sigma(
                np.vstack((source_train, target_train)),
                max_points=density["median_max_points"],
                seed=density_seed,
            )
            predictor = ratio_factory(
                density["method"], source_train, target_train, sigma=sigma, seed=density_seed
            )
            q_source[source_eval] = predictor.predict(source_features[source_eval])
            q_target[target_eval] = predictor.predict(target_features[target_eval])
            diagnostics.append(
                {
                    "fold": fold,
                    "seed": density_seed,
                    "sigma": sigma,
                    "gamma": predictor.gamma,
                    "normalizer": predictor.normalizer,
                    "gamma_multiplier": predictor.weight_estimate.selected_gamma_multiplier,
                    "selection_score": predictor.weight_estimate.selection_score,
                }
            )
    return SharedNuisance(source_folds, target_folds, fhat, q_source, q_target, tuple(diagnostics))


def target_label_masks(
    target_uniform: np.ndarray, split: Mapping[str, Any]
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield nested target label masks for exact counts or Bernoulli probabilities."""
    if "label_counts" in split:
        order = np.argsort(target_uniform, kind="stable")
        for count in split["label_counts"]:
            if type(count) is not int or not 2 <= count < len(target_uniform):
                raise ValueError("Each label count must be an integer from 2 to n_target - 1")
            labeled = np.zeros(len(target_uniform), dtype=bool)
            labeled[order[:count]] = True
            yield (count / len(target_uniform), labeled)
        return
    for alpha in split["label_probabilities"]:
        yield (alpha, target_uniform < alpha)


def evaluate_sample(
    source_features: np.ndarray,
    source_y: np.ndarray,
    target_features: np.ndarray,
    target_y: np.ndarray,
    target_uniform: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> Iterator[TransportEvaluation]:
    """Yield configured common-outcome estimates with shared pooled outcome predictions."""
    methods = [entry["name"] for entry in config["methods"]]
    if set(methods) & {"cafe"}:
        source_q = fusion.fit_source_q(
            source_features, target_features, density_ratio=config["density_ratio"], seed=_seed(seed, 3)
        )
    shared_methods = [entry for entry in config["methods"] if entry["name"] in {"ppi", "source_reweighting"}]
    if shared_methods:
        shared = fit_shared_nuisance(
            source_features,
            source_y,
            target_features,
            config={**config, "methods": shared_methods},
            seed=seed,
            regressor_factory=regressor_factory,
            ratio_factory=ratio_factory,
        )
    else:
        shared = SharedNuisance(
            np.full(len(source_y), -1, dtype=int),
            np.full(len(target_y), -1, dtype=int),
            np.full(len(target_y), np.nan),
            np.full(len(source_y), np.nan),
            np.full(len(target_y), np.nan),
            (),
        )
    for alpha, labeled in target_label_masks(target_uniform, config["split"]):
        if not labeled.any():
            raise ValueError(f"No target labels sampled at alpha={alpha}")
        indices = np.flatnonzero(labeled)
        observed_y = np.full(len(target_y), np.nan)
        observed_y[labeled] = target_y[labeled]
        alpha_hat = float(labeled.mean())
        pooled = PooledNuisance(
            np.full(len(source_y), np.nan),
            np.full(len(target_y), np.nan),
            np.full(len(source_y), np.nan),
            np.full(len(target_y), np.nan),
            (),
        )
        if set(methods) & {"aipw", "pooled_aipw", "pooled_dr", "cafe"}:
            splitter = StratifiedKFold(
                n_splits=config["split"]["n_folds"], shuffle=True, random_state=_seed(seed, 6, 0)
            )
            target_folds = np.empty(len(target_y), dtype=np.int64)
            for fold, (_, heldout) in enumerate(splitter.split(target_features, labeled)):
                target_folds[heldout] = fold
        if set(methods) & {"pooled_aipw", "pooled_dr", "cafe"}:
            source_folds, _ = common.make_fold_ids(
                len(source_y), len(target_y), n_folds=config["split"]["n_folds"], seed=_seed(seed, 0)
            )
            if not set(methods) & {"cafe"}:
                regression = fit_common_mean_crossfit(
                    source_features,
                    source_y,
                    target_features,
                    labeled_indices=indices,
                    labeled_outcome=observed_y[labeled],
                    source_fold_ids=source_folds,
                    target_fold_ids=target_folds,
                    regressor_factory=regressor_factory,
                    seed=_seed(seed, 3),
                )
                pooled = PooledNuisance(
                    regression.source_prediction,
                    regression.target_prediction,
                    pooled.source_w,
                    pooled.target_a,
                    (),
                )
            cafe_order = [name for name in ("cafe",) if name in methods]
            cafe_results = {}
            for fit_order, method in enumerate(cafe_order, start=1):
                arguments = dict(
                    n_folds=config["split"]["n_folds"],
                    n_inner_folds=config[method]["n_inner_folds"],
                    seed=_seed(seed, 3),
                    density_ratio=config["density_ratio"],
                    regressor_factory=regressor_factory,
                    source_q=source_q,
                    source_fold_id=source_folds,
                    target_fold_id=target_folds,
                )
                fitted, tuning = cafe.fit_cafe_cv(
                    source_features,
                    source_y,
                    target_features,
                    labeled,
                    observed_y,
                    coefficient_spec=config["cafe"],
                    **arguments,
                )
                cafe_results[method] = fitted
                pooled = PooledNuisance(
                    fitted.source_m_prediction,
                    fitted.target_m_prediction,
                    fitted.source_w,
                    fitted.target_a,
                    (*pooled.tuning, *tuning),
                )
        if "pooled_dr" in methods:
            pooled_weights = pooled_dr.fit_pooled_label_weights(
                source_features,
                target_features,
                indices,
                density_config=config["density_ratio"],
                seed=_seed(seed, 7),
            )
        estimates = {}
        for method in methods:
            if method == "labelled_only":
                labeled_outcome = observed_y[labeled]
                estimates[method] = {
                    "estimate": float(labeled_outcome.mean()),
                    "standard_error": float(np.std(labeled_outcome, ddof=1) / np.sqrt(len(labeled_outcome)))
                    if len(labeled_outcome) > 1
                    else np.nan,
                }
            elif method == "source_reweighting":
                estimates[method] = {
                    "estimate": float(np.average(source_y, weights=shared.qhat_source)),
                    "standard_error": np.nan,
                }
            elif method == "ppi":
                estimates[method] = estimate_ppi(shared.fhat_target, indices, observed_y[labeled])
            elif method == "aipw":
                estimates[method] = fit_target_aipw(
                    target_features,
                    labeled,
                    observed_y,
                    target_fold_ids=target_folds,
                    label_probability=alpha_hat,
                    seed=_seed(seed, 6, 1),
                    regressor_factory=regressor_factory,
                    empirical_label_probability=True,
                )
            elif method == "pooled_aipw":
                estimates[method] = estimate_crossfit_aipw(
                    pooled.mhat_target,
                    labeled,
                    observed_y,
                    target_fold_ids=target_folds,
                    label_probability=alpha_hat,
                    empirical_label_probability=True,
                )
            elif method == "pooled_dr":
                estimates[method] = estimate_pooled_label_dr(
                    source_y,
                    pooled.mhat_source_oof,
                    observed_y[labeled],
                    indices,
                    pooled.mhat_target,
                    pooled_weights,
                )
            elif method in {"cafe"}:
                fitted = cafe_results[method]
                estimates[method] = {"estimate": fitted.estimate, "standard_error": fitted.standard_error}
            else:
                raise ValueError(f"Unknown common-outcome estimator: {method}")
        yield TransportEvaluation(alpha, labeled, shared, pooled, estimates)
