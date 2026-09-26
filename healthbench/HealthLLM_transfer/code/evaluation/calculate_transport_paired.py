"""Estimate paired-score target means from supplied sample arrays."""

from __future__ import annotations
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
from covariateshift import cafe_pairedscore_inference as inference
from covariateshift import paired_fusion as fusion
from covariateshift import cafe_pairedscore as cafe
from covariateshift import RePPI as reppi
from covariateshift.common_mean_regression import RegressorFactory
from covariateshift.kernel_based_methods import fit_kmm_ratio_predictor
from covariateshift.target_label_calibration import estimate_crossfit_aipw, estimate_ppi, fit_target_aipw
from evaluation import calculate_transport as base
from evaluation.calculate_transport import RatioFactory, _seed

__all__ = ["evaluate_sample"]
METHODS = ("labelled_only", "ppi", "reppi", "aipw", "cafe")
Residuals = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SharedNuisance:
    """Store fixed source predictions and out-of-fold KMM ratios shared across label budgets."""

    fhat_target: np.ndarray
    qhat_source: np.ndarray
    kmm_diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PairedTransportEvaluation:
    """Store one label budget and its paired-score estimates."""

    alpha: float
    labeled: np.ndarray
    shared: SharedNuisance
    estimates: Mapping[str, Mapping[str, float]]


def fit_shared_nuisance(
    source_features: np.ndarray,
    source_b: np.ndarray,
    target_features: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> SharedNuisance:
    """Return one source-trained proxy and cross-fitted KMM ratios for all label budgets."""
    methods = {entry["name"] for entry in config["methods"]}
    prediction = np.full(len(target_features), np.nan)
    ratios = np.full(len(source_b), np.nan)
    diagnostics = []
    if methods & {"ppi", "reppi"}:
        model = regressor_factory(_seed(seed, 1))
        model.fit(source_features, source_b)
        prediction = inference._predict(model, target_features)
        del model
    return SharedNuisance(prediction, ratios, tuple(diagnostics))


def evaluate_sample(
    source_features: np.ndarray,
    source_b: np.ndarray,
    target_features: np.ndarray,
    target_b: np.ndarray,
    target_y: np.ndarray,
    target_uniform: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> Iterator[PairedTransportEvaluation]:
    """Yield target means and SEs with paired scores revealed only at sampled labels."""
    shared = fit_shared_nuisance(
        source_features,
        source_b,
        target_features,
        config=config,
        seed=seed,
        regressor_factory=regressor_factory,
        ratio_factory=ratio_factory,
    )
    methods = [entry["name"] for entry in config["methods"]]
    if "cafe" in methods:
        source_q = fusion.fit_source_q(
            source_features, target_features, density_ratio=config["density_ratio"], seed=_seed(seed, 4)
        )
    for alpha, labeled in base.target_label_masks(target_uniform, config["split"]):
        if not labeled.any():
            raise ValueError(f"No target labels sampled at alpha={alpha}")
        indices = np.flatnonzero(labeled)
        observed_b, observed_y = (np.full(len(target_y), np.nan), np.full(len(target_y), np.nan))
        observed_b[labeled], observed_y[labeled] = (target_b[labeled], target_y[labeled])
        if not np.isfinite(observed_b[labeled]).all() or not np.isfinite(observed_y[labeled]).all():
            raise ValueError("Labeled target pairs must contain finite B and Y")
        if "cafe" in methods:
            cafe_result, _ = cafe.fit_cafe_cv(
                source_features,
                source_b,
                target_features,
                labeled,
                observed_b,
                observed_y,
                n_folds=config["split"]["n_folds"],
                n_inner_folds=config["cafe"]["n_inner_folds"],
                seed=_seed(seed, 4),
                density_ratio=config["density_ratio"],
                regressor_factory=regressor_factory,
                source_q=source_q,
                coefficient_spec=config["cafe"],
            )
        estimates = {}
        for method in methods:
            if method == "labelled_only":
                labeled_outcome = observed_y[labeled]
                estimates[method] = {
                    "estimate": float(labeled_outcome.mean()),
                    "standard_error": float(np.std(labeled_outcome, ddof=1) / np.sqrt(len(labeled_outcome)))
                    if len(labeled_outcome) >= 2
                    else np.nan,
                }
            elif method == "ppi":
                estimates[method] = estimate_ppi(shared.fhat_target, indices, observed_y[labeled])
            elif method == "aipw":
                if set(methods) & {"cafe"}:
                    outcome_result = cafe_result
                    estimates[method] = estimate_crossfit_aipw(
                        outcome_result.target_m_prediction,
                        labeled,
                        observed_y,
                        target_fold_ids=outcome_result.target_fold_id,
                        label_probability=float(labeled.mean()),
                        empirical_label_probability=True,
                    )
                else:
                    target_folds = inference.make_target_fold_ids(
                        len(target_y),
                        n_folds=config["split"]["n_folds"],
                        seed=_seed(seed, 4),
                        target_labeled=labeled,
                    )
                    model_seeds = inference.make_cafe_model_seeds(
                        n_folds=config["split"]["n_folds"], seed=_seed(seed, 4)
                    )
                    estimates[method] = fit_target_aipw(
                        target_features,
                        labeled,
                        observed_y,
                        target_fold_ids=target_folds,
                        label_probability=float(labeled.mean()),
                        seed=_seed(seed, 4),
                        model_seeds=model_seeds[1::2],
                        regressor_factory=regressor_factory,
                        empirical_label_probability=True,
                    )
            elif method == "reppi":
                result = reppi.fit_reppi(
                    target_features,
                    shared.fhat_target,
                    indices,
                    observed_y[labeled],
                    model_factory=regressor_factory,
                    seed=_seed(seed, 3),
                )
                estimates[method] = {
                    "estimate": result.estimate,
                    "plugin_mean": result.plugin_mean,
                    "residual_correction": result.residual_correction,
                    "standard_error": result.standard_error,
                    "zero_coefficient_count": int(np.count_nonzero(result.fold_coefficient == 0)),
                    "fold_count": len(result.fold_coefficient),
                }
            elif method == "cafe":
                estimates[method] = {
                    "estimate": cafe_result.estimate,
                    "standard_error": cafe_result.standard_error,
                }
            else:
                raise ValueError(f"Unknown paired-score estimator: {method}")
        yield PairedTransportEvaluation(alpha, labeled, shared, estimates)
