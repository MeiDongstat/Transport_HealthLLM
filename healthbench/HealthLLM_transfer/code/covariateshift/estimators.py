"""Shared density-ratio estimators for embedding covariate shift."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

try:
    from adapt.instance_based import KLIEP, KMM, RULSIF, ULSIF
    from adapt.metrics import make_uda_scorer, neg_j_score
except ImportError:
    KLIEP = KMM = RULSIF = ULSIF = None
    make_uda_scorer = neg_j_score = None


@dataclass(frozen=True)
class ImportanceWeightEstimate:
    """Represent an estimator's direct output and mean-one weight."""

    raw_weight: np.ndarray
    normalized_weight: np.ndarray
    selected_lambda: float | None = None
    selected_gamma_multiplier: float | None = None
    selected_gamma: float | None = None
    selection_score: float | None = None


def _require_adapt():
    if KMM is None:
        raise ImportError(
            "ADAPT kernel estimators require `adapt`. Install them with: pip install 'healthbench-covshift[analysis]' adapt"
        )


def make_importance_weight_estimate(raw_weight, method_name):
    """Return direct importance weights and their mean-one normalization."""
    raw_weight = np.asarray(raw_weight, dtype=float).reshape(-1)
    if not np.all(np.isfinite(raw_weight)):
        raise ValueError(f"{method_name} raw_weight contains non-finite values.")
    if np.any(raw_weight < 0):
        raise ValueError(f"{method_name} raw_weight contains negative values.")
    if np.all(raw_weight == 0):
        raise ValueError(f"{method_name} raw_weight contains no positive values.")
    mean_weight = np.mean(raw_weight)
    return ImportanceWeightEstimate(raw_weight=raw_weight, normalized_weight=raw_weight / mean_weight)


def median_sigma(Z, max_points=2000, seed=0):
    Z = np.asarray(Z, dtype=float)
    rng = np.random.default_rng(seed)
    if len(Z) > max_points:
        Z = Z[rng.choice(len(Z), size=max_points, replace=False)]
    distances = pairwise_distances(Z)
    sigma = float(np.median(distances[np.triu_indices_from(distances, k=1)]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Median-distance sigma must be positive and finite; got {sigma}.")
    return sigma


def make_domain_fold_ids(n_source, n_target, *, n_splits, random_state):
    """Create deterministic stratified OOF folds for pooled source/target rows."""
    y = np.concatenate([np.zeros(n_source), np.ones(n_target)]).astype(int)
    fold_ids = np.empty(len(y), dtype=np.int16)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (_, heldout) in enumerate(splitter.split(np.zeros((len(y), 1)), y)):
        fold_ids[heldout] = fold_id
    return (y, fold_ids)


def make_tabpfn_classifier_factory(method, checkpoint_path):
    """Build a local, explicitly versioned TabPFN classifier factory."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model_version = getattr(ModelVersion, str(method["model_version"]).upper())
    precision = method.get("inference_precision", "auto")
    if precision == "float32":
        import torch

        precision = torch.float32

    def build(random_state):
        return TabPFNClassifier.create_default_for_version(
            model_version,
            model_path=str(checkpoint_path),
            device=str(method["device"]),
            n_estimators=int(method["n_estimators"]),
            auto_scale_n_estimators=bool(method.get("auto_scale_n_estimators", True)),
            softmax_temperature=float(method.get("softmax_temperature", 0.9)),
            balance_probabilities=False,
            ignore_pretraining_limits=bool(method.get("ignore_pretraining_limits", False)),
            inference_precision=precision,
            fit_mode=str(method.get("fit_mode", "fit_preprocessors")),
            random_state=int(random_state),
            show_progress_bar=False,
        )

    return build


def fit_tabpfn_domain_fold(X_all, y, fold_ids, fold_id, *, classifier_factory, random_state):
    """Fit one TabPFN outer fold and return held-out target-domain probabilities."""
    X_all = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    fold_ids = np.asarray(fold_ids, dtype=int)
    heldout = fold_ids == int(fold_id)
    train = ~heldout
    if not heldout.any() or not train.any():
        raise ValueError(f"Invalid TabPFN domain fold: {fold_id}.")
    classifier = classifier_factory(int(random_state) + int(fold_id))
    classifier.fit(X_all[train], y[train])
    probabilities = np.asarray(classifier.predict_proba(X_all[heldout]), dtype=float)
    classes = np.asarray(classifier.classes_)
    target_columns = np.flatnonzero(classes == 1)
    if target_columns.size != 1:
        raise ValueError(f"TabPFN classifier classes do not contain domain label 1: {classes}.")
    output = probabilities[:, int(target_columns[0])]
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ValueError("TabPFN domain probabilities are not finite probabilities.")
    return output


def tabpfn_classifier_density_ratio(source_probabilities, *, n_source, n_target, probability_clip=None):
    """Return TabPFN source density ratios and their mean-one weights."""
    probabilities = np.asarray(source_probabilities, dtype=float)
    if probabilities.shape != (n_source,):
        raise ValueError(
            f"Source domain probabilities have shape {probabilities.shape}; expected {(n_source,)}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Source domain probabilities contain non-finite values.")
    if probability_clip is None:
        if np.any((probabilities < 0.0) | (probabilities >= 1.0)):
            raise ValueError(
                "Raw TabPFN source probabilities must satisfy 0 <= p < 1 for finite nonnegative density ratios."
            )
        used = probabilities
    else:
        epsilon = float(probability_clip)
        if not 0.0 < epsilon < 0.5:
            raise ValueError("Probability protection epsilon must lie in (0, 0.5).")
        used = np.clip(probabilities, epsilon, 1.0 - epsilon)
    density_ratio = used / (1.0 - used) * (n_source / n_target)
    return make_importance_weight_estimate(density_ratio, "tabpfn_classifier")


def preprocess_for_kernel_weighting(X_source, X_target):
    """Return source and target precomputed features without rescaling."""
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    return (
        np.ascontiguousarray(X_source, dtype=np.float64),
        np.ascontiguousarray(X_target, dtype=np.float64),
    )


def adapt_importance_weight_estimate(model, X_source, X_target, method_name):
    """Return an ADAPT model's direct and mean-one importance weights."""
    model.fit_weights(X_source, X_target)
    raw_weight = np.asarray(model.weights_, dtype=float).reshape(-1)
    if raw_weight.shape[0] != X_source.shape[0]:
        raise ValueError(
            f"{method_name} produced {raw_weight.shape[0]} weights for {X_source.shape[0]} source rows."
        )
    return make_importance_weight_estimate(raw_weight, method_name)


def _build_adapt_model(
    method,
    gamma,
    random_state,
    n_centers,
    lam,
    rulsif_alpha,
    kmm_max_size,
    *,
    kliep_cv=5,
    kliep_algo="FW",
    kliep_max_iter=2000,
):
    """Return the requested ADAPT model and its display label."""
    _require_adapt()
    common = {"kernel": "rbf", "gamma": gamma, "verbose": 0, "random_state": random_state}
    if method == "ulsif":
        return (ULSIF(lambdas=lam, max_centers=n_centers, **common), "uLSIF")
    if method == "rulsif":
        return (RULSIF(alpha=rulsif_alpha, lambdas=lam, max_centers=n_centers, **common), "RuLSIF")
    if method == "kliep":
        return (
            KLIEP(max_centers=n_centers, cv=kliep_cv, algo=kliep_algo, max_iter=kliep_max_iter, **common),
            "KLIEP",
        )
    if method == "kmm":
        return (KMM(B=1000, eps=None, max_size=kmm_max_size, max_iter=100, **common), "KMM")
    raise ValueError(f"Unknown kernel method: {method!r}")


def _with_adapt_selection(estimate, model, method, gamma_baseline):
    """Return a weight estimate with ADAPT's selected kernel parameters."""
    selected_lambda = None
    if method in {"ulsif", "rulsif"}:
        selected_lambda = float(model.best_params_["lamb"])
        selected_gamma = float(model.best_params_["k"]["gamma"])
    elif method == "kliep":
        selected_gamma = float(model.best_params_["gamma"])
    else:
        selected_gamma = float(model.gamma)
    selection_score = None
    if method in {"ulsif", "rulsif", "kliep"} and model.j_scores_:
        selection_score = float(model.j_scores_[str(model.best_params_)])
    return ImportanceWeightEstimate(
        raw_weight=estimate.raw_weight,
        normalized_weight=estimate.normalized_weight,
        selected_lambda=selected_lambda,
        selected_gamma_multiplier=selected_gamma / gamma_baseline,
        selected_gamma=selected_gamma,
        selection_score=selection_score,
    )


def fit_kernel_density_ratio(
    method,
    X_source,
    X_target,
    *,
    random_state,
    n_centers=500,
    sigma=None,
    lam=0.001,
    rulsif_alpha=0.1,
    gamma_multipliers=None,
    kliep_cv=5,
    kliep_algo="FW",
    kliep_max_iter=2000,
    kmm_max_size=None,
):
    """Return density ratios and mean-one weights from one kernel method."""
    Xs, Xt = preprocess_for_kernel_weighting(X_source, X_target)
    if sigma is None:
        sigma = median_sigma(np.vstack([Xs, Xt]), seed=random_state)
    gamma_baseline = 1.0 / (2.0 * sigma**2)
    gamma = gamma_baseline
    if gamma_multipliers is not None:
        gamma = [gamma_baseline * float(multiplier) for multiplier in gamma_multipliers]
    if kmm_max_size is None:
        kmm_max_size = len(Xs)
    model, label = _build_adapt_model(
        method,
        gamma,
        random_state,
        n_centers,
        lam,
        rulsif_alpha,
        kmm_max_size=kmm_max_size,
        kliep_cv=kliep_cv,
        kliep_algo=kliep_algo,
        kliep_max_iter=kliep_max_iter,
    )
    estimate = adapt_importance_weight_estimate(model, Xs, Xt, label)
    return _with_adapt_selection(estimate, model, method, gamma_baseline)


def fit_kmm_density_ratio_cv(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    random_state: int,
    sigma: float,
    gamma_multipliers: Sequence[float],
    cv: int = 5,
    B: float = 1000,
    eps: float | None = None,
    max_size: int | None = None,
    max_iter: int = 100,
    scorer_bandwidth: str = "median_gamma",
) -> ImportanceWeightEstimate:
    """Return KMM weights after unsupervised cross-validation of gamma."""
    _require_adapt()
    Xs, Xt = preprocess_for_kernel_weighting(X_source, X_target)
    gamma_baseline = 1.0 / (2.0 * sigma**2)
    gamma_candidates = [gamma_baseline * float(multiplier) for multiplier in gamma_multipliers]
    if max_size is None:
        max_size = len(Xs)
    search_model = KMM(
        estimator=Ridge(0.1),
        Xt=Xt,
        kernel="rbf",
        gamma=gamma_candidates[0],
        B=B,
        eps=eps,
        max_size=max_size,
        max_iter=max_iter,
        verbose=0,
        random_state=random_state,
    )
    scorer_sigma = {"median_gamma": gamma_baseline, "target_mean_distance": None}[scorer_bandwidth]
    scorer = make_uda_scorer(neg_j_score, Xs, Xt, sigma=scorer_sigma)
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        search_model,
        {"gamma": gamma_candidates},
        scoring=scorer,
        return_train_score=True,
        cv=cv_splitter,
        refit=False,
    )
    search.fit(Xs, np.zeros(len(Xs), dtype=float))
    mean_scores = np.asarray(search.cv_results_["mean_train_score"], dtype=float)
    if not np.all(np.isfinite(mean_scores)):
        raise ValueError("KMM unsupervised CV produced non-finite scores.")
    selected_index = int(np.argmax(mean_scores))
    selected_gamma = gamma_candidates[selected_index]
    model = KMM(
        kernel="rbf",
        gamma=selected_gamma,
        B=B,
        eps=eps,
        max_size=max_size,
        max_iter=max_iter,
        verbose=0,
        random_state=random_state,
    )
    estimate = adapt_importance_weight_estimate(model, Xs, Xt, "KMM")
    return ImportanceWeightEstimate(
        raw_weight=estimate.raw_weight,
        normalized_weight=estimate.normalized_weight,
        selected_gamma_multiplier=selected_gamma / gamma_baseline,
        selected_gamma=selected_gamma,
        selection_score=float(mean_scores[selected_index]),
    )
