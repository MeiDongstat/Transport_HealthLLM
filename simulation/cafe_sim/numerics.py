"""Numerical routines retained from the experiment implementation.

Internal B denotes the manuscript auxiliary Y; internal Y denotes target tilde Y.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = [0.0001,0.001,0.01,0.1,1.,10.,100.,1000.,10000.]
CORRELATION_TOLERANCE=1e-4
PREFERRED_TOLERANCE=1e-6
CALIBRATION_PROTOCOL="fixed_sample_pearson_common_random_numbers_v1"

def seed(*coordinates):
    return int(np.random.SeedSequence(list(coordinates)).generate_state(1)[0])

def _normal_scores(y):
    """Midranks retain ties and have finite normal scores even at 0 and 1."""
    y = np.asarray(y, dtype=np.float64)
    return ndtri((rankdata(y, method="average") - 0.5) / len(y))

def _ridge_grid(X, targets, alphas):
    """Fit all ridge penalties from one eigendecomposition, independently by Y.

    This is the usual squared-error Ridge objective (no division by n) with an
    unpenalized intercept and training-only StandardScaler. Reusing the matrix
    factorization avoids repeating a large SVD for every grid point and judge.
    """
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    center_x = scaled.mean(axis=0)
    center_y = targets.mean(axis=0)
    centered_x = scaled - center_x
    eigenvalues, vectors = np.linalg.eigh(centered_x.T @ centered_x)
    eigenvalues = np.maximum(eigenvalues, 0)
    projected = vectors.T @ (centered_x.T @ (targets - center_y))
    coefficients = np.array([vectors @ (projected / (eigenvalues[:, None] + a)) for a in alphas])
    intercepts = center_y[None, :] - np.einsum("p,apc->ac", center_x, coefficients)
    return scaler, coefficients, intercepts

def _choose_alphas(X, targets, alphas, splits):
    errors = np.zeros((len(alphas), targets.shape[1]))
    for train, test in splits:
        scaler, coefficients, intercepts = _ridge_grid(X[train], targets[train], alphas)
        predicted = np.einsum("np,apc->anc", scaler.transform(X[test]), coefficients) + intercepts[:, None, :]
        errors += np.mean((predicted - targets[test][None, :, :]) ** 2, axis=1)
    # Deterministic tie handling follows the provided alpha grid order.
    return np.argmin(errors, axis=0)

def _fit_latent(X, targets, alphas, outer_folds, inner_folds, fit_seed):
    n, channels = targets.shape
    if not 2 <= outer_folds <= n or not 2 <= inner_folds <= n - int(np.ceil(n / outer_folds)):
        raise ValueError("Insufficient samples for requested nested CV folds")
    outer = list(KFold(outer_folds, shuffle=True, random_state=fit_seed).split(X))
    oof = np.empty_like(targets)
    fold_id = np.empty(n, dtype=np.int64)
    selected = []
    for fold, (train, test) in enumerate(outer):
        inner = list(KFold(inner_folds, shuffle=True, random_state=seed(fit_seed, fold + 1)).split(X[train]))
        indices = _choose_alphas(X[train], targets[train], alphas, inner)
        scaler, coefficients, intercepts = _ridge_grid(X[train], targets[train], alphas)
        for channel, alpha_index in enumerate(indices):
            oof[test, channel] = scaler.transform(X[test]) @ coefficients[alpha_index, :, channel] + intercepts[alpha_index, channel]
        selected.append([float(alphas[index]) for index in indices])
        fold_id[test] = fold
    full_splits = list(KFold(inner_folds, shuffle=True, random_state=seed(fit_seed, 100)).split(X))
    indices = _choose_alphas(X, targets, alphas, full_splits)
    scaler, coefficients, intercepts = _ridge_grid(X, targets, alphas)
    coef = np.column_stack([coefficients[index, :, channel] for channel, index in enumerate(indices)])
    intercept = np.array([intercepts[index, channel] for channel, index in enumerate(indices)])
    means = scaler.transform(X) @ coef + intercept
    return {"means": means, "oof": oof, "fold_ids": fold_id,
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "coefficients": coef,
            "intercepts": intercept, "outer_selected_alphas": selected,
            "full_selected_alphas": [float(alphas[index]) for index in indices]}

def _mixture_cdf(t, means, sigma):
    return float(np.mean(ndtr((t - means) / sigma)))

def _calibrate(means, sigma, reference):
    """Invert the equal-weight finite-X Gaussian mixture CDF once per score."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("A strictly positive finite latent noise scale is required")
    values, counts = np.unique(reference, return_counts=True)
    if len(values) < 2:
        raise ValueError("Constant reference scores cannot satisfy nondegenerate score variance")
    probabilities = counts.astype(float) / len(reference)
    probabilities_cdf = np.cumsum(probabilities)[:-1]
    lower, upper = float(means.min() - 12 * sigma), float(means.max() + 12 * sigma)
    thresholds = np.empty(len(values) - 1)
    for index, p in enumerate(probabilities_cdf):
        thresholds[index] = brentq(lambda t: _mixture_cdf(t, means, sigma) - p,
                                  lower, upper, xtol=1e-13, rtol=1e-14)
        lower = thresholds[index]
    return values, probabilities, thresholds

def _conditional_moments(means, sigma, values, thresholds):
    """Stable sum of Gaussian interval masses, processed in modest-size blocks."""
    expectation = np.empty_like(means)
    moment2 = np.empty_like(means)
    zeros = np.empty_like(means)
    ones = np.empty_like(means)
    for start in range(0, len(means), 128):
        end = min(start + 128, len(means))
        cdf = ndtr((thresholds[None, :] - means[start:end, None]) / sigma)
        probabilities = np.diff(np.column_stack((np.zeros(end - start), cdf, np.ones(end - start))), axis=1)
        expectation[start:end] = probabilities @ values
        moment2[start:end] = probabilities @ (values ** 2)
        zeros[start:end] = probabilities[:, 0] if values[0] == 0 else 0
        ones[start:end] = probabilities[:, -1] if values[-1] == 1 else 0
    variance = np.maximum(moment2 - expectation ** 2, 0)
    return {"mean": expectation, "moment2": moment2, "variance": variance,
            "zero_probability": zeros, "one_probability": ones}

def _map_scores(z, values, thresholds):
    return values[np.searchsorted(thresholds, z, side="left")]

def _safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

def _save_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

def _pearson(B, Y):
    if not np.isfinite(B).all() or not np.isfinite(Y).all():
        raise ValueError("Correlation calibration requires finite score arrays")
    if np.std(B) == 0 or np.std(Y) == 0:
        raise ValueError("Correlation calibration requires nonconstant B and Y scores")
    return float(np.corrcoef(B, Y)[0, 1])

def _draw_at_rho(arrays, errors, rho):
    """Reuse fixed latent errors; no domain, split, or label mask enters here."""
    B = _map_scores(arrays["latent_mean_B"] + float(arrays["sigma_B"]) * errors[:, 0],
                    arrays["values_B"], arrays["thresholds_B"])
    Y = _map_scores(arrays["latent_mean_Y"] + float(arrays["sigma_Y"]) *
                    (rho * errors[:, 0] + np.sqrt(1 - rho ** 2) * errors[:, 1]),
                    arrays["values_Y"], arrays["thresholds_Y"])
    return B, Y

def _calibrate_fixed_draw(arrays, errors, target, rho_limit):
    """Search a bracketed step function and verify the best observed candidate.

    The empirical correlation is not assumed to be continuous or everywhere
    monotone in rho. Brent's bracket search supplies candidate values; success
    requires the actual mapped scores to meet the explicit tolerance.
    """
    B, original_Y = _draw_at_rho(arrays, errors, float(arrays["rho"]))
    original_correlation = _pearson(B, original_Y)
    candidates = {}

    def objective(rho):
        rho = float(rho)
        if rho not in candidates:
            _, Y = _draw_at_rho(arrays, errors, rho)
            candidates[rho] = _pearson(B, Y)
        return candidates[rho] - target

    low, high = -rho_limit, rho_limit
    left, right = objective(low), objective(high)
    if left * right > 0:
        raise ValueError(
            f"Target correlation {target:g} is not bracketed by rho endpoints "
            f"[{low:g}, {high:g}], whose correlations are "
            f"{candidates[low]:.9f} and {candidates[high]:.9f}")
    if left != 0 and right != 0:
        result = brentq(objective, low, high, xtol=1e-14, rtol=1e-14, maxiter=200)
        objective(result)
        # Inspect both floating-point sides of a possible score-map jump.
        objective(max(low, float(np.nextafter(result, -np.inf))))
        objective(min(high, float(np.nextafter(result, np.inf))))
    best_rho = min(candidates, key=lambda value: (abs(candidates[value] - target), value))
    actual = candidates[best_rho]
    if abs(actual - target) > CORRELATION_TOLERANCE:
        raise ValueError(
            f"Search did not attain fixed-draw target correlation {target:g} "
            f"within tolerance {CORRELATION_TOLERANCE:g}; best searched value "
            f"is {actual:.9f} at rho={best_rho:.12g}")
    _, Y = _draw_at_rho(arrays, errors, best_rho)
    return B, Y, {
        "protocol": CALIBRATION_PROTOCOL,
        "target_kind": "empirical_fixed_dataset_pearson",
        "target_correlation": target,
        "actual_correlation": actual,
        "absolute_error": abs(actual - target),
        "rho_used": best_rho,
        "rho_original": float(arrays["rho"]),
        "original_correlation": original_correlation,
        "rho_bounds": [low, high],
        "endpoint_correlations": [candidates[low], candidates[high]],
        "tolerance": CORRELATION_TOLERANCE,
        "preferred_tolerance": PREFERRED_TOLERANCE,
        "preferred_tolerance_met": bool(abs(actual - target) <= PREFERRED_TOLERANCE),
        "objective_evaluations": len(candidates),
        "common_random_numbers": True,
        "conditional_truth": "unchanged fitted conditional means",
        "note": ("Targets describe the realized fixed dataset, not population correlation. "
                 "Only latent error rho changes; score maps, conditional marginal laws, "
                 "X, and Gaussian error draws are shared. No split or domain mask is used."),
    }
