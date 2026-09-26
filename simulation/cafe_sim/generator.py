"""Refit, reconstruct and validate the fixed HealthBench score generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .numerics import (DEFAULT_ALPHAS, _normal_scores, _fit_latent, _calibrate,
                       _conditional_moments, _draw_at_rho, _calibrate_fixed_draw,
                       _mixture_cdf, _save_npz, seed)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("same_evaluator", "corr_040", "corr_060", "corr_080")


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refit(output):
    """Reproduce the Flash-Lite update, preserving the original auxiliary fit.

    The reported experiment retained the previously fitted GPT-4.1 generator
    bit for bit and refitted the target channel with common cross-validation
    folds. The packaged auxiliary fit is therefore part of the reference data.
    All fitting and threshold routines are provided in numerics.py.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refit output must be a new directory")
    reference = arrays(ROOT / "data/reference.npz")
    saved = arrays(ROOT / "data/generator/generator.npz")
    truth = arrays(ROOT / "data/generator/truth.npz")
    normal = np.column_stack([_normal_scores(reference[j]) for j in ("B", "Y")])
    fitted = _fit_latent(reference["X"], normal, np.asarray(DEFAULT_ALPHAS), 5, 3, seed(123, 700))
    residual = normal[:, 1] - fitted["oof"][:, 1]
    sigma = float(np.sqrt(np.mean(residual ** 2)))
    values, probabilities, thresholds = _calibrate(fitted["means"][:, 1], sigma, reference["Y"])
    saved.update(latent_mean_Y=fitted["means"][:, 1], sigma_Y=np.asarray(sigma),
                 values_Y=values, probabilities_Y=probabilities, thresholds_Y=thresholds,
                 oof_latent_Y=fitted["oof"][:, 1], oof_residual_Y=residual,
                 coefficient_Y=fitted["coefficients"][:, 1], intercept_Y=fitted["intercepts"][1],
                 rho=np.asarray(np.clip(np.corrcoef(saved["oof_residual_B"], residual)[0, 1], -.95, .95)))
    moments = _conditional_moments(saved["latent_mean_Y"], sigma, values, thresholds)
    saved["oof_score_mean_Y"] = _conditional_moments(fitted["oof"][:, 1], sigma, values, thresholds)["mean"]
    truth.update({f"Y_{key}": value for key, value in moments.items()})
    output.mkdir(parents=True)
    _save_npz(output / "generator.npz", saved)
    _save_npz(output / "truth.npz", truth)
    json_write(output / "fit.json", {"outer_folds": 5, "inner_folds": 3,
               "alphas": DEFAULT_ALPHAS, "selected_alphas": fitted["full_selected_alphas"],
               "auxiliary_fit": "preserved supplied reference", "base_seed": 123})
    return {"status": "complete", "target_channel_refitted": True}


def regenerate(output, generator_dir=None, recalibrate=False):
    """Generate all four datasets into a new directory, never over reference data."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Reconstruction output must be a new directory")
    base = Path(generator_dir) if generator_dir else ROOT / "data/generator"
    g, truth = arrays(base / "generator.npz"), arrays(base / "truth.npz")
    settings = json.loads((ROOT / "data/settings.json").read_text())
    draws = np.random.default_rng(seed(123, 901)).standard_normal((5000, 2))
    output.mkdir(parents=True)
    for setting in SETTINGS:
        info = settings[setting]
        rho = info["correlation"]["rho_used"]
        if setting != "same_evaluator" and recalibrate:
            b, y, calibration = _calibrate_fixed_draw(g, draws, info["correlation"]["target_correlation"], .95)
        else:
            b, y = _draw_at_rho(g, draws, float(g["rho"]) if rho is None else rho)
            calibration = info["correlation"].copy()
        if setting == "same_evaluator":
            y = b.copy()
        source = truth["B_mean"] + .5 * (b - truth["B_mean"])
        scores = {"B": b, "Y": y, "B_source": source, "B_target": b, "Y_target": y,
                  "generation_seed": np.asarray(seed(123, 901), dtype=np.int64)}
        channel = "B" if setting == "same_evaluator" else "Y"
        if channel == "B":
            scores["Y_source"] = source.copy()
        separate_truth = {"B_conditional_mean": truth["B_mean"], "Y_conditional_mean": truth[f"{channel}_mean"],
                          "B_finite_population": b, "Y_finite_population": y,
                          "B_source_conditional_mean": truth["B_mean"],
                          "B_source_conditional_variance": .25 * truth["B_variance"],
                          "B_target_conditional_variance": truth["B_variance"],
                          "Y_target_conditional_variance": truth[f"{channel}_variance"]}
        folder = output / setting
        folder.mkdir()
        _save_npz(folder / "scores.npz", scores)
        _save_npz(folder / "truth.npz", separate_truth)
        calibration["actual_correlation"] = float(np.corrcoef(b, y)[0, 1])
        json_write(folder / "calibration.json", calibration)
    return {"status": "complete", "settings": list(SETTINGS), "recalibrated": recalibrate}


def validate():
    """Verify conditional moments, calibrated scores, exact counts and raw truth."""
    reference = arrays(ROOT / "data/reference.npz")
    g, truth = arrays(ROOT / "data/generator/generator.npz"), arrays(ROOT / "data/generator/truth.npz")
    assert reference["X"].shape == (5000, 80) and np.isfinite(reference["X"]).all()
    assert reference["target_masks"].shape == (500, 5000)
    assert np.all(reference["target_masks"].sum(axis=1) == 3561)
    marginal_error, moment_error = 0., 0.
    for channel in ("B", "Y"):
        means, scale = g[f"latent_mean_{channel}"], float(g[f"sigma_{channel}"])
        thresholds, values = g[f"thresholds_{channel}"], g[f"values_{channel}"]
        expected = np.cumsum(g[f"probabilities_{channel}"])[:-1]
        actual = np.array([_mixture_cdf(t, means, scale) for t in thresholds])
        marginal_error = max(marginal_error, float(np.max(np.abs(actual - expected))))
        moments = _conditional_moments(means, scale, values, thresholds)
        for key, value in moments.items():
            moment_error = max(moment_error, float(np.max(np.abs(value - truth[f"{channel}_{key}"])) ))
        for power in (1, 2):
            key = "mean" if power == 1 else "moment2"
            assert abs(moments[key].mean() - np.mean(reference[channel] ** power)) < 1e-10
        for endpoint in (0, 1):
            key = "zero_probability" if endpoint == 0 else "one_probability"
            assert abs(moments[key].mean() - np.mean(reference[channel] == endpoint)) < 1e-10
    assert marginal_error <= 1e-10 and moment_error <= 1e-12
    diagnostics = {}
    errors = np.random.default_rng(seed(123, 901)).standard_normal((5000, 2))
    settings = json.loads((ROOT / "data/settings.json").read_text())
    for setting in SETTINGS:
        s = arrays(ROOT / "data/datasets" / setting / "scores.npz")
        t = arrays(ROOT / "data/datasets" / setting / "truth.npz")
        info = settings[setting]["correlation"]
        b, y = _draw_at_rho(g, errors, float(g["rho"]) if info["rho_used"] is None else info["rho_used"])
        if setting == "same_evaluator":
            y = b.copy()
            assert np.array_equal(s["Y_source"], s["B_source"])
        assert np.array_equal(s["B_target"], b) and np.array_equal(s["Y_target"], y)
        assert np.array_equal(s["B_source"], truth["B_mean"] + .5 * (b - truth["B_mean"]))
        assert np.array_equal(t["B_source_conditional_mean"], t["B_conditional_mean"])
        assert np.array_equal(t["B_source_conditional_variance"], .25 * t["B_target_conditional_variance"])
        for key in ("B_source", "B_target", "Y_target"):
            assert np.isfinite(s[key]).all() and np.all((s[key] >= 0) & (s[key] <= 1))
        corr = float(np.corrcoef(b, y)[0, 1])
        target = 1. if setting == "same_evaluator" else info["target_correlation"]
        assert abs(corr - target) <= 1e-4
        for i, mask in enumerate(reference["target_masks"]):
            case_seed = seed(123, int(reference["split_seeds"][i]), 2)
            priority = np.random.default_rng(seed(case_seed, 0)).random(5000)[mask]
            order = np.argsort(priority, kind="stable")
            previous = set()
            for n in (200, 300, 500):
                current = set(order[:n])
                assert len(current) == n and previous <= current
                previous = current
        diagnostics[setting] = {"target_correlation": target, "actual_correlation": corr,
                                "source_variance_ratio": .25}
    return {"status": "passed", "marginal_cdf_max_error": marginal_error,
            "conditional_moment_max_error": moment_error, "settings": diagnostics,
            "split_count": 500, "source_count": 1439, "target_count": 3561}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "regenerate", "refit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generator-dir", type=Path)
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()
    if args.command != "validate" and args.output is None:
        parser.error("--output is required for refit and regenerate")
    result = validate() if args.command == "validate" else refit(args.output) if args.command == "refit" else regenerate(args.output, args.generator_dir, args.recalibrate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
