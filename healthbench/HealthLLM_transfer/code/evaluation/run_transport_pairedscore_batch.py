"""Execute paired-score target-mean estimation on observed scores."""

from __future__ import annotations
import json
import os
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from covariateshift.common_mean_regression import RegressorFactory, make_regressor_factory, regression_config
from covariateshift.kernel_based_methods import fit_kmm_ratio_predictor
from evaluation import calculate_transport_paired as paired
from evaluation import run_transport_batch as batch
from evaluation import summarize_transport_paired as report
from evaluation.calculate_transport import RatioFactory, _seed
from evaluation.calculate_transport_paired import METHODS
from evaluation.reweighting_execution import (
    load_common_inputs,
    load_config,
    load_feature,
    named,
    resolve_path,
    score_path_keys,
    select_score_set,
)
from evaluation.summarize_transport_paired import ESTIMATE_COLUMNS
from meta_eval.config import ExperimentConfig
from meta_eval.provenance import build_run_provenance, git_state

DEFAULT_CONFIG = ROOT / "configs/healthbench_transport_paired_flashlite_ridge_fixed_labels.yaml"
RUNNER = "evaluation.run_transport_pairedscore_batch"


def _estimate_columns(config: Mapping[str, Any]) -> list[str]:
    columns = [*ESTIMATE_COLUMNS]
    if "label_counts" in config["split"]:
        columns.extend(["n_target_labels", "n_target"])
    if "crossfit_comparison" in config:
        columns.append("outer_folds")
    return ["score_set", *columns] if "score_sets" in config else columns


def load_transport_config(path: Path) -> dict[str, Any]:
    """Return paired-score settings with resolved data, feature, and KMM configuration."""
    config = load_config(path)
    config["regression"] = regression_config(config["regression"])
    if config["regression"]["name"] == "tabpfn":
        checkpoint = Path(os.environ[config["regression"]["checkpoint_env"]]).expanduser().resolve()
        config["regression"]["checkpoint"] = str(checkpoint)
    if "source_control" in config:
        raise ValueError("source_control overrides are incompatible with committed-run provenance")
    if Path(config["artifact_root"]) != Path("artifacts/runs"):
        raise ValueError("artifact_root must be artifacts/runs")
    density = config["density_ratio"]
    feature = dict(named(config, "feature_sets", config["feature_set"]))
    representation = dict(named(config, "representations", feature["representation"]))
    representation["path"] = str(resolve_path(representation["path"]).resolve())
    config["representations"] = [representation]
    config["feature_sets"] = [feature]
    cases = [case["name"] for case in config["cases"]]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("cases must select unique case names")
    method = density["method"]
    if method["name"] != "kmm":
        raise ValueError("density_ratio.method must be KMM")
    if config["split"]["n_folds"] < 2 or density["median_max_points"] < 2:
        raise ValueError("KMM requires at least two folds and median-distance points")
    config["paths"] = {key: str(resolve_path(value).resolve()) for key, value in config["paths"].items()}
    if "target_mean_csv" in config.get("export", {}):
        config["export"]["target_mean_csv"] = str(resolve_path(config["export"]["target_mean_csv"]).resolve())
    with np.load(config["paths"]["split"], allow_pickle=False) as archive:
        n_splits = len(archive["seeds"])
    ids = config["split"]["ids"]
    config["split"]["ids"] = list(range(1, n_splits + 1)) if ids is None else ids
    ids = config["split"]["ids"]
    if not ids or len(set(ids)) != len(ids) or any((not 1 <= index <= n_splits for index in ids)):
        raise ValueError("split.ids must be unique one-based IDs in the saved split archive")
    methods = [entry["name"] for entry in config["methods"]]
    if not methods or len(set(methods)) != len(methods) or (not set(methods) <= set(METHODS)):
        raise ValueError("methods must select unique supported paired-score estimators")
    if ("label_counts" in config["split"]) == ("label_probabilities" in config["split"]):
        raise ValueError("Specify exactly one of split.label_counts and split.label_probabilities")
    if "label_counts" in config["split"]:
        counts = config["split"]["label_counts"]
        if (
            not counts
            or len(set(counts)) != len(counts)
            or any((type(count) is not int or count < 2 for count in counts))
        ):
            raise ValueError("label_counts must be unique integers of at least two")
    else:
        probabilities = config["split"]["label_probabilities"]
        if (
            not probabilities
            or len(set(probabilities)) != len(probabilities)
            or any((not 0 < value < 1 for value in probabilities))
        ):
            raise ValueError("label_probabilities must be unique values in (0, 1)")
    if not config["models"] or len(set(config["models"])) != len(config["models"]):
        raise ValueError("models must select unique model names")
    if "cafe" in methods:
        spec = config["cafe"]
        for key in ("n_inner_folds", "median_max_points"):
            if type(spec[key]) is not int or spec[key] < 2:
                raise ValueError(f"cafe.{key} must be an integer of at least two")
        candidates = np.asarray(spec["lambda_a"], dtype=float)
        if (
            candidates.ndim != 1
            or not len(candidates)
            or (not np.all(np.isfinite(candidates) & (candidates > 0)))
        ):
            raise ValueError("cafe.lambda_a must be nonempty, finite, and positive")
    if "score_sets" in config:
        if not config["score_sets"]:
            raise ValueError("score_sets must contain at least one named score specification")
        for name in config["score_sets"]:
            select_score_set(config, name, dataset_keys=("auxiliary_score_key", "outcome_score_key"))
        if "scores" not in score_path_keys(config):
            config["paths"].pop("scores", None)
        config["dataset"] = {
            key: value
            for key, value in config["dataset"].items()
            if key not in {"auxiliary_score_key", "outcome_score_key"}
        }
    return config


def _load_case_inputs(config: Mapping[str, Any], case_name: str) -> batch.TransportCaseInputs:
    loader_config = {
        **config,
        "representations": {entry["name"]: entry for entry in config["representations"]},
    }
    prompt_ids, seeds, target_masks, models, auxiliary_scores = load_common_inputs(
        config, case_name, score_key=config["dataset"]["auxiliary_score_key"]
    )
    outcome_config = config
    if "outcome_scores" in config["paths"]:
        outcome_config = {**config, "paths": {**config["paths"], "scores": config["paths"]["outcome_scores"]}}
    _, _, _, outcome_models, outcome_scores = load_common_inputs(
        outcome_config, case_name, score_key=config["dataset"]["outcome_score_key"]
    )
    if set(outcome_models) != set(models):
        raise ValueError("Auxiliary and outcome score model identifiers differ")
    outcome_columns = {name: index for index, name in enumerate(outcome_models)}
    outcome_scores = outcome_scores[:, [outcome_columns[name] for name in models]]
    features = load_feature(loader_config, config["feature_set"], prompt_ids)
    if np.isinf(auxiliary_scores).any() or np.isinf(outcome_scores).any():
        raise ValueError("Scores must be finite or NaN")
    return batch.TransportCaseInputs(
        prompt_ids, seeds, target_masks, models, auxiliary_scores, outcome_scores, features
    )


def calculate_estimates(
    config: Mapping[str, Any],
    *,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
    split_ids: Sequence[int] | None = None,
    input_cache: batch.TransportInputCache | None = None,
) -> pd.DataFrame:
    """Return configured paired-score means and SEs for one or more score sets."""
    if input_cache is None:
        input_cache = {}
    if "crossfit_comparison" in config and len(config["crossfit_comparison"]["outer_folds"]) > 1:
        tables = [
            calculate_estimates(
                batch.crossfit_config(config, folds),
                regressor_factory=regressor_factory,
                ratio_factory=ratio_factory,
                split_ids=split_ids,
                input_cache=input_cache,
            )
            for folds in config["crossfit_comparison"]["outer_folds"]
        ]
        combined = pd.concat(tables, ignore_index=True)
        combined.attrs["reppi_zero_coefficient_counts"] = [
            row for table in tables for row in table.attrs["reppi_zero_coefficient_counts"]
        ]
        return combined
    if "score_sets" in config:
        tables = []
        diagnostics = []
        for name in config["score_sets"]:
            print(f"score_set={name}", flush=True)
            selected = select_score_set(
                config, name, dataset_keys=("auxiliary_score_key", "outcome_score_key")
            )
            table = _calculate_score_set(
                selected,
                regressor_factory=regressor_factory,
                ratio_factory=ratio_factory,
                split_ids=split_ids,
                input_cache=input_cache,
                score_set=name,
            )
            tables.append(table.assign(score_set=name))
            diagnostics.extend(
                ({**row, "score_set": name} for row in table.attrs["reppi_zero_coefficient_counts"])
            )
        combined = pd.concat(tables, ignore_index=True)[_estimate_columns(config)]
        combined.attrs["reppi_zero_coefficient_counts"] = diagnostics
        return combined
    return _calculate_score_set(
        config,
        regressor_factory=regressor_factory,
        ratio_factory=ratio_factory,
        split_ids=split_ids,
        input_cache=input_cache,
    )


def _calculate_score_set(
    config: Mapping[str, Any],
    *,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory,
    split_ids: Sequence[int] | None,
    input_cache: batch.TransportInputCache,
    score_set: str | None = None,
) -> pd.DataFrame:
    selected_ids = config["split"]["ids"] if split_ids is None else split_ids
    if not set(selected_ids) <= set(config["split"]["ids"]):
        raise ValueError("Requested split IDs are outside the resolved configuration")
    estimates = []
    diagnostics = []
    for case in config["cases"]:
        key = (score_set, case["name"])
        if key not in input_cache:
            input_cache[key] = _load_case_inputs(config, case["name"])
        inputs = input_cache[key]
        prompt_ids, seeds, target_masks = (inputs.prompt_ids, inputs.seeds, inputs.target_masks)
        models, auxiliary_scores, outcome_scores = (inputs.models, inputs.source_scores, inputs.target_scores)
        features = inputs.features
        model_columns = {name: index for index, name in enumerate(models)}
        for split_id in selected_ids:
            target_mask = target_masks[split_id - 1]
            split_seed = int(seeds[split_id - 1])
            case_seed = _seed(config["random_seed"], split_seed, int(case["name"].removeprefix("case")))
            uniform = np.random.default_rng(_seed(case_seed, 0)).random(len(prompt_ids))
            for model_index, model_name in enumerate(config["models"]):
                column = model_columns[model_name]
                auxiliary, outcome = (auxiliary_scores[:, column], outcome_scores[:, column])
                source = ~target_mask & np.isfinite(auxiliary)
                target = target_mask & np.isfinite(auxiliary) & np.isfinite(outcome)
                context = {
                    "case": case["name"],
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "model": model_name,
                    "feature_set": config["feature_set"],
                }
                if "crossfit_comparison" in config:
                    context["outer_folds"] = config["split"]["n_folds"]
                evaluations = paired.evaluate_sample(
                    features[source],
                    auxiliary[source],
                    features[target],
                    auxiliary[target],
                    outcome[target],
                    uniform[target],
                    config=config,
                    seed=_seed(case_seed, 1, model_index),
                    regressor_factory=regressor_factory,
                    ratio_factory=ratio_factory,
                )
                for result in evaluations:
                    budget = {"alpha": result.alpha}
                    if "label_counts" in config["split"]:
                        budget.update(n_target_labels=int(result.labeled.sum()), n_target=len(result.labeled))
                    if "reppi" in result.estimates:
                        diagnostics.append(
                            {
                                **context,
                                **budget,
                                "zero_coefficient_count": result.estimates["reppi"]["zero_coefficient_count"],
                                "fold_count": result.estimates["reppi"]["fold_count"],
                            }
                        )
                    for method, estimate in result.estimates.items():
                        estimates.append(
                            {
                                **context,
                                **budget,
                                "method": method,
                                "estimate": estimate["estimate"],
                                "standard_error": estimate["standard_error"],
                            }
                        )
                print(f"{case['name']} split={split_id} model={model_name}", flush=True)
    table = pd.DataFrame(estimates, columns=_estimate_columns(config))
    table.attrs["reppi_zero_coefficient_counts"] = diagnostics
    if not np.isfinite(table["estimate"].to_numpy()).all():
        raise ValueError("Paired-score target mean estimates must be finite")
    return table


def validate_estimates(
    table: pd.DataFrame, config: Mapping[str, Any], split_ids: Sequence[int]
) -> pd.DataFrame:
    """Return ordered estimates and SEs, with NaN for unavailable sampling variances."""
    if "crossfit_comparison" in config:
        folds = config["crossfit_comparison"]["outer_folds"]
        columns = _estimate_columns(config)
        if list(table.columns) != columns or set(table["outer_folds"]) != set(folds):
            raise ValueError("Estimate outer folds differ from the configured comparison")
        branch = {key: value for key, value in config.items() if key != "crossfit_comparison"}
        parts = [
            validate_estimates(
                table.loc[table["outer_folds"] == value, _estimate_columns(branch)], branch, split_ids
            ).assign(outer_folds=value)
            for value in folds
        ]
        ordered = pd.concat(parts, ignore_index=True)[columns]
        ordered.attrs = table.attrs.copy()
        if not np.isfinite(ordered["standard_error"]).all():
            raise ValueError("Cross-fit comparison requires finite standard errors")
        if len(folds) > 1:
            budget_column = "n_target_labels" if "label_counts" in config["split"] else "alpha"
            keys = ["case", "split_id", "model", budget_column, "method"]
            controls = ordered[ordered["method"].isin(["labelled_only", "ppi", "reppi"])]
            reference = controls[controls["outer_folds"] == folds[0]].set_index(keys)
            for value in folds[1:]:
                candidate = (
                    controls[controls["outer_folds"] == value].set_index(keys).reindex(reference.index)
                )
                if not np.allclose(
                    reference[["estimate", "standard_error"]],
                    candidate[["estimate", "standard_error"]],
                    rtol=1e-06,
                    atol=1e-08,
                ):
                    raise ValueError("Fixed comparison baselines differ across outer folds")
        return ordered
    if "score_sets" in config:
        if list(table.columns) != _estimate_columns(config):
            raise ValueError("Estimate columns differ from the score-set schema")
        if set(table["score_set"]) != set(config["score_sets"]):
            raise ValueError("Estimate score sets differ from the resolved configuration")
        tables = []
        for name in config["score_sets"]:
            selected = select_score_set(
                config, name, dataset_keys=("auxiliary_score_key", "outcome_score_key")
            )
            part = table.loc[table["score_set"] == name, _estimate_columns(selected)]
            tables.append(validate_estimates(part, selected, split_ids).assign(score_set=name))
        ordered = pd.concat(tables, ignore_index=True)[_estimate_columns(config)]
        ordered.attrs = table.attrs.copy()
        return ordered
    ordered = batch.validate_estimates(table, config, split_ids, columns=_estimate_columns(config))
    ordered.attrs = table.attrs.copy()
    standard_error = ordered["standard_error"].to_numpy(dtype=float)
    if np.isinf(standard_error).any() or np.any(standard_error < 0):
        raise ValueError("Standard errors must be nonnegative and finite or NaN")
    if "label_counts" in config["split"] and (not np.isfinite(standard_error).all()):
        raise ValueError("Fixed-count standard errors must be finite")
    return ordered


def reppi_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    """Return validated zero-coefficient counts aligned with the RePPI estimates."""
    identifiers = ["case", "split_id", "split_seed", "model", "feature_set", "alpha"]
    if "n_target_labels" in table:
        identifiers.extend(["n_target_labels", "n_target"])
    if "outer_folds" in table:
        identifiers.insert(0, "outer_folds")
    if "score_set" in table:
        identifiers.insert(0, "score_set")
    expected = pd.MultiIndex.from_frame(table.loc[table["method"].eq("reppi"), identifiers])
    diagnostics = pd.DataFrame(table.attrs["reppi_zero_coefficient_counts"]).set_index(identifiers)
    if (
        not diagnostics.index.is_unique
        or len(diagnostics) != len(expected)
        or (not diagnostics.index.isin(expected).all())
    ):
        raise ValueError("RePPI diagnostic identifiers differ from the estimates")
    counts = diagnostics["zero_coefficient_count"].to_numpy()
    if not np.issubdtype(counts.dtype, np.integer) or np.any((counts < 0) | (counts > 3)):
        raise ValueError("RePPI zero-coefficient counts must be integers between zero and three")
    if not diagnostics["fold_count"].eq(3).all():
        raise ValueError("RePPI diagnostics must record three folds per estimate")
    return diagnostics.reindex(expected).reset_index()


def _regressor_factory(config: Mapping[str, Any]) -> RegressorFactory:
    return make_regressor_factory(config["regression"])


def execute_paired_transport(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> pd.DataFrame:
    """Write and return the configured paired-score estimates in one run directory."""
    path = run_dir / "estimates.parquet"
    if path.exists():
        raise FileExistsError(path)
    table = calculate_estimates(config, regressor_factory=regressor_factory, ratio_factory=ratio_factory)
    batch._write_table(table, path)
    if "export" in config:
        report.export_target_means(validate_estimates(table, config, config["split"]["ids"]), config)
    return table


def run_paired(config_path: Path) -> Path:
    """Run YAML-configured paired-score estimation from a clean committed worktree."""
    state = git_state(PROJECT_ROOT)
    if state["dirty"] or state["commit"] is None:
        raise RuntimeError("Paired transport evaluation requires a clean, committed execution worktree")
    config = load_transport_config(config_path)
    regressor_factory = _regressor_factory(config)
    data_files = {
        **{key: Path(value) for key, value in config["paths"].items()},
        "embedding": Path(config["representations"][0]["path"]),
    }
    code_paths = [
        Path("HealthLLM_transfer/code"),
        Path("HealthLLM_transfer/configs"),
        Path("src/meta_eval"),
        Path("pyproject.toml"),
    ]
    runtime_packages = ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "PyYAML", "adapt", "cvxopt")
    if config["regression"]["name"] == "tabpfn":
        data_files["tabpfn_checkpoint"] = Path(config["regression"]["checkpoint"])
        runtime_packages += ("tabpfn", "torch")
    provenance = build_run_provenance(
        ExperimentConfig.from_mapping(config),
        PROJECT_ROOT,
        data_files=data_files,
        code_paths=tuple(code_paths),
        dependency_versions={name: version(name) for name in runtime_packages},
    )
    identity = provenance["identity"]
    if (
        identity["git_commit"] != state["commit"]
        or identity["git_worktree"]["dirty"]
        or provenance["observed_repository_state"]["dirty"]
    ):
        raise RuntimeError("Execution worktree changed while preparing paired transport provenance")
    run_dir = ROOT / config["artifact_root"] / provenance["run_id"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config.resolved.yaml").open("x", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False))
    with (run_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(provenance, indent=2) + "\n")
    execute_paired_transport(config, run_dir, regressor_factory=regressor_factory)
    with (run_dir / "_SUCCESS").open("x", encoding="utf-8") as handle:
        handle.write(provenance["fingerprint"] + "\n")
    return run_dir


def main() -> None:
    """Run observed-score estimation with the shared batch interface."""
    batch.main(default_config=DEFAULT_CONFIG, expected_module="evaluation.calculate_transport_paired")


if __name__ == "__main__":
    main()
