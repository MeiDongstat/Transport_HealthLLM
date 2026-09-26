"""Run YAML-selected transport estimators in batches with per-split results."""

from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
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
from evaluation import calculate_transport as base
from evaluation.calculate_transport import METHODS, RatioFactory, _seed
from evaluation.reweighting_execution import (
    load_common_inputs,
    load_config,
    load_feature,
    named,
    resolve_path,
    score_path_keys,
    select_score_set,
)
from meta_eval.config import ExperimentConfig
from meta_eval.provenance import build_run_provenance, git_state

RUNNER = "evaluation.run_transport_batch"
MODULES = ("evaluation.calculate_transport", "evaluation.calculate_transport_paired")
ESTIMATE_COLUMNS = ["case", "split_id", "split_seed", "model", "feature_set", "alpha", "method", "estimate"]
ESTIMATE_KEYS = ["case", "split_id", "model", "alpha", "method"]
TRANSPORT_ESTIMATE_COLUMNS = [*ESTIMATE_COLUMNS, "standard_error"]
SCORE_SET_COLUMNS = ["score_set", *TRANSPORT_ESTIMATE_COLUMNS]
COUNT_ESTIMATE_COLUMNS = [*TRANSPORT_ESTIMATE_COLUMNS, "n_target_labels", "n_target"]
TIMING_COLUMNS = ["fit_seconds", "q_seconds", "fit_order"]


@dataclass(frozen=True)
class TransportCaseInputs:
    """Hold aligned, split-independent inputs for one case and score set."""

    prompt_ids: np.ndarray
    seeds: np.ndarray
    target_masks: np.ndarray
    models: np.ndarray
    source_scores: np.ndarray
    target_scores: np.ndarray
    features: np.ndarray


TransportInputCache = dict[tuple[str | None, str], TransportCaseInputs]


def _estimate_columns(config: Mapping[str, Any]) -> list[str]:
    columns = COUNT_ESTIMATE_COLUMNS if "label_counts" in config["split"] else TRANSPORT_ESTIMATE_COLUMNS
    return ["score_set", *columns] if "score_sets" in config else list(columns)


def _load_base_config(path: Path, *, supported_methods: Sequence[str] = METHODS) -> dict[str, Any]:
    """Return transport settings resolved against the canonical kernel configuration."""
    config = load_config(path)
    kernel_path = resolve_path(config["density_ratio"]["config_path"]).resolve()
    kernel_config = load_config(kernel_path)
    feature = dict(named(kernel_config, "feature_sets", config["feature_set"]))
    representation = dict(kernel_config["representations"][feature["representation"]])
    representation.update(name=feature["representation"], path=str(resolve_path(representation["path"])))
    config["representations"] = [representation]
    config["feature_sets"] = [feature]
    config["cases"] = [dict(named(kernel_config, "cases", name)) for name in config["cases"]]
    config["density_ratio"] = {
        "config_path": str(kernel_path),
        "method": dict(named(kernel_config, "methods", "kmm")),
        "median_max_points": int(kernel_config["execution"]["median_max_points"]),
    }
    config["paths"] = {key: str(resolve_path(value).resolve()) for key, value in config["paths"].items()}
    with np.load(config["paths"]["split"], allow_pickle=False) as archive:
        n_splits = len(archive["seeds"])
    ids = config["split"]["ids"]
    config["split"]["ids"] = list(range(1, n_splits + 1)) if ids is None else ids
    ids = config["split"]["ids"]
    if not ids or len(set(ids)) != len(ids) or any((not 1 <= i <= n_splits for i in ids)):
        raise ValueError("split.ids must be unique one-based IDs in the saved split archive")
    methods = [entry["name"] for entry in config["methods"]]
    if not methods or len(set(methods)) != len(methods) or (not set(methods) <= set(supported_methods)):
        raise ValueError("methods must select unique supported estimators")
    if ("label_counts" in config["split"]) == ("label_probabilities" in config["split"]):
        raise ValueError("Specify exactly one of split.label_counts and split.label_probabilities")
    if "label_counts" in config["split"]:
        counts = config["split"]["label_counts"]
        if (
            not counts
            or any((type(value) is not int or value < 2 for value in counts))
            or len(set(counts)) != len(counts)
        ):
            raise ValueError("label_counts must be unique integers of at least two")
        if config["execution"]["module"] != MODULES[0]:
            raise ValueError("label_counts requires the common-outcome calculation module")
    else:
        probabilities = config["split"]["label_probabilities"]
        if (
            not probabilities
            or len(set(probabilities)) != len(probabilities)
            or any((not 0 < value < 1 for value in probabilities))
        ):
            raise ValueError("label_probabilities must be unique values in (0, 1)")
    if len(set(config["models"])) != len(config["models"]):
        raise ValueError("Duplicate configured model names")
    if "output" in config:
        config["output"]["target_mean"] = str(resolve_path(config["output"]["target_mean"]).resolve())
    return config


def _result_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    if "output" in config:
        return Path(config["output"]["target_mean"])
    return run_dir / "estimates.parquet"


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        if path.suffix == ".csv":
            table.to_csv(temporary, index=False)
        else:
            table.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_estimate_rows(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    split_ids: Sequence[int],
    *,
    columns: Sequence[str] = ESTIMATE_COLUMNS,
) -> pd.DataFrame:
    """Return externally produced estimates in configuration order after coverage checks."""
    if list(table.columns) != list(columns):
        raise ValueError("Estimate columns differ from the transport schema")
    fixed_counts = "label_counts" in config["split"]
    budget_column = "n_target_labels" if fixed_counts else "alpha"
    keys = ["case", "split_id", "model", budget_column, "method"]
    budgets = config["split"]["label_counts" if fixed_counts else "label_probabilities"]
    expected = pd.MultiIndex.from_product(
        [
            [case["name"] for case in config["cases"]],
            split_ids,
            config["models"],
            budgets,
            [method["name"] for method in config["methods"]],
        ],
        names=keys,
    )
    indexed = table.set_index(keys)
    if (
        not indexed.index.is_unique
        or len(indexed) != len(expected)
        or (not indexed.index.isin(expected).all())
    ):
        raise ValueError("Estimate identifiers contain missing, duplicate, or unexpected combinations")
    ordered = indexed.reindex(expected).reset_index()[list(columns)]
    if fixed_counts:
        labels, target = (ordered["n_target_labels"].to_numpy(), ordered["n_target"].to_numpy())
        if any((not np.issubdtype(values.dtype, np.integer) for values in (labels, target))) or np.any(
            (labels < 2) | (labels >= target)
        ):
            raise ValueError("Fixed label counts must be integers between two and n_target - 1")
        if not np.allclose(ordered["alpha"], labels / target, rtol=1e-12, atol=0):
            raise ValueError("Fixed-count alpha must equal n_target_labels / n_target")
        if not ordered.groupby(["case", "split_id", "model"])["n_target"].nunique().eq(1).all():
            raise ValueError("n_target must agree across label budgets and methods")
    if not np.isfinite(ordered["estimate"].to_numpy()).all():
        raise ValueError("Target mean estimates must be finite")
    if not ordered["feature_set"].eq(config["feature_set"]).all():
        raise ValueError("Estimate feature set differs from the resolved configuration")
    with np.load(config["paths"]["split"], allow_pickle=False) as archive:
        expected_seeds = archive["seeds"][ordered["split_id"].to_numpy(dtype=int) - 1]
    if not np.array_equal(ordered["split_seed"].to_numpy(), expected_seeds):
        raise ValueError("Estimate seeds differ from the saved splits")
    return ordered


def _load_common_outcome_config(path: Path) -> dict[str, Any]:
    """Return resolved settings for six baselines and CAFE."""
    config = _load_base_config(path, supported_methods=METHODS)
    if any((method["name"] == "cafe" for method in config["methods"])):
        specification = config["cafe"]
        for key in ("n_inner_folds", "median_max_points"):
            if type(specification[key]) is not int or specification[key] < 2:
                raise ValueError(f"cafe.{key} must be an integer of at least two")
        penalties = specification["lambda_a"]
        if (
            not isinstance(penalties, list)
            or not penalties
            or any(
                (
                    type(value) not in (int, float) or not np.isfinite(value) or value <= 0
                    for value in penalties
                )
            )
        ):
            raise ValueError("cafe.lambda_a must be a nonempty list of finite positive penalties")
    config["regression"] = regression_config(config["regression"])
    if config["regression"]["name"] == "tabpfn":
        checkpoint = Path(os.environ[config["regression"]["checkpoint_env"]]).expanduser().resolve()
        config["regression"]["checkpoint"] = str(checkpoint)
    if "source_control" in config:
        raise ValueError("source_control overrides are incompatible with committed-run provenance")
    if Path(config["artifact_root"]) != Path("artifacts/runs"):
        raise ValueError("artifact_root must be artifacts/runs")
    if "score_sets" in config:
        if not config["score_sets"]:
            raise ValueError("score_sets must contain at least one named score specification")
        for name in config["score_sets"]:
            select_score_set(config, name)
        if "scores" not in score_path_keys(config):
            config["paths"].pop("scores", None)
        config["dataset"] = {
            key: value
            for key, value in config["dataset"].items()
            if key not in {"score_key", "target_score_key"}
        }
    config["runner"] = RUNNER
    return config


def _regressor_factory(config: Mapping[str, Any]) -> RegressorFactory:
    return make_regressor_factory(config["regression"])


def _load_case_inputs(config: Mapping[str, Any], case_name: str) -> TransportCaseInputs:
    loader_config = {**config, "representations": {r["name"]: r for r in config["representations"]}}
    source_key = config["dataset"]["score_key"]
    target_key = config["dataset"].get("target_score_key", source_key)
    prompt_ids, seeds, target_masks, models, source_scores = load_common_inputs(
        config, case_name, score_key=config["dataset"]["score_key"]
    )
    target_scores = source_scores
    if target_key != source_key:
        _, _, _, _, target_scores = load_common_inputs(config, case_name, score_key=target_key)
    features = load_feature(loader_config, config["feature_set"], prompt_ids)
    if np.isinf(source_scores).any() or np.isinf(target_scores).any():
        raise ValueError("Scores must be finite or NaN")
    return TransportCaseInputs(
        prompt_ids, seeds, target_masks, models, source_scores, target_scores, features
    )


def calculate_estimates(
    config: Mapping[str, Any],
    *,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
    split_ids: Sequence[int] | None = None,
    input_cache: TransportInputCache | None = None,
) -> pd.DataFrame:
    """Return target means for configured splits or a supplied subset of their IDs."""
    if input_cache is None:
        input_cache = {}
    if "score_sets" in config:
        tables = []
        for name in config["score_sets"]:
            print(f"score_set={name}", flush=True)
            table = _calculate_score_set(
                select_score_set(config, name),
                regressor_factory=regressor_factory,
                ratio_factory=ratio_factory,
                split_ids=split_ids,
                input_cache=input_cache,
                score_set=name,
            )
            tables.append(table.assign(score_set=name))
        return pd.concat(tables, ignore_index=True)[_estimate_columns(config)]
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
    input_cache: TransportInputCache,
    score_set: str | None = None,
) -> pd.DataFrame:
    selected_ids = config["split"]["ids"] if split_ids is None else split_ids
    if not set(selected_ids) <= set(config["split"]["ids"]):
        raise ValueError("Requested split IDs are outside the resolved configuration")
    estimates = []
    for case in config["cases"]:
        key = (score_set, case["name"])
        if key not in input_cache:
            input_cache[key] = _load_case_inputs(config, case["name"])
        inputs = input_cache[key]
        prompt_ids, seeds, target_masks = (inputs.prompt_ids, inputs.seeds, inputs.target_masks)
        models, source_scores, target_scores = (inputs.models, inputs.source_scores, inputs.target_scores)
        features = inputs.features
        model_columns = {name: index for index, name in enumerate(models)}
        for split_id in selected_ids:
            target_mask = target_masks[split_id - 1]
            split_seed = int(seeds[split_id - 1])
            case_seed = _seed(config["random_seed"], split_seed, int(case["name"].removeprefix("case")))
            uniform = np.random.default_rng(_seed(case_seed, 0)).random(len(prompt_ids))
            for model_index, model_name in enumerate(config["models"]):
                column = model_columns[model_name]
                source_outcome, target_outcome = (source_scores[:, column], target_scores[:, column])
                source = np.isfinite(source_outcome) & ~target_mask
                target = np.isfinite(target_outcome) & target_mask
                context = {
                    "case": case["name"],
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "model": model_name,
                    "feature_set": config["feature_set"],
                }
                unit_seed = _seed(case_seed, 1, model_index)
                evaluations = base.evaluate_sample(
                    features[source],
                    source_outcome[source],
                    features[target],
                    target_outcome[target],
                    uniform[target],
                    config=config,
                    seed=unit_seed,
                    regressor_factory=regressor_factory,
                    ratio_factory=ratio_factory,
                )
                for result in evaluations:
                    for method, estimate in result.estimates.items():
                        row = {
                            **context,
                            "alpha": result.alpha,
                            "method": method,
                            "estimate": estimate["estimate"],
                            "standard_error": estimate["standard_error"],
                        }
                        if "label_counts" in config["split"]:
                            row.update(
                                n_target_labels=int(result.labeled.sum()), n_target=len(result.labeled)
                            )
                        estimates.append(row)
                print(f"{case['name']} split={split_id} model={model_name}", flush=True)
    return pd.DataFrame(estimates, columns=_estimate_columns(config))


def validate_estimates(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    split_ids: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return ordered estimates and SEs, with NaN for unavailable sampling variances."""
    if columns is not None:
        return _validate_estimate_rows(table, config, split_ids, columns=columns)
    if "score_sets" in config:
        if list(table.columns) != _estimate_columns(config):
            raise ValueError("Estimate columns differ from the score-set schema")
        if set(table["score_set"]) != set(config["score_sets"]):
            raise ValueError("Estimate score sets differ from the resolved configuration")
        tables = []
        for name in config["score_sets"]:
            selected_config = select_score_set(config, name)
            part = table.loc[table["score_set"] == name, _estimate_columns(selected_config)]
            ordered = validate_estimates(part, selected_config, split_ids)
            tables.append(ordered.assign(score_set=name))
        return pd.concat(tables, ignore_index=True)[_estimate_columns(config)]
    ordered = _validate_estimate_rows(table, config, split_ids, columns=_estimate_columns(config))
    standard_error = ordered["standard_error"].to_numpy(dtype=float)
    if "label_counts" in config["split"] and (not np.isfinite(standard_error).all()):
        raise ValueError("Fixed-count standard errors must be finite and nonnegative")
    if np.isinf(standard_error).any() or np.any(standard_error < 0):
        raise ValueError("Standard errors must be nonnegative and finite or NaN")
    source_reweighting = ordered["method"].eq("source_reweighting").to_numpy()
    if not np.isnan(standard_error[source_reweighting]).all():
        raise ValueError("Source reweighting standard errors must be NaN")
    return ordered


def execute_transport(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    regressor_factory: RegressorFactory,
    ratio_factory: RatioFactory = fit_kmm_ratio_predictor,
) -> pd.DataFrame:
    """Write and return all configured target means in one sequential calculation."""
    path = _result_path(config, run_dir)
    if path.exists():
        raise FileExistsError(path)
    table = calculate_estimates(config, regressor_factory=regressor_factory, ratio_factory=ratio_factory)
    _write_table(table, path)
    return table


def _load_prepared_run(
    run_dir: Path, *, require_clean: bool, expected_runner: str = RUNNER
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    if config.get("runner") != expected_runner:
        raise ValueError("Prepared run does not use the configured transport runner")
    state = git_state(PROJECT_ROOT)
    if state["commit"] != manifest["identity"]["git_commit"]:
        raise ValueError("Execution commit differs from the prepared run")
    if require_clean and state["dirty"]:
        raise RuntimeError("Transport execution requires a clean, committed worktree")
    if ExperimentConfig.from_mapping(config).to_dict() != manifest["identity"]["config"]:
        raise ValueError("Resolved configuration differs from the run manifest")
    return (config, manifest)


def run_split(run_dir: Path, split_id: int, *, require_clean: bool = True) -> Path:
    """Write one split's target means using a previously prepared run configuration."""
    config, manifest = _load_run(run_dir, require_clean=require_clean)
    if split_id not in config["split"]["ids"]:
        raise ValueError("Requested split ID is outside the resolved configuration")
    if _uses_batches(config) and any(
        (config["execution"].get(key, False) for key in ("batch_by_score_set", "batch_by_case"))
    ):
        raise ValueError("Grouped runs require batch mode")
    path = run_dir / "parts" / f"split_{split_id:04d}.parquet"
    if path.exists():
        raise FileExistsError(path)
    backend = _backend(config)
    table = backend.calculate_estimates(
        config, split_ids=[split_id], regressor_factory=backend._regressor_factory(config)
    )
    table.attrs["run_fingerprint"] = manifest["fingerprint"]
    _write_table(table, path)
    return path


def _uses_batches(config: Mapping[str, Any]) -> bool:
    return "execution" in config and "module" in config["execution"]


def _backend(config: Mapping[str, Any]) -> ModuleType:
    if not _uses_batches(config):
        return sys.modules[__name__]
    module = config["execution"]["module"]
    if module not in MODULES:
        raise ValueError(f"execution.module must be one of {MODULES}")
    if module == "evaluation.calculate_transport_paired":
        return importlib.import_module("evaluation.run_transport_pairedscore_batch")
    return sys.modules[__name__]


def _validate_execution(config: Mapping[str, Any]) -> None:
    for name in ("batch_size", "max_parallel_tasks"):
        value = config["execution"][name]
        if type(value) is not int or value < 1:
            raise ValueError(f"execution.{name} must be a positive integer")
    if Path(config["artifact_root"]) != Path("artifacts/runs") or "output" in config:
        raise ValueError("Batch outputs must stay in artifacts/runs/<run_id>; remove output overrides")
    separate = config["execution"].get("batch_by_score_set", False)
    if type(separate) is not bool:
        raise ValueError("execution.batch_by_score_set must be Boolean")
    if separate:
        if not config.get("score_sets"):
            raise ValueError("Separate score-set batches require score_sets")
        for name in config["score_sets"]:
            if not isinstance(name, str) or not name or name in {".", ".."} or (Path(name).name != name):
                raise ValueError("Score-set names must be nonempty directory names")
    by_case = config["execution"].get("batch_by_case", False)
    if type(by_case) is not bool:
        raise ValueError("execution.batch_by_case must be Boolean")
    if by_case and separate:
        raise ValueError("Select either batch_by_case or batch_by_score_set")
    if "crossfit_comparison" in config:
        folds = config["crossfit_comparison"]["outer_folds"]
        if (
            not folds
            or any((type(value) is not int or value < 2 for value in folds))
            or len(set(folds)) != len(folds)
        ):
            raise ValueError("crossfit_comparison.outer_folds must contain unique integers of at least two")
        if separate or "score_sets" in config or config["execution"]["module"] != MODULES[1]:
            raise ValueError("Cross-fit comparison requires paired-score batches without score-set grouping")
    case_exports = config.get("export", {}).get("case_estimates", False)
    if type(case_exports) is not bool:
        raise ValueError("export.case_estimates must be Boolean")
    if by_case or case_exports:
        names = [case["name"] for case in config["cases"]]
        if (
            not names
            or len(set(names)) != len(names)
            or any(
                (
                    not isinstance(name, str) or not name or name in {".", ".."} or (Path(name).name != name)
                    for name in names
                )
            )
        ):
            raise ValueError("Case names must be unique nonempty directory names")


def load_transport_config(path: Path) -> dict[str, Any]:
    """Return the selected calculation module's resolved configuration and batch settings."""
    backend = _backend(load_config(path))
    if backend is sys.modules[__name__]:
        config = _load_common_outcome_config(path)
    else:
        config = backend.load_transport_config(path)
    if _uses_batches(config):
        _validate_execution(config)
    config["runner"] = RUNNER
    return config


def select_splits(config: Mapping[str, Any], batch_id: int) -> list[int]:
    """Return one zero-based batch of split IDs in their configured order."""
    split_ids = config["split"]["ids"]
    batch_size = config["execution"]["batch_size"]
    n_batches = (len(split_ids) + batch_size - 1) // batch_size
    if not 0 <= batch_id < n_batches:
        raise ValueError(f"batch_id must be between 0 and {n_batches - 1}")
    return list(split_ids[batch_id * batch_size : (batch_id + 1) * batch_size])


def array_spec(config: Mapping[str, Any], *, outer_folds: int | None = None) -> str:
    """Return the batch range and concurrency limit for the configured computation."""
    batch_size = config["execution"]["batch_size"]
    n_batches = (len(config["split"]["ids"]) + batch_size - 1) // batch_size
    groups = list(_execution_groups(config))
    if outer_folds is not None:
        crossfit_config(config, outer_folds)
        selected = [
            index
            for index, name in enumerate(groups)
            if name == f"folds_{outer_folds}" or name.startswith(f"folds_{outer_folds}/")
        ]
        start, stop = (selected[0] * n_batches, (selected[-1] + 1) * n_batches - 1)
        return f"{start}-{stop}%{config['execution']['max_parallel_tasks']}"
    return f"0-{n_batches * len(groups) - 1}%{config['execution']['max_parallel_tasks']}"


def select_batch(config: Mapping[str, Any], batch_id: int) -> tuple[str | None, list[int]]:
    """Return the case or score-set group and split IDs assigned to an array task."""
    batch_size = config["execution"]["batch_size"]
    n_batches = (len(config["split"]["ids"]) + batch_size - 1) // batch_size
    names = list(_execution_groups(config))
    if not 0 <= batch_id < n_batches * len(names):
        raise ValueError(f"batch_id must be between 0 and {n_batches * len(names) - 1}")
    score_index, split_batch = divmod(batch_id, n_batches)
    return (names[score_index], select_splits(config, split_batch))


def _score_set_config(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {**config, "score_sets": {name: config["score_sets"][name]}}


def crossfit_config(config: Mapping[str, Any], outer_folds: int) -> dict[str, Any]:
    """Return the configured branch with its integer outer fold count."""
    if outer_folds not in config["crossfit_comparison"]["outer_folds"]:
        raise ValueError("Requested outer_folds is outside the configured comparison")
    return {
        **config,
        "crossfit_comparison": {"outer_folds": [outer_folds]},
        "split": {**config["split"], "n_folds": outer_folds},
    }


def _execution_groups(config: Mapping[str, Any]) -> dict[str | None, Mapping[str, Any]]:
    if "crossfit_comparison" in config:
        if config["execution"].get("batch_by_case", False):
            return {
                f"folds_{folds}/{case['name']}": {**crossfit_config(config, folds), "cases": [case]}
                for folds in config["crossfit_comparison"]["outer_folds"]
                for case in config["cases"]
            }
        return {
            f"folds_{folds}": crossfit_config(config, folds)
            for folds in config["crossfit_comparison"]["outer_folds"]
        }
    if _uses_batches(config) and config["execution"].get("batch_by_case", False):
        return {case["name"]: {**config, "cases": [case]} for case in config["cases"]}
    if _uses_batches(config) and config["execution"].get("batch_by_score_set", False):
        return {name: _score_set_config(config, name) for name in config["score_sets"]}
    return {None: config}


def _load_run(run_dir: Path, *, require_clean: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    config, manifest = _load_prepared_run(run_dir, require_clean=require_clean, expected_runner=RUNNER)
    if not _uses_batches(config):
        return (config, manifest)
    if run_dir.resolve().parent != (ROOT / "artifacts/runs").resolve():
        raise ValueError("Run directory must be artifacts/runs/<run_id>")
    if config["runner"] != RUNNER or run_dir.name != manifest["run_id"]:
        raise ValueError("Prepared run does not match the batch runner or run directory")
    if manifest["identity"]["git_worktree"]["dirty"] or manifest["observed_repository_state"]["dirty"]:
        raise ValueError("Batch provenance requires a clean committed execution worktree")
    _validate_execution(config)
    _backend(config)
    return (config, manifest)


def _validate_estimates(
    table: pd.DataFrame, config: Mapping[str, Any], split_ids: Sequence[int]
) -> pd.DataFrame:
    return _backend(config).validate_estimates(table, config, split_ids)


def _read_result(
    path: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], split_ids: Sequence[int]
) -> pd.DataFrame:
    table = pd.read_parquet(path)
    if table.attrs.get("run_fingerprint") != manifest["fingerprint"]:
        raise ValueError(f"Result belongs to a different run: {path}")
    result = _validate_estimates(table, config, split_ids)
    if _has_reppi(config):
        _backend(config).reppi_diagnostics(result)
    return result


def _has_reppi(config: Mapping[str, Any]) -> bool:
    return (
        _uses_batches(config)
        and config["execution"]["module"] == MODULES[1]
        and any((method["name"] == "reppi" for method in config["methods"]))
    )


def _case_export_tables(
    table: pd.DataFrame, config: Mapping[str, Any], run_dir: Path
) -> Iterator[tuple[Path, pd.DataFrame]]:
    if not config.get("export", {}).get("case_estimates", False):
        return
    if "crossfit_comparison" in config:
        for folds in config["crossfit_comparison"]["outer_folds"]:
            selected = table.loc[table["outer_folds"] == folds].reset_index(drop=True)
            selected.attrs["reppi_zero_coefficient_counts"] = [
                row for row in table.attrs["reppi_zero_coefficient_counts"] if row["outer_folds"] == folds
            ]
            branch = {key: value for key, value in config.items() if key != "crossfit_comparison"}
            for path, part in _case_export_tables(selected, branch, run_dir):
                yield (path.parent / f"folds_{folds}" / path.name, part)
        return
    for case in config["cases"]:
        selected = table.loc[table["case"] == case["name"]].reset_index(drop=True)
        if "reppi_zero_coefficient_counts" in selected.attrs:
            selected.attrs["reppi_zero_coefficient_counts"] = [
                row for row in table.attrs["reppi_zero_coefficient_counts"] if row["case"] == case["name"]
            ]
        yield (run_dir / "report" / case["name"] / "estimates.parquet", selected)


def _verify_case_export(path: Path, expected: pd.DataFrame) -> None:
    existing = pd.read_parquet(path)
    if existing.attrs.get("run_fingerprint") != expected.attrs["run_fingerprint"]:
        raise ValueError(f"Case export belongs to a different run: {path}")
    pd.testing.assert_frame_equal(existing, expected, check_exact=True)


def _export_run_tables(
    table: pd.DataFrame, config: Mapping[str, Any], run_dir: Path, *, outer_folds: int | None = None
) -> None:
    backend = _backend(config)
    if _has_reppi(config):
        diagnostics = backend.reppi_diagnostics(table)
        diagnostics.attrs["run_fingerprint"] = table.attrs["run_fingerprint"]
        output_dir = run_dir if outer_folds is None else run_dir / f"folds_{outer_folds}"
        path = output_dir / "diagnostics.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            if existing.attrs["run_fingerprint"] != table.attrs["run_fingerprint"]:
                raise ValueError("RePPI diagnostics belong to a different run")
            pd.testing.assert_frame_equal(existing, diagnostics)
        else:
            _write_table(diagnostics, path)
    for path, selected in _case_export_tables(table, config, run_dir):
        if path.exists():
            _verify_case_export(path, selected)
        else:
            _write_table(selected, path)
    if "target_mean_csv" in config.get("export", {}):
        from evaluation.summarize_transport_paired import export_target_means

        export_target_means(table, config)


def _completed_result(
    run_dir: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], *, outer_folds: int | None = None
) -> Path:
    output_dir = run_dir if outer_folds is None else run_dir / f"folds_{outer_folds}"
    if (output_dir / "_SUCCESS").read_text(encoding="utf-8").strip() != manifest["fingerprint"]:
        raise ValueError("Success marker differs from the run fingerprint")
    path = _result_path(config, output_dir)
    table = _read_result(path, config, manifest, config["split"]["ids"])
    if _has_reppi(config):
        diagnostics = pd.read_parquet(output_dir / "diagnostics.parquet")
        if diagnostics.attrs["run_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("RePPI diagnostics belong to a different run")
        pd.testing.assert_frame_equal(diagnostics, _backend(config).reppi_diagnostics(table))
    for export_path, selected in _case_export_tables(table, config, run_dir):
        _verify_case_export(export_path, selected)
    return path


def prepare_run(config_path: Path) -> Path:
    """Return a prepared run directory with committed provenance and resolved settings."""
    state = git_state(PROJECT_ROOT)
    if state["dirty"] or state["commit"] is None:
        raise RuntimeError("Batch preparation requires a clean, committed execution worktree")
    config = load_transport_config(config_path)
    data_files = {
        **{key: Path(value) for key, value in config["paths"].items()},
        "embedding": Path(config["representations"][0]["path"]),
    }
    if "config_path" in config["density_ratio"]:
        data_files["kernel_config"] = Path(config["density_ratio"]["config_path"])
    runtime_packages = ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "PyYAML", "adapt", "cvxopt")
    if config["regression"]["name"] == "tabpfn":
        data_files["tabpfn_checkpoint"] = Path(config["regression"]["checkpoint"])
        runtime_packages += ("tabpfn", "torch")
    provenance = build_run_provenance(
        ExperimentConfig.from_mapping(config),
        PROJECT_ROOT,
        data_files=data_files,
        code_paths=(
            Path("HealthLLM_transfer/code"),
            Path("HealthLLM_transfer/configs"),
            Path("src/meta_eval"),
            Path("pyproject.toml"),
        ),
        dependency_versions={name: version(name) for name in runtime_packages},
    )
    identity = provenance["identity"]
    current = git_state(PROJECT_ROOT)
    if (
        current["commit"] != state["commit"]
        or current["dirty"]
        or identity["git_commit"] != state["commit"]
        or identity["git_worktree"]["dirty"]
        or provenance["observed_repository_state"]["dirty"]
    ):
        raise RuntimeError("Execution worktree changed while preparing batch provenance")
    run_dir = ROOT / config["artifact_root"] / provenance["run_id"]
    if run_dir.exists():
        previous_config, previous = _load_run(run_dir)
        if previous["fingerprint"] != provenance["fingerprint"] or previous_config != config:
            raise ValueError("Existing run differs from the prepared configuration or fingerprint")
        if (run_dir / "_SUCCESS").exists():
            _completed_result(run_dir, config, previous)
        return run_dir
    run_dir.mkdir(parents=True)
    (run_dir / "parts").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return run_dir


def run_batch(run_dir: Path, batch_id: int, *, split_id: int | None = None) -> list[Path]:
    """Return result paths for a batch or one selected split, reusing verified results."""
    config, manifest = _load_run(run_dir)
    group_name, split_ids = select_batch(config, batch_id)
    if split_id is not None:
        if split_id not in split_ids:
            raise ValueError("Requested split ID is outside the selected batch")
        split_ids = [split_id]
    if (run_dir / "_SUCCESS").exists():
        return [_completed_result(run_dir, config, manifest)]
    selected = _execution_groups(config)[group_name]
    return _run_splits(run_dir, selected, manifest, split_ids, group_name=group_name)


def _run_splits(
    run_dir: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split_ids: Sequence[int],
    *,
    group_name: str | None = None,
) -> list[Path]:
    backend = _backend(config)
    parts_dir = run_dir / "parts"
    if group_name is not None:
        parts_dir /= group_name
    parts_dir.mkdir(parents=True, exist_ok=True)
    factory = None
    input_cache: TransportInputCache = {}
    paths = []
    for split_id in split_ids:
        path = parts_dir / f"split_{split_id:04d}.parquet"
        if path.exists():
            _read_result(path, config, manifest, [split_id])
        else:
            if factory is None:
                factory = backend._regressor_factory(config)
            table = backend.calculate_estimates(
                config, split_ids=[split_id], regressor_factory=factory, input_cache=input_cache
            )
            table.attrs["run_fingerprint"] = manifest["fingerprint"]
            _write_table(table, path)
        paths.append(path)
    return paths


def finalize_run(run_dir: Path, *, require_clean: bool = True, outer_folds: int | None = None) -> Path:
    """Return the merged result after verifying every configured split."""
    config, manifest = _load_run(run_dir, require_clean=require_clean)
    if outer_folds is not None:
        config = crossfit_config(config, outer_folds)
    output_dir = run_dir if outer_folds is None else run_dir / f"folds_{outer_folds}"
    if (output_dir / "_SUCCESS").exists():
        return _completed_result(run_dir, config, manifest, outer_folds=outer_folds)
    path = _result_path(config, output_dir)
    if path.exists():
        result = _read_result(path, config, manifest, config["split"]["ids"])
        _export_run_tables(result, config, run_dir, outer_folds=outer_folds)
        with (output_dir / "_SUCCESS").open("x", encoding="utf-8") as handle:
            handle.write(manifest["fingerprint"] + "\n")
        return path
    parts = []
    for name, selected in _execution_groups(config).items():
        parts_dir = run_dir / "parts"
        if name is not None:
            parts_dir /= name
        for split_id in config["split"]["ids"]:
            part_path = parts_dir / f"split_{split_id:04d}.parquet"
            parts.append(_read_result(part_path, selected, manifest, [split_id]))
    combined = pd.concat(parts, ignore_index=True)
    if _has_reppi(config):
        combined.attrs["reppi_zero_coefficient_counts"] = [
            row for part in parts for row in part.attrs["reppi_zero_coefficient_counts"]
        ]
    result = _validate_estimates(combined, config, config["split"]["ids"])
    result.attrs["run_fingerprint"] = manifest["fingerprint"]
    _write_table(result, path)
    _export_run_tables(result, config, run_dir, outer_folds=outer_folds)
    with (output_dir / "_SUCCESS").open("x", encoding="utf-8") as handle:
        handle.write(manifest["fingerprint"] + "\n")
    return path


def run(config_path: Path) -> Path:
    """Prepare and execute all configured splits sequentially, reusing verified results."""
    run_dir = prepare_run(config_path)
    config, manifest = _load_run(run_dir)
    if (run_dir / "_SUCCESS").exists():
        _completed_result(run_dir, config, manifest)
        return run_dir
    for name, selected in _execution_groups(config).items():
        _run_splits(run_dir, selected, manifest, config["split"]["ids"], group_name=name)
    finalize_run(run_dir)
    return run_dir


def main(*, default_config: Path | None = None, expected_module: str | None = None) -> None:
    """Prepare, execute, or finalize batches selected through YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=("run", "prepare", "split", "batch", "finalize", "array")
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--batch-id", type=int)
    parser.add_argument("--split-id", type=int)
    parser.add_argument("--outer-folds", type=int)
    args = parser.parse_args()
    if args.outer_folds is not None and args.mode not in {"finalize", "array"}:
        parser.error("--outer-folds applies to finalize and array modes")
    if args.mode in {"run", "prepare"}:
        if args.config is None:
            parser.error("--config is required for run and prepare modes")
    elif args.run_dir is None:
        parser.error("--run-dir is required for split, batch, finalize, and array modes")
    if args.mode == "batch" and args.batch_id is None:
        parser.error("--batch-id is required for batch mode")
    if args.mode == "split" and args.split_id is None:
        parser.error("--split-id is required for split mode")
    if expected_module is not None:
        config_path = (
            args.config if args.mode in {"run", "prepare"} else args.run_dir / "config.resolved.yaml"
        )
        if load_config(config_path)["execution"]["module"] != expected_module:
            parser.error(f"execution.module must be {expected_module}")
    if args.mode == "run":
        print(run(args.config), flush=True)
        return
    if args.mode == "prepare":
        print(prepare_run(args.config), flush=True)
        return
    if args.mode == "split":
        print(run_split(args.run_dir, args.split_id), flush=True)
    elif args.mode == "batch":
        for path in run_batch(args.run_dir, args.batch_id, split_id=args.split_id):
            print(path, flush=True)
    elif args.mode == "finalize":
        print(finalize_run(args.run_dir, outer_folds=args.outer_folds), flush=True)
    else:
        config, _ = _load_run(args.run_dir)
        print(array_spec(config, outer_folds=args.outer_folds), flush=True)


if __name__ == "__main__":
    main()
