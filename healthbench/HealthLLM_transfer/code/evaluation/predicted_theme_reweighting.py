"""Execute predicted-theme reweighting over repeated splits."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = HEALTHLLM_TRANSFER_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from covariateshift.estimators import make_tabpfn_classifier_factory
from covariateshift.predicted_theme import (
    METHOD_NAMES,
)
from covariateshift.predicted_theme import (
    fit_predicted_theme_weights as fit_one_split,
)
from evaluation.reweighting_execution import (
    SplitWeights,
    batch_bounds,
    configured_batch_count,
    feature_specs,
    load_batches,
    load_common_inputs,
    load_config,
    load_feature,
    load_themes,
    named,
    resolve_path,
    run_repeated_splits,
    save_result_pair,
    task_mapping,
)
from evaluation.reweighting_execution import (
    task_count as repeated_task_count,
)

CONFIG_PATH = HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_predicted_theme_reweighting.yaml"
METHODS = np.asarray(METHOD_NAMES)
FINAL_SCORE_FIELDS = frozenset(
    {
        "method",
        "feature_set",
        "expected_dimension",
        "split_id",
        "seed",
        "model",
        "n_source_complete",
        "reweighted_scores",
    }
)
FINAL_WEIGHT_FIELDS = frozenset(
    {
        "method",
        "feature_set",
        "expected_dimension",
        "theme",
        "split_id",
        "seed",
        "raw_weights",
        "weights",
        "kmm_sigma",
        "selected_gamma_multiplier",
        "selected_gamma",
    }
)


def task_count(config: Mapping[str, Any]) -> int:
    """Return the YAML-defined feature-by-batch task count."""
    return repeated_task_count(config)


def output_methods(config: Mapping[str, Any]) -> np.ndarray:
    """Return the configured predicted-theme output methods."""
    methods = np.asarray(config["output_methods"]).astype(str)
    if tuple(methods) != METHOD_NAMES:
        raise ValueError(f"Predicted-theme methods must be ordered as {METHOD_NAMES}")
    return methods


def theme_labels(config: Mapping[str, Any]) -> np.ndarray:
    """Return the distinct YAML-defined theme labels."""
    labels = np.asarray(config["theme_labels"]).astype(str)
    if labels.ndim != 1 or len(labels) != len(np.unique(labels)):
        raise ValueError("Theme labels must be a unique one-dimensional sequence")
    return labels


def n_batches(config: Mapping[str, Any]) -> int:
    """Return the number of split batches for one feature set."""
    return configured_batch_count(config)


def checkpoint_paths(
    config: Mapping[str, Any], case: str, feature_set: str, batch_id: int
) -> tuple[Path, Path]:
    """Return score and weight checkpoint paths for one batch."""
    start, stop = batch_bounds(config, batch_id)
    directory = resolve_path(config["paths"]["checkpoint_dir"]) / case / "batches" / feature_set
    stem = f"splits_{start + 1:04d}_{stop:04d}"
    return directory / f"{stem}_scores.npz", directory / f"{stem}_weights.npz"


def smoke_paths(config: Mapping[str, Any], case: str, feature_set: str) -> tuple[Path, Path]:
    """Return score and weight paths for one smoke run."""
    directory = resolve_path(config["paths"]["checkpoint_dir"]) / case / "smoke" / feature_set
    return directory / "scores.npz", directory / "weights.npz"


def final_paths(config: Mapping[str, Any], case: str) -> tuple[Path, Path]:
    """Return final score and weight paths for one case."""
    directory = resolve_path(config["paths"]["final_output_dir"]) / case
    return (
        directory / str(config["paths"]["score_filename"]),
        directory / str(config["paths"]["weight_filename"]),
    )


def build_result_arrays(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    split_indices: Sequence[int],
    classifier_factory: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return score and weight checkpoints for selected splits."""
    feature_spec = named(config, "feature_sets", feature_set)
    kmm_spec = named(config, "methods", "predicted_probability_kmm")
    prompt_id, seeds, target_masks, models, scores = load_common_inputs(config, case)
    themes = load_themes(config, prompt_id)
    labels = theme_labels(config)
    features = load_feature(config, feature_set, prompt_id)

    def fit_method(
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_indices: np.ndarray,
        seed: int,
    ) -> SplitWeights:
        result = fit_one_split(
            source_features,
            target_features,
            themes[source_indices],
            labels,
            seed=seed,
            classifier_factory=classifier_factory,
            n_folds=int(config["cross_fitting"]["n_folds"]),
            median_max_points=int(config["execution"]["median_max_points"]),
            kmm_spec=kmm_spec,
        )
        return SplitWeights(
            raw_weights=result.raw_weights,
            weights=result.weights,
            diagnostics={
                "kmm_sigma": result.kmm_sigma,
                "selected_gamma_multiplier": result.selected_gamma_multiplier,
                "selected_gamma": result.selected_gamma,
            },
        )

    batch = run_repeated_splits(
        seeds=seeds,
        target_masks=target_masks,
        scores=scores,
        features=features,
        split_indices=split_indices,
        fit_method=fit_method,
    )
    common = {
        "case": np.asarray(case),
        "method": output_methods(config),
        "feature_set": np.asarray(feature_set),
        "expected_dimension": np.asarray(int(feature_spec["expected_dimension"]), dtype=np.int64),
        "split_id": batch.split_indices + 1,
        "seed": seeds[batch.split_indices],
    }
    return (
        {
            **common,
            "model": models,
            "n_source_complete": batch.n_source_complete,
            "reweighted_scores": batch.reweighted_scores,
        },
        {
            **common,
            "theme": labels,
            "raw_weights": batch.raw_weights,
            "weights": batch.weights,
            **batch.diagnostics,
        },
    )


def _run_result_pair(
    score_path: Path,
    weight_path: Path,
    build_arrays: Any,
) -> tuple[Path, Path]:
    if score_path.exists() and weight_path.exists():
        print(f"EXISTS {score_path} {weight_path}")
        return score_path, weight_path
    if score_path.exists() or weight_path.exists():
        raise ValueError(f"Incomplete score/weight pair: {score_path} {weight_path}")
    score_arrays, weight_arrays = build_arrays()
    save_result_pair(
        weight_path,
        score_path,
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )
    return score_path, weight_path


def run_batch(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    batch_id: int,
    classifier_factory: Any,
) -> tuple[Path, Path]:
    """Fit and save one case-feature batch."""
    score_path, weight_path = checkpoint_paths(config, case, feature_set, batch_id)
    start, stop = batch_bounds(config, batch_id)
    return _run_result_pair(
        score_path,
        weight_path,
        lambda: build_result_arrays(
            config,
            case=case,
            feature_set=feature_set,
            split_indices=range(start, stop),
            classifier_factory=classifier_factory,
        ),
    )


def run_smoke(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    classifier_factory: Any,
) -> tuple[Path, Path]:
    """Fit and save the configured smoke split."""
    score_path, weight_path = smoke_paths(config, case, feature_set)
    split_index = int(config["execution"]["smoke_split_id"]) - 1
    return _run_result_pair(
        score_path,
        weight_path,
        lambda: build_result_arrays(
            config,
            case=case,
            feature_set=feature_set,
            split_indices=(split_index,),
            classifier_factory=classifier_factory,
        ),
    )


def load_feature_batches(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    **_expected: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return one feature set assembled from all checkpoint batches."""
    score_paths = []
    weight_paths = []
    for batch_id in range(n_batches(config)):
        score_path, weight_path = checkpoint_paths(config, case, feature_set, batch_id)
        score_paths.append(score_path)
        weight_paths.append(weight_path)
    constants = ("case", "method", "feature_set", "expected_dimension")
    shared = ("split_id", "seed")
    scores = load_batches(
        score_paths,
        constant_fields=constants + ("model",),
        concatenated_fields=shared + ("n_source_complete", "reweighted_scores"),
    )
    weights = load_batches(
        weight_paths,
        constant_fields=constants + ("theme",),
        concatenated_fields=shared
        + (
            "raw_weights",
            "weights",
            "kmm_sigma",
            "selected_gamma_multiplier",
            "selected_gamma",
        ),
    )
    return scores, weights


def finalize_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Create final case-level score and compressed-weight files."""
    _, seeds, _, models, _ = load_common_inputs(config, case)
    seeds = seeds[: int(config["execution"]["n_splits"])]
    feature_names = []
    dimensions = []
    score_rows = []
    raw_rows = []
    weight_rows = []
    diagnostic_rows = {
        name: []
        for name in (
            "kmm_sigma",
            "selected_gamma_multiplier",
            "selected_gamma",
        )
    }
    complete = None
    for specification in feature_specs(config):
        feature_set = str(specification["name"])
        score_part, weight_part = load_feature_batches(
            config,
            case=case,
            feature_set=feature_set,
        )
        feature_names.append(feature_set)
        dimensions.append(int(specification["expected_dimension"]))
        score_rows.append(score_part["reweighted_scores"])
        raw_rows.append(weight_part["raw_weights"])
        weight_rows.append(weight_part["weights"])
        for name, values in diagnostic_rows.items():
            values.append(weight_part[name])
        if complete is None:
            complete = score_part["n_source_complete"]
        else:
            assert np.array_equal(complete, score_part["n_source_complete"])

    common = {
        "method": output_methods(config),
        "feature_set": np.asarray(feature_names),
        "expected_dimension": np.asarray(dimensions, dtype=np.int64),
        "split_id": np.arange(1, len(seeds) + 1, dtype=np.int64),
        "seed": seeds,
    }
    score_arrays = {
        **common,
        "model": models,
        "n_source_complete": complete,
        "reweighted_scores": np.stack(score_rows).transpose(0, 2, 1, 3),
    }
    weight_arrays = {
        **common,
        "theme": theme_labels(config),
        "raw_weights": np.stack(raw_rows).transpose(0, 2, 1, 3),
        "weights": np.stack(weight_rows).transpose(0, 2, 1, 3),
        **{name: np.stack(values) for name, values in diagnostic_rows.items()},
    }
    if set(score_arrays) != FINAL_SCORE_FIELDS:
        raise ValueError("Final score fields do not match the output contract")
    if set(weight_arrays) != FINAL_WEIGHT_FIELDS:
        raise ValueError("Final weight fields do not match the output contract")
    score_path, weight_path = final_paths(config, case)
    saved_weight, saved_score = save_result_pair(
        weight_path,
        score_path,
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )
    return saved_score, saved_weight


def verify_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Delegate final-output verification to the independent validator."""
    from evaluation.validate_reweighting_outputs import verify_predicted_theme_case

    return verify_predicted_theme_case(config, case=case)


def validate_inputs(config: Mapping[str, Any]) -> None:
    """Validate YAML-defined cases, features, and aligned inputs."""
    output_methods(config)
    labels = theme_labels(config)
    specifications = feature_specs(config)
    n_splits = int(config["execution"]["n_splits"])
    for case_spec in config["cases"]:
        case = str(case_spec["name"])
        prompt_id, seeds, masks, models, scores = load_common_inputs(config, case)
        themes = load_themes(config, prompt_id)
        if not set(np.unique(themes)).issubset(set(labels)):
            raise ValueError(f"Theme labels differ for {case}")
        if seeds.shape != (n_splits,) or masks.shape != (n_splits, len(prompt_id)):
            raise ValueError(f"Split arrays differ for {case}")
        if scores.shape != (len(prompt_id), len(models)):
            raise ValueError(f"Score matrix differs for {case}")
        source_counts = np.unique(np.sum(~masks, axis=1))
        if source_counts.size != 1:
            raise ValueError(f"Source count varies across splits for {case}")
        for specification in specifications:
            values = load_feature(config, str(specification["name"]), prompt_id)
            assert values.shape[1] == int(specification["expected_dimension"])
        print(
            f"VALID {case} source={int(source_counts[0])} "
            f"target={len(prompt_id) - int(source_counts[0])} "
            f"features={len(specifications)} methods={len(METHOD_NAMES)} "
            f"splits={n_splits} models={len(models)}",
            flush=True,
        )


def classifier_factory(config: Mapping[str, Any]) -> Any:
    """Return the configured local TabPFN classifier factory."""
    method = named(config, "methods", "tabpfn_theme_classifier")
    checkpoint = resolve_path(method["checkpoint_path"])
    return make_tabpfn_classifier_factory(method, checkpoint)


def build_parser() -> argparse.ArgumentParser:
    """Return the lifecycle command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("validate", "smoke", "batch", "finalize", "verify"),
    )
    parser.add_argument("--case")
    parser.add_argument("--feature-set")
    parser.add_argument("--task-id", type=int)
    return parser


def main() -> None:
    """Run the selected predicted-theme lifecycle stage."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        validate_inputs(config)
        return
    if args.case is None:
        parser.error("--case is required outside validate mode")
    if args.mode == "smoke":
        if args.feature_set is None:
            parser.error("smoke mode requires --feature-set")
        run_smoke(
            config,
            case=args.case,
            feature_set=args.feature_set,
            classifier_factory=classifier_factory(config),
        )
        return
    if args.mode == "batch":
        if args.task_id is None:
            parser.error("batch mode requires --task-id")
        feature_set, batch_id = task_mapping(config, args.task_id)
        run_batch(
            config,
            case=args.case,
            feature_set=feature_set,
            batch_id=batch_id,
            classifier_factory=classifier_factory(config),
        )
        return
    if args.mode == "finalize":
        finalize_case(config, case=args.case)
        return
    verify_case(config, case=args.case)


if __name__ == "__main__":
    main()
