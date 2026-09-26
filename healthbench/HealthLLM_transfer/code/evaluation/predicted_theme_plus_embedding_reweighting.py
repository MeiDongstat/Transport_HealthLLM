"""Execute predicted-theme plus embedding reweighting over repeated splits."""

from __future__ import annotations

import argparse
import os
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
from covariateshift.predicted_theme_plus_embedding import (
    fit_predicted_theme_plus_embedding_weights as fit_oof_split,
)
from evaluation.reweighting_execution import (
    SplitWeights,
    batch_bounds,
    configured_batch_count,
    configured_split_count,
    feature_specs,
    load_batches,
    load_common_inputs,
    load_config,
    load_feature,
    load_themes,
    named,
    resolve_path,
    run_repeated_splits,
    save_batch,
    save_result_pair,
    task_mapping,
)

CONFIG_PATH = HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_predicted_theme_plus_embedding_reweighting.yaml"
METHOD_NAMES = ("kmm",)
batch_task = task_mapping


def method_specs(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return kernel methods in their configured output order."""
    methods = tuple(config["methods"])
    names = tuple(str(item["name"]) for item in methods)
    if names != METHOD_NAMES:
        raise ValueError(f"Kernel methods must be ordered as {METHOD_NAMES}; got {names}")
    return methods


def tabpfn_method(config: Mapping[str, Any], checkpoint: Path, device: str | None = None) -> dict[str, Any]:
    """Return the YAML-defined TabPFN theme-classifier settings."""
    method = dict(config["theme_classifier"])
    method["device"] = str(device if device is not None else method["device"])
    method["checkpoint"] = checkpoint
    return method


def checkpoint_path(config: Mapping[str, Any], case: str, feature_set: str, batch_id: int) -> Path:
    """Return one case-feature batch checkpoint path."""
    start, stop = batch_bounds(config, batch_id)
    return (
        resolve_path(config["paths"]["batch_output_dir"])
        / case
        / feature_set
        / f"splits_{start + 1:04d}_{stop:04d}.npz"
    )


def smoke_path(config: Mapping[str, Any], case: str, feature_set: str) -> Path:
    """Return the configured smoke output path."""
    split_id = int(config["execution"]["smoke_split_id"])
    return (
        resolve_path(config["paths"]["smoke_output_dir"]) / case / feature_set / f"split_{split_id:04d}.npz"
    )


def final_weights_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the compressed final weight path for one case."""
    return (
        resolve_path(config["paths"]["final_output_dir"])
        / case
        / str(config["paths"]["final_weights_filename"])
    )


def final_scores_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the uncompressed final score path for one case."""
    return (
        resolve_path(config["paths"]["final_output_dir"])
        / case
        / str(config["paths"]["final_scores_filename"])
    )


def build_result_arrays(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    split_indices: Sequence[int],
    classifier_factory: Any,
    n_folds: int | None = None,
) -> dict[str, np.ndarray]:
    """Return checkpoint arrays for selected split indices."""
    feature_spec = named(config, "feature_sets", feature_set)
    methods = method_specs(config)
    prompt_id, seeds, target_masks, models, scores = load_common_inputs(config, case)
    themes = load_themes(config, prompt_id)
    labels = np.asarray(config["theme_labels"]).astype(str)
    features = np.ascontiguousarray(load_feature(config, feature_set, prompt_id), dtype=np.float32)
    fold_count = int(n_folds) if n_folds is not None else int(config["cross_fitting"]["n_folds"])

    def fit_method(
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_indices: np.ndarray,
        seed: int,
    ) -> SplitWeights:
        raw, normalized, _, sigma = fit_oof_split(
            source_features,
            target_features,
            themes[source_indices],
            labels,
            methods,
            seed=seed,
            classifier_factory=classifier_factory,
            n_folds=fold_count,
            median_max_points=int(config["execution"]["median_max_points"]),
        )
        return SplitWeights(
            raw_weights=raw,
            weights=normalized,
            diagnostics={"sigma": sigma},
        )

    batch = run_repeated_splits(
        seeds=seeds,
        target_masks=target_masks,
        scores=scores,
        features=features,
        split_indices=split_indices,
        fit_method=fit_method,
    )
    expected_dimension = int(feature_spec["expected_dimension"])
    return {
        "case": np.asarray(case),
        "feature_set": np.asarray(feature_set),
        "expected_dimension": np.asarray(expected_dimension, dtype=np.int64),
        "hybrid_dimension": np.asarray(expected_dimension + len(labels), dtype=np.int64),
        "method": np.asarray(METHOD_NAMES),
        "n_folds": np.asarray(fold_count, dtype=np.int64),
        "split_id": batch.split_indices + 1,
        "seed": seeds[batch.split_indices],
        "raw_weights": batch.raw_weights,
        "weights": batch.weights,
        "sigma": batch.diagnostics["sigma"],
        "model": models,
        "n_source_complete": batch.n_source_complete,
        "reweighted_scores": batch.reweighted_scores,
    }


def _run_checkpoint(path: Path, build_arrays: Any) -> Path:
    if path.exists():
        print(f"EXISTS {path}", flush=True)
        return path
    save_batch(path, **build_arrays())
    print(f"SAVED {path}", flush=True)
    return path


def run_batch(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    batch_id: int,
    classifier_factory: Any,
    n_folds: int | None = None,
) -> Path:
    """Fit and save one case-feature split batch."""
    path = checkpoint_path(config, case, feature_set, batch_id)
    start, stop = batch_bounds(config, batch_id)
    return _run_checkpoint(
        path,
        lambda: build_result_arrays(
            config,
            case=case,
            feature_set=feature_set,
            split_indices=range(start, stop),
            classifier_factory=classifier_factory,
            n_folds=n_folds,
        ),
    )


def run_smoke(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    classifier_factory: Any,
    n_folds: int | None = None,
) -> Path:
    """Fit and save the configured smoke split."""
    path = smoke_path(config, case, feature_set)
    split_index = int(config["execution"]["smoke_split_id"]) - 1
    return _run_checkpoint(
        path,
        lambda: build_result_arrays(
            config,
            case=case,
            feature_set=feature_set,
            split_indices=(split_index,),
            classifier_factory=classifier_factory,
            n_folds=n_folds,
        ),
    )


def load_feature_batches(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_spec: Mapping[str, Any],
    **_expected: Any,
) -> dict[str, np.ndarray]:
    """Return one feature set assembled from all checkpoints."""
    feature_set = str(feature_spec["name"])
    return load_batches(
        [
            checkpoint_path(config, case, feature_set, batch_id)
            for batch_id in range(configured_batch_count(config))
        ],
        constant_fields=(
            "case",
            "feature_set",
            "expected_dimension",
            "hybrid_dimension",
            "method",
            "n_folds",
            "model",
        ),
        concatenated_fields=(
            "split_id",
            "seed",
            "raw_weights",
            "weights",
            "sigma",
            "n_source_complete",
            "reweighted_scores",
        ),
    )


def finalize_case(config: Mapping[str, Any], *, case: str, n_folds: int | None = None) -> tuple[Path, Path]:
    """Merge checkpoints into separate final weight and score NPZs."""
    _, seeds, _, models, _ = load_common_inputs(config, case)
    labels = np.asarray(config["theme_labels"]).astype(str)
    n_splits = configured_split_count(config)
    fold_count = int(n_folds) if n_folds is not None else int(config["cross_fitting"]["n_folds"])
    specifications = feature_specs(config)
    results = [
        load_feature_batches(config, case=case, feature_spec=specification)
        for specification in specifications
    ]
    feature_names = np.asarray([str(item["name"]) for item in specifications])
    dimensions = np.asarray([int(item["expected_dimension"]) for item in specifications], dtype=np.int64)
    common = {
        "method": np.asarray(METHOD_NAMES),
        "feature_set": feature_names,
        "expected_dimension": dimensions,
        "hybrid_dimension": dimensions + len(labels),
        "n_folds": np.asarray(fold_count, dtype=np.int64),
        "split_id": np.arange(1, n_splits + 1, dtype=np.int64),
        "seed": seeds[:n_splits],
    }
    weight_arrays = {
        **common,
        "raw_weights": np.stack([result["raw_weights"] for result in results]).transpose(0, 2, 1, 3),
        "weights": np.stack([result["weights"] for result in results]).transpose(0, 2, 1, 3),
    }
    score_arrays = {
        **common,
        "model": models,
        "n_source_complete": results[0]["n_source_complete"],
        "reweighted_scores": np.stack([result["reweighted_scores"] for result in results]).transpose(
            0, 2, 1, 3
        ),
    }
    return save_result_pair(
        final_weights_path(config, case),
        final_scores_path(config, case),
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )


def verify_case(config: Mapping[str, Any], *, case: str, n_folds: int | None = None) -> tuple[Path, Path]:
    """Delegate final-output verification to the independent validator."""
    from evaluation.validate_reweighting_outputs import verify_plus_embedding_case

    return verify_plus_embedding_case(config, case=case, n_folds=n_folds)


def validate_inputs(config: Mapping[str, Any]) -> None:
    """Validate YAML-defined cases, features, methods, and local inputs."""
    methods = method_specs(config)
    specifications = feature_specs(config)
    labels = np.asarray(config["theme_labels"]).astype(str)
    if labels.ndim != 1 or len(labels) != len(np.unique(labels)):
        raise ValueError("Theme labels must be unique")
    n_splits = configured_split_count(config)
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
            f"features={len(specifications)} methods={len(methods)} "
            f"splits={n_splits} models={len(models)}",
            flush=True,
        )


def classifier_factory(config: Mapping[str, Any], checkpoint: Path, device: str | None = None) -> Any:
    """Return the configured local TabPFN classifier factory."""
    method = tabpfn_method(config, checkpoint, device)
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("TABPFN_CLASSIFIER_CHECKPOINT"),
    )
    parser.add_argument("--device")
    return parser


def main() -> None:
    """Run the requested lifecycle stage."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        validate_inputs(config)
        return
    if args.case is None:
        parser.error("--case is required outside validate mode")
    if args.mode in {"smoke", "batch"} and args.checkpoint is None:
        parser.error("--checkpoint or TABPFN_CLASSIFIER_CHECKPOINT is required")
    if args.mode == "smoke":
        if args.feature_set is None:
            parser.error("smoke mode requires --feature-set")
        run_smoke(
            config,
            case=args.case,
            feature_set=args.feature_set,
            classifier_factory=classifier_factory(config, args.checkpoint, args.device),
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
            classifier_factory=classifier_factory(config, args.checkpoint, args.device),
        )
        return
    if args.mode == "finalize":
        finalize_case(config, case=args.case)
        return
    verify_case(config, case=args.case)


if __name__ == "__main__":
    main()
