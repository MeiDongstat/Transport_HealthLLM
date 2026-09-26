"""Execute domain-classifier reweighting over repeated splits."""

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

from covariateshift.domain_classifier import (  # noqa: E402
    fit_domain_classifier_weights as fit_one_split,
)
from covariateshift.estimators import make_tabpfn_classifier_factory  # noqa: E402
from evaluation.reweighting_execution import (  # noqa: E402
    SplitWeights,
    batch_bounds,
    configured_batch_count,
    configured_split_count,
    feature_specs,
    load_batches,
    load_common_inputs,
    load_config,
    load_feature,
    named,
    resolve_path,
    run_repeated_splits,
    save_result_pair,
    task_mapping,
)

CONFIG_PATH = HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_domain_classifier.yaml"
OUTPUT_METHODS = np.asarray(("TabPFN",))
batch_task = task_mapping


def method_spec(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sole configured TabPFN domain-classifier method."""
    if len(config["methods"]) != 1:
        raise ValueError("Domain classification requires exactly one method")
    method = config["methods"][0]
    if str(method["name"]) != "tabpfn_domain_classifier":
        raise ValueError("The configured method must be tabpfn_domain_classifier")
    return method


def batch_path(config: Mapping[str, Any], case: str, feature_set: str, batch_id: int) -> Path:
    """Return one case-feature batch checkpoint path."""
    start, stop = batch_bounds(config, batch_id)
    return (
        resolve_path(config["paths"]["batch_output_dir"])
        / case
        / feature_set
        / f"splits_{start + 1:04d}_{stop:04d}.npz"
    )


def smoke_path(config: Mapping[str, Any], case: str, feature_set: str) -> Path:
    """Return one case-feature smoke-result path."""
    split_id = int(config["execution"]["smoke_split_id"])
    return (
        resolve_path(config["paths"]["smoke_output_dir"]) / case / feature_set / f"split_{split_id:04d}.npz"
    )


def final_weights_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the final compressed case-level weight path."""
    return (
        resolve_path(config["paths"]["final_output_dir"])
        / case
        / str(config["paths"]["final_weights_filename"])
    )


def final_scores_path(config: Mapping[str, Any], case: str) -> Path:
    """Return the final uncompressed case-level score path."""
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
) -> dict[str, np.ndarray]:
    """Return weights and scores for selected split indices."""
    feature_spec = named(config, "feature_sets", feature_set)
    prompt_id, seeds, target_masks, models, scores = load_common_inputs(config, case)
    features = load_feature(config, feature_set, prompt_id)
    method = method_spec(config)
    n_folds = int(config["cross_fitting"]["n_folds"])

    def fit_method(
        source_features: np.ndarray,
        target_features: np.ndarray,
        _source_indices: np.ndarray,
        seed: int,
    ) -> SplitWeights:
        result = fit_one_split(
            source_features,
            target_features,
            seed=seed,
            classifier_factory=classifier_factory,
            n_folds=n_folds,
            probability_clip=method["probability_clip"],
        )
        return SplitWeights(
            raw_weights=result.raw_weights,
            weights=result.weights,
            diagnostics={
                "source_probability": result.source_probability,
                "target_probability": result.target_probability,
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
    return {
        "case": np.asarray(case),
        "feature_set": np.asarray(feature_set),
        "expected_dimension": np.asarray(int(feature_spec["expected_dimension"])),
        "method": np.asarray(str(method["name"])),
        "n_folds": np.asarray(n_folds),
        "split_id": batch.split_indices + 1,
        "seed": seeds[batch.split_indices],
        "raw_weights": batch.raw_weights[:, 0],
        "weights": batch.weights[:, 0],
        "source_probability": batch.diagnostics["source_probability"],
        "target_probability": batch.diagnostics["target_probability"],
        "model": models,
        "n_source_complete": batch.n_source_complete,
        "reweighted_scores": batch.reweighted_scores[:, 0],
    }


def _run_checkpoint(
    path: Path,
    build_arrays: Any,
) -> Path:
    if path.exists():
        print(f"EXISTS {path}", flush=True)
        return path
    arrays = build_arrays()
    from evaluation.reweighting_execution import save_batch

    save_batch(path, **arrays)
    print(f"SAVED {path}", flush=True)
    return path


def run_batch(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_set: str,
    batch_id: int,
    classifier_factory: Any,
) -> Path:
    """Fit and save one case-feature TabPFN batch."""
    named(config, "cases", case)
    named(config, "feature_sets", feature_set)
    path = batch_path(config, case, feature_set, batch_id)
    start, stop = batch_bounds(config, batch_id)
    return _run_checkpoint(
        path,
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
) -> Path:
    """Fit and save the configured single-split smoke result."""
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
        ),
    )


def load_feature_batches(
    config: Mapping[str, Any],
    *,
    case: str,
    feature_spec: Mapping[str, Any],
    **_expected: Any,
) -> dict[str, np.ndarray]:
    """Return one feature set assembled from all batch checkpoints."""
    feature_set = str(feature_spec["name"])
    return load_batches(
        [
            batch_path(config, case, feature_set, batch_id)
            for batch_id in range(configured_batch_count(config))
        ],
        constant_fields=(
            "case",
            "feature_set",
            "expected_dimension",
            "method",
            "n_folds",
            "model",
        ),
        concatenated_fields=(
            "split_id",
            "seed",
            "raw_weights",
            "weights",
            "source_probability",
            "target_probability",
            "n_source_complete",
            "reweighted_scores",
        ),
    )


def finalize_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Merge checkpoints into separate case-level weight and score NPZs."""
    named(config, "cases", case)
    _, seeds, _, models, _ = load_common_inputs(config, case)
    n_splits = configured_split_count(config)
    n_folds = int(config["cross_fitting"]["n_folds"])
    specifications = feature_specs(config)
    results = [
        load_feature_batches(config, case=case, feature_spec=specification)
        for specification in specifications
    ]
    common = {
        "method": OUTPUT_METHODS,
        "feature_set": np.asarray([str(item["name"]) for item in specifications]),
        "expected_dimension": np.asarray(
            [int(item["expected_dimension"]) for item in specifications],
            dtype=np.int64,
        ),
        "n_folds": np.asarray(n_folds, dtype=np.int64),
        "split_id": np.arange(1, n_splits + 1, dtype=np.int64),
        "seed": seeds[:n_splits],
    }
    weight_arrays = {
        **common,
        "raw_weights": np.stack([result["raw_weights"] for result in results])[:, np.newaxis],
        "weights": np.stack([result["weights"] for result in results])[:, np.newaxis],
        "source_probability": np.stack([result["source_probability"] for result in results])[:, np.newaxis],
        "target_probability": np.stack([result["target_probability"] for result in results])[:, np.newaxis],
    }
    score_arrays = {
        **common,
        "model": models,
        "n_source_complete": results[0]["n_source_complete"],
        "reweighted_scores": np.stack([result["reweighted_scores"] for result in results])[:, np.newaxis],
    }
    return save_result_pair(
        final_weights_path(config, case),
        final_scores_path(config, case),
        weight_arrays=weight_arrays,
        score_arrays=score_arrays,
    )


def verify_case(config: Mapping[str, Any], *, case: str) -> tuple[Path, Path]:
    """Delegate final-output verification to the independent validator."""
    from evaluation.validate_reweighting_outputs import verify_domain_case

    return verify_domain_case(config, case=case)


def validate_inputs(config: Mapping[str, Any]) -> None:
    """Validate YAML-defined cases, features, and local input arrays."""
    method_spec(config)
    specifications = feature_specs(config)
    n_splits = configured_split_count(config)
    for case_spec in config["cases"]:
        case = str(case_spec["name"])
        prompt_id, seeds, masks, models, scores = load_common_inputs(config, case)
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
            f"features={len(specifications)} splits={n_splits} models={len(models)}",
            flush=True,
        )


def classifier_factory(config: Mapping[str, Any]) -> Any:
    """Return the YAML-configured TabPFN classifier factory."""
    method = method_spec(config)
    checkpoint = Path(os.environ[str(method["checkpoint_env"])])
    return make_tabpfn_classifier_factory(method, checkpoint)


def build_parser() -> argparse.ArgumentParser:
    """Return the domain-classifier command-line parser."""
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
    """Run the selected domain-classifier lifecycle stage."""
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
