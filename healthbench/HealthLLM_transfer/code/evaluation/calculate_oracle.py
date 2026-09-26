"""Calculate known-selection oracle scores for repeated HealthBench splits."""

import argparse
import sys
from pathlib import Path

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = HEALTHLLM_TRANSFER_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from covariateshift.oracle import selection_design_raw_weights
from evaluation.reweighting_execution import expected_source_indices, recompute_scores

SPLIT_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/embedding/healthbench_splits.npz"
METADATA_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/healthbench_metadata.npz"
CASES = ("case1", "case2", "case3")


def load_inputs(
    cases: tuple[str, ...] = CASES,
    *,
    metadata_path: Path = METADATA_PATH,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return models, seeds, aligned scores, selected masks, themes, and languages."""
    with np.load(metadata_path, allow_pickle=False) as metadata:
        metadata_prompt_id = metadata["prompt_id"].astype(str)
        model = metadata["model"].astype(str)
        final_score = metadata["final_score"].astype(np.float64)
        language = metadata["language"].astype(str)

    with np.load(SPLIT_PATH, allow_pickle=False) as splits:
        split_prompt_id = splits["all_prompt_id"].astype(str)
        seeds = splits["seeds"][:500].astype(np.int64)
        themes = splits["all_theme"].astype(str)
        target_masks = np.stack([splits[f"all_{case}_target"][:500].astype(bool) for case in cases])

    if len(set(metadata_prompt_id)) != len(metadata_prompt_id):
        raise ValueError("Duplicate prompt_id in HealthBench metadata")
    if set(metadata_prompt_id) != set(split_prompt_id):
        raise ValueError("Split and score prompt IDs do not match")

    metadata_index = {prompt_id: index for index, prompt_id in enumerate(metadata_prompt_id)}
    aligned_rows = np.asarray(
        [metadata_index[prompt_id] for prompt_id in split_prompt_id],
        dtype=np.int64,
    )
    aligned_final_score = final_score[aligned_rows]
    aligned_language = language[aligned_rows]
    return model, seeds, aligned_final_score, target_masks, themes, aligned_language


def calculate_oracle_case(
    final_score: np.ndarray,
    themes: np.ndarray,
    target_masks: np.ndarray,
    target_fraction_by_theme: dict[str, float],
) -> dict[str, np.ndarray]:
    """Return source rows, weights, complete counts, and scores for one case."""
    source_index_rows = []
    raw_weight_rows = []
    weight_rows = []
    complete_rows = []
    score_rows = []
    for target_mask in target_masks:
        source_indices = np.flatnonzero(~target_mask).astype(np.int64)
        source_scores = final_score[source_indices]
        raw_weights = selection_design_raw_weights(
            themes[source_indices],
            target_fraction_by_theme,
        )
        weights = raw_weights / raw_weights.mean()
        observed = np.isfinite(source_scores)
        weighted_score_sum = np.nansum(
            source_scores * weights[:, np.newaxis],
            axis=0,
        )
        observed_weight_sum = np.sum(
            observed * weights[:, np.newaxis],
            axis=0,
        )
        source_index_rows.append(source_indices)
        raw_weight_rows.append(raw_weights)
        weight_rows.append(weights)
        complete_rows.append(observed.sum(axis=0).astype(np.int64))
        score_rows.append(weighted_score_sum / observed_weight_sum)
    return {
        "source_indices": np.stack(source_index_rows),
        "raw_weights": np.stack(raw_weight_rows),
        "weights": np.stack(weight_rows),
        "n_source_complete": np.stack(complete_rows),
        "reweighted_scores": np.stack(score_rows),
    }


def calculate_oracle_scores(
    final_score: np.ndarray,
    themes: np.ndarray,
    target_masks: np.ndarray,
    target_fraction_by_theme: dict[str, float],
) -> np.ndarray:
    """Return complete-case oracle Hájek means for every split and model."""
    return calculate_oracle_case(
        final_score,
        themes,
        target_masks,
        target_fraction_by_theme,
    )["reweighted_scores"]


def save_oracle_case(
    weight_path: Path,
    score_path: Path,
    *,
    case: str,
    model: np.ndarray,
    seeds: np.ndarray,
    payload: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    """Save one case's weights and reweighted scores separately."""
    common = {
        "case": np.asarray(case),
        "method": np.asarray("oracle"),
        "split_id": np.arange(1, len(seeds) + 1, dtype=np.int64),
        "seed": seeds,
    }
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_temporary = weight_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        weight_temporary,
        **common,
        source_indices=payload["source_indices"],
        raw_weights=payload["raw_weights"],
        weights=payload["weights"],
    )
    weight_temporary.replace(weight_path)

    save_oracle_scores(score_path, case=case, model=model, seeds=seeds, payload=payload)
    return weight_path, score_path


def save_oracle_scores(
    path: Path,
    *,
    case: str,
    model: np.ndarray,
    seeds: np.ndarray,
    payload: dict[str, np.ndarray],
) -> None:
    """Save one case's oracle scores and model-specific observed counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    score_temporary = path.with_suffix(".tmp.npz")
    np.savez(
        score_temporary,
        case=np.asarray(case),
        method=np.asarray("oracle"),
        split_id=np.arange(1, len(seeds) + 1, dtype=np.int64),
        seed=seeds,
        model=model,
        n_source_complete=payload["n_source_complete"],
        reweighted_scores=payload["reweighted_scores"],
    )
    score_temporary.replace(path)


def rescore_oracle_case(
    weight_path: Path,
    *,
    case: str,
    seeds: np.ndarray,
    final_score: np.ndarray,
    target_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return oracle scores from saved weights aligned to the requested splits."""
    with np.load(weight_path, allow_pickle=False) as saved:
        if str(saved["case"].item()) != case or str(saved["method"].item()) != "oracle":
            raise ValueError(f"Oracle weight identity differs: {weight_path}")
        if not np.array_equal(saved["seed"][: len(seeds)], seeds) or not np.array_equal(
            saved["split_id"][: len(seeds)], np.arange(1, len(seeds) + 1)
        ):
            raise ValueError(f"Oracle weight splits or seeds differ: {weight_path}")
        source_indices = expected_source_indices(target_masks, len(seeds))
        if not np.array_equal(saved["source_indices"][: len(seeds)], source_indices):
            raise ValueError(f"Oracle source rows differ: {weight_path}")
        weights = saved["weights"][: len(seeds)].astype(np.float64)
    if weights.shape != source_indices.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"Invalid oracle weights: {weight_path}")
    reweighted_scores, complete = recompute_scores(final_score, source_indices, weights)
    return {"reweighted_scores": reweighted_scores[:, 0], "n_source_complete": complete}


def main() -> None:
    """Recompute oracle scores from saved weights for the selected metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Input metadata NPZ.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Score directory containing case subdirectories."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=HEALTHLLM_TRANSFER_ROOT / "weights",
        help="Existing oracle-weight directory containing case subdirectories.",
    )
    parser.add_argument(
        "--case",
        choices=CASES,
        nargs="+",
        default=list(CASES),
        help="Cases to calculate; defaults to all three cases.",
    )
    args = parser.parse_args()
    cases = tuple(args.case)
    model, seeds, final_score, target_masks, _, _ = load_inputs(cases, metadata_path=args.metadata)

    for case_index, case in enumerate(cases):
        weight_path = args.weights_dir / case / "oracle_weights.npz"
        score_path = args.output_dir / case / "oracle_reweighted_scores.npz"
        payload = rescore_oracle_case(
            weight_path,
            case=case,
            seeds=seeds,
            final_score=final_score,
            target_masks=target_masks[case_index],
        )
        save_oracle_scores(
            score_path,
            case=case,
            model=model,
            seeds=seeds,
            payload=payload,
        )
        print(f"{case} oracle score shape: {payload['reweighted_scores'].shape}")
        print(f"{case} weight path: {weight_path}")
        print(f"{case} score path: {score_path}")

    print(f"Models: {len(model)}")
    print(f"Repeated splits: {len(seeds)}")


if __name__ == "__main__":
    main()
