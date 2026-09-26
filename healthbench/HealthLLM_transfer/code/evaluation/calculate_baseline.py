"""Calculate unweighted source-score baselines for repeated HealthBench splits."""

import argparse
from pathlib import Path

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/embedding/healthbench_splits.npz"
METADATA_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/healthbench_metadata_gpt4.1.npz"
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
]:
    """Return models, seeds, aligned scores, and the selected target masks."""
    with np.load(metadata_path, allow_pickle=False) as metadata:
        metadata_prompt_id = metadata["prompt_id"].astype(str)
        model = metadata["model"].astype(str)
        final_score = metadata["final_score"].astype(np.float64)

    with np.load(SPLIT_PATH, allow_pickle=False) as splits:
        split_prompt_id = splits["all_prompt_id"].astype(str)
        seeds = splits["seeds"][:500].astype(np.int64)
        target_masks = np.stack([splits[f"all_{case}_target"][:500].astype(bool) for case in cases])

    if len(set(metadata_prompt_id)) != len(metadata_prompt_id):
        raise ValueError("Duplicate prompt_id in HealthBench metadata")
    if len(set(split_prompt_id)) != len(split_prompt_id):
        raise ValueError("Duplicate prompt_id in HealthBench splits")
    if set(metadata_prompt_id) != set(split_prompt_id):
        raise ValueError("Split and score prompt IDs do not match")

    metadata_index = {prompt_id: index for index, prompt_id in enumerate(metadata_prompt_id)}
    aligned_rows = np.asarray(
        [metadata_index[prompt_id] for prompt_id in split_prompt_id],
        dtype=np.int64,
    )
    aligned_final_score = final_score[aligned_rows]
    if target_masks.shape[1] != len(seeds):
        raise ValueError("Split masks and seeds do not have the same length")
    return model, seeds, aligned_final_score, target_masks


def calculate_baseline_case(
    final_score: np.ndarray,
    target_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return complete counts and unweighted source means for one case."""
    complete_rows = []
    score_rows = []
    for target_mask in target_masks:
        source_scores = final_score[~target_mask]
        complete_rows.append(np.isfinite(source_scores).sum(axis=0))
        score_rows.append(np.nanmean(source_scores, axis=0))
    return {
        "n_source_complete": np.stack(complete_rows).astype(np.int64),
        "reweighted_scores": np.stack(score_rows),
    }


def save_baseline_case(
    path: Path,
    *,
    case: str,
    model: np.ndarray,
    seeds: np.ndarray,
    payload: dict[str, np.ndarray],
) -> None:
    """Save one case's baseline scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case=np.asarray(case),
        method=np.asarray("baseline"),
        split_id=np.arange(1, len(seeds) + 1, dtype=np.int64),
        seed=seeds,
        model=model,
        **payload,
    )


def main() -> None:
    """Calculate and save baseline scores for the selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Input metadata NPZ.")
    parser.add_argument("--n-splits", type=int, default=500, help="Number of splits, starting at split 1.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Score directory containing case subdirectories."
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
    model, seeds, final_score, target_masks = load_inputs(cases, metadata_path=args.metadata)
    if not 1 <= args.n_splits <= len(seeds):
        parser.error(f"--n-splits must be between 1 and {len(seeds)}")
    seeds = seeds[:args.n_splits]
    target_masks = target_masks[:, :args.n_splits]

    for case_index, case in enumerate(cases):
        case_target_masks = target_masks[case_index]
        payload = calculate_baseline_case(final_score, case_target_masks)
        output_path = args.output_dir / case / "baseline.npz"
        save_baseline_case(
            output_path,
            case=case,
            model=model,
            seeds=seeds,
            payload=payload,
        )
        source_counts = np.count_nonzero(~case_target_masks, axis=1)
        print(f"{case} source rows per split: {np.unique(source_counts).tolist()}")
        print(f"{case} baseline shape: {payload['reweighted_scores'].shape}")
        print(f"{case} output path: {output_path}")

    print(f"Models: {len(model)}")
    print(f"Repeated splits: {len(seeds)}")


if __name__ == "__main__":
    main()
