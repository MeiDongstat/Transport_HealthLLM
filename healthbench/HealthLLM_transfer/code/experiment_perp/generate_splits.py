"""Generate repeated HealthBench Case 1, Case 2, and Case 3 splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/healthbench_oss_lan.jsonl"
EMBEDDING_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/embedding/healthbench_bge_m3_embeddings.npz"
CASE3_CONFIG_PATH = HEALTHLLM_TRANSFER_ROOT / "configs/healthbench_case3_splits.yaml"

N_SPLITS = 500
START_SEED = 123

TARGET_FRACTIONS = {
    "case1": {
        "communication": 0.30,
        "complex_responses": 0.30,
        "context_seeking": 0.95,
        "emergency_referrals": 0.95,
        "global_health": 0.95,
        "health_data_tasks": 0.95,
        "hedging": 0.30,
    },
    "case2": {
        "communication": 0.50,
        "complex_responses": 0.50,
        "context_seeking": 0.90,
        "emergency_referrals": 0.90,
        "global_health": 0.90,
        "health_data_tasks": 0.90,
        "hedging": 0.50,
    },
}


def extract_theme(row: dict) -> str:
    """Return the single HealthBench theme for one row."""
    themes = [
        tag.removeprefix("theme:")
        for tag in row["example_tags"]
        if isinstance(tag, str) and tag.startswith("theme:")
    ]
    if len(themes) != 1:
        raise ValueError(f"prompt_id={row['prompt_id']} has {len(themes)} theme tags")
    return themes[0]


def load_metadata(
    data_path: Path = DATA_PATH,
    embedding_path: Path = EMBEDDING_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prompt IDs, themes, and languages in embedding row order."""
    with data_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    metadata_prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(set(metadata_prompt_ids)) != len(metadata_prompt_ids):
        raise ValueError("Duplicate prompt_id in HealthBench metadata")

    theme_by_prompt_id = {str(row["prompt_id"]): extract_theme(row) for row in rows}
    language_by_prompt_id = {str(row["prompt_id"]): str(row["language"]).lower() for row in rows}
    with np.load(embedding_path, allow_pickle=False) as embeddings:
        all_prompt_id = embeddings["prompt_id"].astype(str)

    if len(set(all_prompt_id)) != len(all_prompt_id):
        raise ValueError("Duplicate prompt_id in HealthBench embeddings")

    missing_metadata = [prompt_id for prompt_id in all_prompt_id if prompt_id not in theme_by_prompt_id]
    if missing_metadata:
        raise ValueError(f"{len(missing_metadata)} embedding prompt IDs lack theme metadata")

    missing_embeddings = set(theme_by_prompt_id) - set(all_prompt_id)
    if missing_embeddings:
        raise ValueError(f"{len(missing_embeddings)} metadata prompt IDs lack embeddings")

    all_theme = np.asarray(
        [theme_by_prompt_id[prompt_id] for prompt_id in all_prompt_id],
        dtype=str,
    )
    all_language = np.asarray(
        [language_by_prompt_id[prompt_id] for prompt_id in all_prompt_id],
        dtype=str,
    )
    return all_prompt_id, all_theme, all_language


def split_by_theme(
    themes: np.ndarray,
    fractions: dict[str, float],
    seed: int,
) -> np.ndarray:
    """Return a Boolean target mask for one repeated split."""
    observed_themes = set(themes)
    expected_themes = set(fractions)
    if observed_themes != expected_themes:
        raise ValueError(
            "Theme mismatch: "
            f"missing={sorted(expected_themes - observed_themes)}, "
            f"unexpected={sorted(observed_themes - expected_themes)}"
        )

    rng = np.random.default_rng(seed)
    target = np.zeros(len(themes), dtype=bool)
    for theme, target_fraction in fractions.items():
        theme_indices = np.flatnonzero(themes == theme)
        n_target = int(np.floor(len(theme_indices) * target_fraction + 0.5))
        target_indices = rng.choice(theme_indices, size=n_target, replace=False)
        target[target_indices] = True
    return target


def generate_splits(
    themes: np.ndarray,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return repeated target masks for Case 1 and Case 2."""
    case1_target = np.stack([split_by_theme(themes, TARGET_FRACTIONS["case1"], int(seed)) for seed in seeds])
    case2_target = np.stack([split_by_theme(themes, TARGET_FRACTIONS["case2"], int(seed)) for seed in seeds])
    return case1_target, case2_target


def split_by_theme_language(
    themes: np.ndarray,
    languages: np.ndarray,
    fractions: dict[str, dict[str, float]],
    seed: int,
) -> np.ndarray:
    """Return a target mask from fixed theme-by-language quotas."""
    if themes.shape != languages.shape:
        raise ValueError("Theme and language arrays must have the same shape")
    if set(themes) != set(fractions):
        raise ValueError("Configured and observed themes do not match")

    rng = np.random.default_rng(seed)
    target = np.zeros(len(themes), dtype=bool)
    for theme, language_fractions in fractions.items():
        for language, language_mask in (
            ("en", languages == "en"),
            ("non-en", languages != "en"),
        ):
            fraction = language_fractions[language]
            if not 0 <= fraction < 1:
                raise ValueError(f"Invalid target fraction for {theme}/{language}")
            indices = np.flatnonzero((themes == theme) & language_mask)
            n_target = 0
            if fraction > 0:
                if len(indices) < 2:
                    raise ValueError(f"{theme}/{language} needs at least two cases")
                n_target = min(
                    len(indices) - 1,
                    max(1, int(np.floor(len(indices) * fraction + 0.5))),
                )
            target[rng.choice(indices, size=n_target, replace=False)] = True
    return target


def generate_case3_splits(
    themes: np.ndarray,
    languages: np.ndarray,
    fractions: dict[str, dict[str, float]],
    seeds: np.ndarray,
) -> np.ndarray:
    """Return repeated Case 3 target masks in the shared prompt order."""
    return np.stack([split_by_theme_language(themes, languages, fractions, int(seed)) for seed in seeds])


def main() -> None:
    """Generate and save full HealthBench splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CASE3_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=HEALTHLLM_TRANSFER_ROOT / "data")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        case3_config = yaml.safe_load(handle)
    all_prompt_id, all_theme, all_language = load_metadata(
        args.data_root / "healthbench/healthbench_oss_lan.jsonl",
        args.data_root / "healthbench/embedding/healthbench_bge_m3_embeddings.npz",
    )
    seeds = np.arange(START_SEED, START_SEED + N_SPLITS, dtype=np.int64)
    all_case1_target, all_case2_target = generate_splits(all_theme, seeds)
    all_case3_target = generate_case3_splits(all_theme, all_language, case3_config["target_fractions"], seeds)

    print(f"HealthBench sample size: {len(all_theme)}")
    for name, target in (
        ("Case 1", all_case1_target),
        ("Case 2", all_case2_target),
        ("Case 3", all_case3_target),
    ):
        n_target = int(target[0].sum())
        print(f"{name}: source={len(all_theme) - n_target}, target={n_target}")
    print(f"Repeated splits: {len(seeds)}")

    output_path = args.data_root / "healthbench/embedding/healthbench_splits.npz"
    np.savez_compressed(
        output_path,
        all_prompt_id=all_prompt_id,
        all_theme=all_theme,
        seeds=seeds,
        all_case1_target=all_case1_target,
        all_case2_target=all_case2_target,
        all_case3_target=all_case3_target,
    )
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
