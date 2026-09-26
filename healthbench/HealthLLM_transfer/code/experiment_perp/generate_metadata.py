"""Align observed HealthBench judge scores with prompt metadata."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DATA_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/healthbench_oss_lan.jsonl"
EVALUATION_DIR = PROJECT_ROOT / "HealthBench-evaluation"
OUTPUT_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/healthbench_metadata.npz"

SCORE_FILES = (
    "gpt-5.4.jsonl",
    "gemini-3.1-pro.jsonl",
    "gemini-3.1-pro-large-context.jsonl",
    "claude-opus-4.6.jsonl",
    "gemini-2.5-flash.jsonl",
)


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


def load_case_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prompt IDs, themes, and languages in HealthBench row order."""
    with DATA_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    prompt_id = np.asarray([str(row["prompt_id"]) for row in rows], dtype=str)
    if len(set(prompt_id)) != len(prompt_id):
        raise ValueError("Duplicate prompt_id in HealthBench metadata")

    theme = np.asarray([extract_theme(row) for row in rows], dtype=str)
    language = np.asarray([str(row["language"]).lower() for row in rows], dtype=str)
    return prompt_id, theme, language


def load_final_scores(prompt_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return model names and final scores aligned to prompt IDs."""
    expected_prompt_ids = set(prompt_id)
    models = []
    score_columns = []
    for filename in SCORE_FILES:
        path = EVALUATION_DIR / filename
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        case_ids = [str(row["case_id"]) for row in rows]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"Duplicate case_id in {path}")
        if set(case_ids) != expected_prompt_ids:
            raise ValueError(f"Score coverage does not match prompt_id in {path}")

        observed_models = {str(row["model"]) for row in rows}
        if len(observed_models) != 1:
            raise ValueError(f"Expected one model in {path}")
        models.append(observed_models.pop())

        score_by_prompt_id = {}
        for row in rows:
            value = row["score"]
            score = np.nan if value is None else float(value)
            if not np.isnan(score) and not np.isfinite(score):
                raise ValueError(f"Non-finite score in {path}")
            score_by_prompt_id[str(row["case_id"])] = score
        score_columns.append(
            np.asarray(
                [score_by_prompt_id[value] for value in prompt_id],
                dtype=np.float64,
            )
        )

    if len(set(models)) != len(models):
        raise ValueError("Duplicate model names across score files")
    return np.asarray(models, dtype=str), np.column_stack(score_columns)


def _write_metadata(path: Path, metadata: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **metadata)
        # A same-filesystem hard link publishes the complete archive without overwriting.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Save observed scores and metadata aligned by prompt and model identifiers."""
    global DATA_PATH, EVALUATION_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    DATA_PATH, EVALUATION_DIR = args.metadata_jsonl, args.score_dir
    prompt_id, theme, language = load_case_metadata()
    model, final_score = load_final_scores(prompt_id)
    _write_metadata(
        args.output,
        dict(prompt_id=prompt_id, theme=theme, language=language, model=model, final_score=final_score),
    )


if __name__ == "__main__":
    main()
