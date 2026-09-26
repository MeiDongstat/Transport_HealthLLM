"""Generate aligned prompt, rubric, and rubric-information features for HealthBench."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# Shared inputs and embedding outputs live at the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data/healthbench/healthbench_oss_lan.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "embedding/healthbench_bge_m3_embeddings.npz"

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
MAX_SEQUENCE_LENGTH = 8192
DEFAULT_BATCH_SIZE = 4

AXES = (
    "accuracy",
    "communication_quality",
    "completeness",
    "context_awareness",
    "instruction_following",
)
RUBRIC_INFORMATION_COLUMNS = (
    *(f"axis_prop_{axis}" for axis in AXES),
    "log_criterion_count",
    "total_point_value",
    "negative_criterion_prop",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return the non-empty JSON objects stored in a JSONL file."""

    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: every JSONL record must be an object")
    return rows


def prompt_text(row: dict[str, Any]) -> str:
    """Return the patient-side conversation used as prompt representation P."""

    # Only patient turns define P; assistant responses would leak outcome information.
    user_turns = [message["content"].strip() for message in row["prompt"] if message["role"] == "user"]
    text = "\n".join(turn for turn in user_turns if turn)
    if not text:
        raise ValueError(f"No patient text for prompt_id={row['prompt_id']}")
    return text


def rubric_texts(row: dict[str, Any]) -> list[str]:
    """Return the criterion texts used to construct rubric representation R."""

    texts = [item["criterion"].strip() for item in row["rubrics"]]
    if not texts or any(not text for text in texts):
        raise ValueError(f"Invalid rubric criteria for prompt_id={row['prompt_id']}")
    return texts


def build_rubric_information(rows: list[dict[str, Any]]) -> np.ndarray:
    """Return five axis proportions and three rubric-structure features."""

    features = []
    for row in rows:
        axis_counts = {axis: 0 for axis in AXES}
        points = []
        for rubric in row["rubrics"]:
            axis_tags = [tag.removeprefix("axis:") for tag in rubric["tags"] if tag.startswith("axis:")]
            if len(axis_tags) != 1 or axis_tags[0] not in axis_counts:
                raise ValueError(f"Invalid rubric axis for prompt_id={row['prompt_id']}")
            point = float(rubric["points"])
            if not math.isfinite(point):
                raise ValueError(f"Non-finite rubric points for prompt_id={row['prompt_id']}")
            axis_counts[axis_tags[0]] += 1
            points.append(point)

        criterion_count = len(points)
        if criterion_count == 0:
            raise ValueError(f"No rubric criteria for prompt_id={row['prompt_id']}")

        # Negative points remain signed in the total and define the penalty proportion.
        features.append(
            [
                *(axis_counts[axis] / criterion_count for axis in AXES),
                math.log(criterion_count),
                sum(points),
                sum(point < 0 for point in points) / criterion_count,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def load_inputs(
    input_path: Path,
    *,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], list[str], list[list[str]]]:
    """Return HealthBench rows and their prompt and rubric texts."""

    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")

    rows = read_jsonl(input_path)
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"{input_path}: duplicate prompt_id")

    selected_rows = rows if max_records is None else rows[:max_records]
    prompt_texts = [prompt_text(row) for row in selected_rows]
    criteria_by_case = [rubric_texts(row) for row in selected_rows]
    return selected_rows, prompt_texts, criteria_by_case


def load_model(*, device: str, allow_download: bool) -> Any:
    """Return the fixed BGE encoder on the requested device."""

    from sentence_transformers import SentenceTransformer

    model_options: dict[str, Any] = {
        "revision": MODEL_REVISION,
        "local_files_only": not allow_download,
    }
    # Omitting the device lets SentenceTransformers choose CUDA, MPS, or CPU.
    if device != "auto":
        model_options["device"] = device

    model = SentenceTransformer(MODEL_ID, **model_options)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model.eval()
    return model


def encode(model: Any, texts: list[str], *, batch_size: int) -> np.ndarray:
    """Return normalized 1024-dimensional float32 embeddings."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    embeddings = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if embeddings.shape != expected_shape:
        raise ValueError(f"BGE-M3 embeddings shape={embeddings.shape}; expected {expected_shape}")
    return embeddings


def pool_rubrics(
    model: Any,
    criteria_by_case: list[list[str]],
    *,
    batch_size: int,
) -> np.ndarray:
    """Embed criteria separately and return one normalized mean per case."""

    # Separate encoding prevents one long concatenated rubric from being truncated.
    flat_criteria = [criterion for criteria in criteria_by_case for criterion in criteria]
    item_embeddings = encode(model, flat_criteria, batch_size=batch_size)

    pooled_embeddings = []
    start = 0
    for criteria in criteria_by_case:
        stop = start + len(criteria)
        vector = item_embeddings[start:stop].mean(axis=0)

        # Mean pooling gives every criterion equal weight, then restores unit length.
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("A pooled rubric embedding has zero norm")
        pooled_embeddings.append(vector / norm)
        start = stop

    return np.asarray(pooled_embeddings, dtype=np.float32)


def save_embeddings(
    output_path: Path,
    rows: list[dict[str, Any]],
    prompt_embeddings: np.ndarray,
    rubric_embeddings: np.ndarray,
    rubric_information: np.ndarray,
    criteria_by_case: list[list[str]],
    *,
    overwrite: bool,
) -> None:
    """Save embeddings and row-alignment metadata in one compressed NPZ file."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prompt_id=np.asarray([str(row["prompt_id"]) for row in rows]),
        prompt_embeddings=prompt_embeddings,
        rubric_embeddings=rubric_embeddings,
        rubric_information=rubric_information,
        rubric_information_columns=np.asarray(RUBRIC_INFORMATION_COLUMNS),
        criterion_count=np.asarray([len(criteria) for criteria in criteria_by_case], dtype=np.int16),
        model_id=np.asarray(MODEL_ID),
        model_revision=np.asarray(MODEL_REVISION),
    )


def parse_args() -> argparse.Namespace:
    """Return command-line arguments for one embedding run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformers device, such as auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading the fixed model revision when it is not cached.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Encode only the first N aligned cases for a smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate and save aligned HealthBench representation features."""

    args = parse_args()
    rows, prompt_texts, criteria_by_case = load_inputs(
        args.input_path,
        max_records=args.max_records,
    )
    model = load_model(device=args.device, allow_download=args.allow_download)
    prompt_embeddings = encode(model, prompt_texts, batch_size=args.batch_size)
    rubric_embeddings = pool_rubrics(
        model,
        criteria_by_case,
        batch_size=args.batch_size,
    )
    rubric_information = build_rubric_information(rows)
    save_embeddings(
        args.output_path,
        rows,
        prompt_embeddings,
        rubric_embeddings,
        rubric_information,
        criteria_by_case,
        overwrite=args.overwrite,
    )

    print(f"Saved {len(rows)} cases to {args.output_path}")
    print(f"Prompt embeddings: {prompt_embeddings.shape}")
    print(f"Rubric embeddings: {rubric_embeddings.shape}")
    print(f"Rubric information: {rubric_information.shape}")


if __name__ == "__main__":
    main()
