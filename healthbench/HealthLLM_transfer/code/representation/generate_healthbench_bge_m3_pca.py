"""Create 30%, 50%, 80%, and 95% variance PCA features from BGE-M3 embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


HEALTHLLM_TRANSFER_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/embedding/healthbench_bge_m3_embeddings.npz"
OUTPUT_DIR = HEALTHLLM_TRANSFER_ROOT / "data/healthbench/embedding"
VARIANCE_LEVELS = (0.30, 0.50, 0.80, 0.95)


def fit_pca(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCA scores and their individual and cumulative variance ratios."""

    pca = PCA(svd_solver="full")
    scores = pca.fit_transform(embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    return scores, explained_variance_ratio, cumulative_variance


def n_components_for_variance(
    cumulative_variance: np.ndarray,
    threshold: float,
) -> int:
    """Return the fewest principal components meeting a variance threshold."""

    return int(np.searchsorted(cumulative_variance, threshold) + 1)


def main() -> None:
    """Generate separate prompt and rubric PCA representations."""

    with np.load(INPUT_PATH, allow_pickle=False) as data:
        prompt_embeddings = data["prompt_embeddings"].astype(np.float32)
        rubric_embeddings = data["rubric_embeddings"].astype(np.float32)

        # Alignment and rubric-information fields must remain paired with each row.
        metadata = {
            key: data[key] for key in data.files if key not in {"prompt_embeddings", "rubric_embeddings"}
        }

    if prompt_embeddings.shape[0] != rubric_embeddings.shape[0]:
        raise ValueError("Prompt and rubric embeddings have different row counts")

    print(f"Cases: {prompt_embeddings.shape[0]}")
    print(f"Prompt embeddings: {prompt_embeddings.shape}")
    print(f"Rubric embeddings: {rubric_embeddings.shape}")

    # Fitting each block once ensures all thresholds use the same PCA basis.
    (
        prompt_scores,
        prompt_explained_variance_ratio,
        prompt_cumulative_variance,
    ) = fit_pca(prompt_embeddings)
    (
        rubric_scores,
        rubric_explained_variance_ratio,
        rubric_cumulative_variance,
    ) = fit_pca(rubric_embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for threshold in VARIANCE_LEVELS:
        percent = round(threshold * 100)
        prompt_n_components = n_components_for_variance(
            prompt_cumulative_variance,
            threshold,
        )
        rubric_n_components = n_components_for_variance(
            rubric_cumulative_variance,
            threshold,
        )
        output_path = OUTPUT_DIR / f"healthbench_bge_m3_pca{percent}_embeddings.npz"

        np.savez_compressed(
            output_path,
            **metadata,
            prompt_pcs=prompt_scores[:, :prompt_n_components].astype(np.float32),
            rubric_pcs=rubric_scores[:, :rubric_n_components].astype(np.float32),
            variance_threshold=np.float32(threshold),
            prompt_n_components=np.int64(prompt_n_components),
            prompt_explained_variance_ratio=prompt_explained_variance_ratio[:prompt_n_components].astype(
                np.float32
            ),
            prompt_cumulative_explained_variance=np.float32(
                prompt_cumulative_variance[prompt_n_components - 1]
            ),
            rubric_n_components=np.int64(rubric_n_components),
            rubric_explained_variance_ratio=rubric_explained_variance_ratio[:rubric_n_components].astype(
                np.float32
            ),
            rubric_cumulative_explained_variance=np.float32(
                rubric_cumulative_variance[rubric_n_components - 1]
            ),
        )

        print(
            f"PCA {percent}% | "
            f"prompt PCs={prompt_n_components} "
            f"(variance={prompt_cumulative_variance[prompt_n_components - 1]:.4f}) | "
            f"rubric PCs={rubric_n_components} "
            f"(variance={rubric_cumulative_variance[rubric_n_components - 1]:.4f})"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
