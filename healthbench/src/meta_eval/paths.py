"""Stable project paths used by repository-backed command modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Return a path rooted at the repository checkout."""
    return PROJECT_ROOT.joinpath(*parts)
