"""Summarize and export paired target-mean estimates without fitting models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from evaluation import run_transport_batch as base
from evaluation.run_transport_batch import _write_table

ESTIMATE_COLUMNS = [*base.ESTIMATE_COLUMNS, "standard_error"]
SCORE_SET_COLUMNS = ["score_set", *ESTIMATE_COLUMNS]


def export_target_means(table: pd.DataFrame, config: Mapping[str, Any]) -> Path:
    """Return the CSV path containing one row per split, model, and label budget."""
    identifiers = ["case", "split_id", "split_seed", "model", "feature_set", "alpha"]
    methods = [method["name"] for method in config["methods"]]
    wide = table.pivot(index=identifiers, columns="method", values="estimate")
    order = pd.MultiIndex.from_frame(table[identifiers].drop_duplicates())
    wide = wide.reindex(index=order, columns=methods).reset_index()
    wide.columns.name = None
    path = Path(config["export"]["target_mean_csv"])
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), wide, check_dtype=False, atol=1e-12, rtol=1e-12)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(wide, path)
    return path
