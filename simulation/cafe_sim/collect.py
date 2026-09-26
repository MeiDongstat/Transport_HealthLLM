"""Inspect or collect validated fixed-data results without discarding failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import tempfile

from .runner import (ROOT, MODEL, SETTINGS, METHODS, LABEL_COUNTS, _array_hash,
                     atomic_json, digest, make_labels, read_json, seed,
                     validate_part, verify_release)


def _resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _inventory(output):
    counts = {setting: {"complete_units": 0, "complete_budgets": 0,
                         "expected_units": 500, "expected_budgets": 1500}
              for setting in SETTINGS}
    expected = {f"{setting}__case2__{split:04d}": (setting, split)
                for setting in SETTINGS for split in range(1, 501)}
    for directory in sorted((output / "parts").glob("*")):
        if not directory.is_dir():
            if not directory.name.startswith("."):
                raise ValueError("Unexpected file in the result-parts directory")
            continue
        if directory.name not in expected:
            raise ValueError(f"Unexpected result unit: {directory.name}")
        setting, _ = expected[directory.name]
        parts = list(directory.glob("labels_*.json"))
        if any(p.name not in {f"labels_{n}.json" for n in LABEL_COUNTS} for p in parts):
            raise ValueError("Unexpected label budget in results")
        counts[setting]["complete_budgets"] += len(parts)
        counts[setting]["complete_units"] += int(len(parts) == len(LABEL_COUNTS))
    return expected, counts


def status(output="results", *, root=ROOT):
    """Count checkpoints; formal collection performs the numerical validation."""
    output = _resolve(output, root)
    _, counts = _inventory(output)
    units = sum(v["complete_units"] for v in counts.values())
    budgets = sum(v["complete_budgets"] for v in counts.values())
    return {"status": "ready_to_validate" if units == 2000 else "incomplete",
            "complete_units": units, "expected_units": 2000,
            "complete_budgets": budgets, "expected_budgets": 6000,
            "failed_attempts": len(list((output / "failures").glob("*.json"))),
            "settings": counts,
            "note": "Counts show saved checkpoints. Use collect to validate all rows before reporting."}


def validated_rows(output="results", *, root=ROOT):
    """Validate all available rows against the release and exact saved design."""
    import numpy as np
    root, output = Path(root).resolve(), _resolve(output, root)
    release_id = verify_release(root)
    identity = read_json(output / "experiment.json")
    if (identity.get("release_id") != release_id
            or identity.get("fingerprint") != digest({k: v for k, v in identity.items() if k != "fingerprint"})):
        raise ValueError("Results belong to another release or have an altered runtime identity")
    expected, counts = _inventory(output)
    settings = read_json(root / "data/settings.json")
    with np.load(root / "data/reference.npz", allow_pickle=False) as archive:
        target_masks = archive["target_masks"].astype(bool)
        split_seeds = archive["split_seeds"].copy()
    cached = {}
    for setting in SETTINGS:
        with np.load(root / "data/datasets" / setting / "scores.npz", allow_pickle=False) as archive:
            data_hash = _array_hash(*(archive[k] for k in ("B_source", "B_target", "Y_target")))
        with np.load(root / "data/datasets" / setting / "truth.npz", allow_pickle=False) as archive:
            cached[setting] = (data_hash, archive["Y_conditional_mean"].copy(), archive["Y_finite_population"].copy())
    design = {}
    for split in range(1, 501):
        target = target_masks[split - 1]
        case_seed = seed(123, int(split_seeds[split - 1]), 2)
        priorities = np.random.default_rng(seed(case_seed, 0)).random(len(target))[target]
        masks = make_labels(priorities)
        target_rows = np.flatnonzero(target)
        design[split] = {n: target_rows[masks[n]].tolist() for n in LABEL_COUNTS}
    rows = []
    for key, (setting, split) in expected.items():
        scenario = SETTINGS[setting]
        unit = dict(setting_id=setting, scenario=scenario, case="case2", split_id=split)
        data_hash, means, realized = cached[setting]
        target = target_masks[split - 1]
        conditional_truth = float(means[target].mean())
        finite_truth = float(realized[target].mean())
        for count in LABEL_COUNTS:
            path = output / "parts" / key / f"labels_{count}.json"
            if not path.exists():
                continue
            part_rows = validate_part(read_json(path), identity["fingerprint"], unit, count,
                                      design[split][count], data_hash)
            for row in part_rows:
                required = {"model": MODEL, "n_source": int((~target).sum()),
                            "n_target": int(target.sum()), "split_seed": int(split_seeds[split - 1]),
                            "generation_seed": seed(123, 901), "outer_folds": 5,
                            "release_id": release_id}
                if any(row.get(k) != v for k, v in required.items()):
                    raise ValueError("Result population, seeds, model, folds, or release differs")
                if (abs(row["truth"] - conditional_truth) > 1e-12
                        or abs(row["truth_finite_population"] - finite_truth) > 1e-12):
                    raise ValueError("Saved evaluation truth differs from the fixed generator")
                row = dict(row)
                correlation = settings[setting]["correlation"]
                row.update(target_correlation=correlation["target_correlation"],
                           actual_correlation=correlation["actual_correlation"], rho=correlation["rho_used"])
                rows.append(row)
    failures = []
    for path in sorted((output / "failures").glob("*.json")):
        record = read_json(path)
        if record.get("fingerprint") != identity["fingerprint"]:
            raise ValueError("Failure log belongs to a different experiment")
        failures.append({"record": path.name, **record})
    complete = all(v["complete_units"] == 500 and v["complete_budgets"] == 1500 for v in counts.values())
    if complete and len(rows) != 31500:
        raise ValueError("Complete experiment must contain exactly 31500 estimates")
    overview = dict(status="complete" if complete else "incomplete", settings=counts,
                    complete_units=sum(v["complete_units"] for v in counts.values()), expected_units=2000,
                    complete_budgets=sum(v["complete_budgets"] for v in counts.values()), expected_budgets=6000,
                    result_rows=len(rows), expected_result_rows=31500, failed_attempts=len(failures),
                    release_id=release_id, fingerprint=identity["fingerprint"])
    return rows, failures, overview


def _atomic_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".table-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if path.suffix == ".parquet":
            table.to_parquet(temporary, index=False)
        else:
            table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(output="results", *, destination=None, allow_incomplete=False, root=ROOT):
    import pandas as pd
    output = _resolve(output, root)
    destination = _resolve(destination, root) if destination else output / "collected"
    rows, failures, overview = validated_rows(output, root=root)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "failures.json", failures)
    atomic_json(destination / "collection.json", overview)
    if overview["status"] != "complete" and not allow_incomplete:
        raise ValueError(f"Formal collection requires all 2000 units; {overview['complete_units']} are complete. "
                         "Use --allow-incomplete only for inspection.")
    stem = "raw_results" if overview["status"] == "complete" else "raw_results_incomplete"
    table = pd.DataFrame(rows)
    if not table.empty:
        keys = ["setting_id", "scenario", "case", "split_id", "n_target_labels", "method"]
        if table.duplicated(keys).any():
            raise ValueError("Duplicate estimates in validated collection")
        table = table.sort_values(keys).reset_index(drop=True)
    for suffix in (".csv", ".parquet"):
        _atomic_table(table, destination / (stem + suffix))
    return overview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "collect"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        result = status(args.output)
    else:
        result = collect(args.output, destination=args.destination, allow_incomplete=args.allow_incomplete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
