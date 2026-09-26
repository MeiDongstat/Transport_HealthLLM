"""Run one fixed-data Case 2 split with resumable, validated checkpoints.

Legacy estimator symbols B and Y denote the manuscript scores Y and tilde Y,
respectively. Numerical estimator implementations and seeds are unchanged.
Simulation and HealthBench use the same estimator modules. Oracle moments
are read only after an estimate has been returned.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Gemini 3.1 Pro"
LABEL_COUNTS = (200, 300, 500)
SETTINGS = {
    "same_evaluator": "B_equals_Y",
    "corr_040": "B_not_equals_Y",
    "corr_060": "B_not_equals_Y",
    "corr_080": "B_not_equals_Y",
}
METHODS = {
    "B_equals_Y": ("labelled_only", "ppi", "aipw", "pooled_aipw", "pooled_dr", "cafe"),
    "B_not_equals_Y": ("labelled_only", "ppi", "reppi", "aipw", "cafe"),
}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".checkpoint-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_release(root=ROOT):
    """Verify simulation and shared-core checksums relative to the package root."""
    root = Path(root).resolve()
    manifest = read_json(root / "release_manifest.json")
    package_root = root.parent
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("Release manifest has no file checksums")
    for relative, expected in files.items():
        path = (package_root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(package_root):
            raise ValueError("Unsafe release manifest path")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Release integrity check failed: {relative}")
    return digest(manifest)


def environment_identity(threads):
    packages = ("numpy", "scipy", "scikit-learn", "pandas", "adapt", "cvxopt",
                "tensorflow", "keras", "joblib", "threadpoolctl")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "dependencies": versions, "threads": threads}


def seed(*coordinates):
    import numpy as np
    return int(np.random.SeedSequence(list(coordinates)).generate_state(1)[0])


def configure_threads(threads):
    if type(threads) is not int or threads < 1:
        raise ValueError("Threads must be a positive integer")
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@contextlib.contextmanager
def lock(path, *, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise RuntimeError("This split is already running; no duplicate work was started") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _experiment(output, release_id, environment):
    identity = {"release_id": release_id, "environment": environment}
    identity["fingerprint"] = digest(identity)
    with lock(output / ".identity.lock"):
        path = output / "experiment.json"
        if path.exists():
            if read_json(path) != identity:
                raise ValueError("Output belongs to another release or runtime; use a new output directory")
        else:
            atomic_json(path, identity)
    return identity["fingerprint"]


def make_labels(priorities, counts=LABEL_COUNTS):
    import numpy as np
    order = np.argsort(priorities, kind="stable")
    result = {}
    for count in counts:
        if not 2 <= count < len(priorities):
            raise ValueError("Invalid target label count")
        mask = np.zeros(len(priorities), dtype=bool)
        mask[order[:count]] = True
        result[count] = mask
    return result


def _load_estimator(root, scenario):
    """Return the shared HealthBench evaluator, configuration, and regression factory."""
    project = Path(root).resolve().parent / "healthbench"
    for name in ("covariateshift.common_mean_regression", "evaluation.calculate_transport"):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(project.resolve()):
            raise RuntimeError("Estimator modules were imported from another code directory")
    sys.path[:0] = [str(project / "HealthLLM_transfer/code"), str(project / "src")]
    module_name = "evaluation.calculate_transport" if scenario == "B_equals_Y" else "evaluation.calculate_transport_paired"
    module = importlib.import_module(module_name)
    common = importlib.import_module("covariateshift.common_mean_regression")
    config = read_json(Path(root) / "configs/estimators" / f"{scenario}.json")
    if tuple(entry["name"] for entry in config["methods"]) != METHODS[scenario]:
        raise ValueError("Estimator method inventory differs from the experiment")
    if config["models"].index(MODEL) != 1:
        raise ValueError("Original response-model index must remain one for seed compatibility")
    return module.evaluate_sample, config, common.make_regressor_factory(config["regression"])


def _load_inputs(root, setting, split_id):
    import numpy as np
    with np.load(Path(root) / "data/reference.npz", allow_pickle=False) as archive:
        X = archive["X"].copy()
        masks = archive["target_masks"]
        split_seeds = archive["split_seeds"]
        if masks.shape != (500, len(X)) or split_seeds.shape != (500,) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Invalid saved Case 2 partition archive")
        is_target = masks[split_id - 1].astype(bool)
        split_seed = int(split_seeds[split_id - 1])
    with np.load(Path(root) / "data/datasets" / setting / "scores.npz", allow_pickle=False) as archive:
        scores = {name: archive[name].copy() for name in ("B_source", "B_target", "Y_target")}
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Covariates must be a finite matrix")
    for name, values in scores.items():
        if values.shape != (len(X),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid score array: {name}")
    if (int(is_target.sum()), int((~is_target).sum())) != (3561, 1439):
        raise ValueError("Case 2 must contain 1439 source and 3561 target rows")
    if setting == "same_evaluator" and not np.array_equal(scores["B_target"], scores["Y_target"]):
        raise ValueError("Same-evaluator target scores differ")
    return X, is_target, split_seed, scores


def _load_truth(root, setting, is_target):
    import numpy as np
    with np.load(Path(root) / "data/datasets" / setting / "truth.npz", allow_pickle=False) as archive:
        means = archive["Y_conditional_mean"]
        realized = archive["Y_finite_population"]
        if any(v.shape != is_target.shape or not np.isfinite(v).all() for v in (means, realized)):
            raise ValueError("Invalid evaluation truth arrays")
        return float(means[is_target].mean()), float(realized[is_target].mean())


def _array_hash(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, str(array.dtype))).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def validate_part(part, identity, unit, count, label_rows, data_hash):
    if any(part.get(k) != v for k, v in {"fingerprint": identity, "unit": unit,
                                        "data_hash": data_hash, "label_rows": label_rows}.items()):
        raise ValueError("Checkpoint identity, labels, or generated data differ")
    rows = part.get("rows", [])
    if sorted(row.get("method", "") for row in rows) != sorted(METHODS[unit["scenario"]]):
        raise ValueError("Checkpoint has missing or duplicate methods")
    for row in rows:
        if any(row.get(k) != v for k, v in unit.items()) or row.get("fingerprint") != identity:
            raise ValueError("Checkpoint row identity differs")
        if row.get("n_target_labels") != count or row.get("status") != "ok":
            raise ValueError("Checkpoint budget or status differs")
        if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) for k in
               ("estimate", "standard_error", "truth", "error", "truth_conditional_mean", "truth_finite_population")):
            raise ValueError("Checkpoint has a nonfinite estimate, standard error, or truth")
        if row["standard_error"] < 0 or abs(row["error"] - row["estimate"] + row["truth"]) > 1e-12:
            raise ValueError("Checkpoint has an invalid error or standard error")
        if row["truth"] != row["truth_conditional_mean"] or row["truth"] != rows[0]["truth"]:
            raise ValueError("Checkpoint evaluation truth differs")
    return rows


def _safe_error(exc, root, output):
    text = str(exc)
    for path in sorted({str(Path(root).resolve()), str(Path(output).resolve()), str(Path.home())}, key=len, reverse=True):
        text = text.replace(path, "<local-path>")
    return text


def run_unit(setting, split_id, *, output="results", labels=LABEL_COUNTS, threads=8, root=ROOT):
    """Run/resume requested budgets, keeping failed attempts for later audit."""
    if setting not in SETTINGS or type(split_id) is not int or not 1 <= split_id <= 500:
        raise ValueError("Choose a known setting and a split in 1..500")
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels) or not set(labels).issubset(LABEL_COUNTS):
        raise ValueError("Label budgets must be a unique subset of 200,300,500")
    labels = tuple(sorted(labels))
    configure_threads(threads)
    import numpy as np
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    release_id = verify_release(root)
    identity = _experiment(output, release_id, environment_identity(threads))
    scenario = SETTINGS[setting]
    unit = dict(setting_id=setting, scenario=scenario, case="case2", split_id=split_id)
    key = f"{setting}__case2__{split_id:04d}"
    directory = output / "parts" / key
    start = time.monotonic()
    with lock(directory / ".unit.lock", nonblocking=True), threadpool_limits(limits=threads):
        try:
            X, target, split_seed, scores = _load_inputs(root, setting, split_id)
            source = ~target
            case_seed = seed(123, split_seed, 2)
            priorities = np.random.default_rng(seed(case_seed, 0)).random(len(X))[target]
            masks = make_labels(priorities)
            target_rows = np.flatnonzero(target)
            data_hash = _array_hash(scores["B_source"], scores["B_target"], scores["Y_target"])
            completed = set()
            for count in LABEL_COUNTS:
                part_path = directory / f"labels_{count}.json"
                if part_path.exists():
                    validate_part(read_json(part_path), identity, unit, count,
                                  target_rows[masks[count]].tolist(), data_hash)
                    completed.add(count)
            remaining = [count for count in labels if count not in completed]
            if remaining:
                evaluate, config, factory = _load_estimator(root, scenario)
                config = copy.deepcopy(config)
                config["split"]["label_counts"] = remaining
                estimator_seed = seed(case_seed, 1, config["models"].index(MODEL))
                # Scores outside every requested label set are never passed to
                # the estimator. evaluate_sample separately masks each budget.
                visible = masks[max(remaining)]
                target_y = np.where(visible, scores["Y_target"][target], np.nan)
                if scenario == "B_equals_Y":
                    arguments = (X[source], scores["B_source"][source], X[target], target_y, priorities)
                else:
                    target_b = np.where(visible, scores["B_target"][target], np.nan)
                    arguments = (X[source], scores["B_source"][source], X[target], target_b, target_y, priorities)
                for result in evaluate(*arguments, config=config, seed=estimator_seed, regressor_factory=factory):
                    count = int(result.labeled.sum())
                    if count not in remaining or count in completed or not np.array_equal(result.labeled, masks[count]):
                        raise ValueError("Estimator returned duplicate or unexpected label masks")
                    truth, finite_truth = _load_truth(root, setting, target)
                    rows = []
                    for method, values in result.estimates.items():
                        row = dict(unit, model=MODEL, split_seed=split_seed,
                                   generation_seed=seed(123, 901), n_source=int(source.sum()),
                                   n_target=int(target.sum()), n_target_labels=count, outer_folds=5,
                                   method=method, estimate=float(values["estimate"]),
                                   standard_error=float(values["standard_error"]), truth=truth,
                                   error=float(values["estimate"]) - truth, truth_conditional_mean=truth,
                                   truth_finite_population=finite_truth, fingerprint=identity,
                                   release_id=release_id, status="ok", elapsed_seconds=time.monotonic() - start)
                        for name in ("plugin_mean", "residual_correction", "zero_coefficient_count", "fold_count"):
                            if name in values:
                                row[name] = float(values[name]) if name in ("plugin_mean", "residual_correction") else int(values[name])
                        rows.append(row)
                    part = dict(fingerprint=identity, unit=unit, data_hash=data_hash,
                                label_rows=target_rows[masks[count]].tolist(), rows=rows)
                    validate_part(part, identity, unit, count, target_rows[masks[count]].tolist(), data_hash)
                    atomic_json(directory / f"labels_{count}.json", part)
                    completed.add(count)
                    print(f"Completed {key}: {count} labels, {len(rows)} methods", flush=True)
                if any(count not in completed for count in remaining):
                    raise RuntimeError("Estimator did not return all requested budgets")
            result = dict(unit, fingerprint=identity, data_hash=data_hash,
                          completed_labels=sorted(completed), row_count=len(completed) * len(METHODS[scenario]),
                          status="complete" if completed == set(LABEL_COUNTS) else "partial")
            if result["status"] == "complete":
                marker = directory / "complete.json"
                if marker.exists() and read_json(marker) != result:
                    raise ValueError("Existing completion marker conflicts with checkpoints")
                if not marker.exists():
                    atomic_json(marker, result)
            return result
        except BaseException as exc:
            atomic_json(output / "failures" / f"{key}__{uuid.uuid4().hex}.json",
                        dict(unit, fingerprint=identity, error_type=type(exc).__name__,
                             error=_safe_error(exc, root, output), status="failed",
                             elapsed_seconds=time.monotonic() - start))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--split", required=True, type=int)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--labels", type=int, nargs="+", default=list(LABEL_COUNTS),
                        help="Optional subset for smoke checks; all three budgets are the default")
    args = parser.parse_args(argv)
    result = run_unit(args.setting, args.split, output=args.output, labels=args.labels, threads=args.threads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
