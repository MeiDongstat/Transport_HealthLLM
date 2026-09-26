"""Safe lifecycle for immutable experiment run directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .config import ExperimentConfig
from .provenance import build_run_provenance, sha256_file


class RunDisposition(str, Enum):
    CREATED = "created"
    RESUME = "resume"
    REUSE = "reuse"


@dataclass(frozen=True)
class ArtifactRun:
    project_root: Path
    config: ExperimentConfig
    provenance: Mapping[str, Any]
    run_dir: Path

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        project_root: Path,
        *,
        data_files: Optional[Mapping[str, Path]] = None,
        model_revisions: Optional[Mapping[str, str]] = None,
    ) -> "ArtifactRun":
        root = Path(project_root).resolve()
        provenance = build_run_provenance(
            config,
            root,
            data_files=data_files,
            model_revisions=model_revisions,
        )
        artifact_root = (root / config.artifact_root).resolve()
        try:
            artifact_root.relative_to(root)
        except ValueError as error:
            raise ValueError("artifact_root escapes project_root.") from error
        run_dir = artifact_root / str(provenance["run_id"])
        return cls(
            project_root=root,
            config=config,
            provenance=provenance,
            run_dir=run_dir,
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def success_path(self) -> Path:
        return self.run_dir / "_SUCCESS"

    def _manifest(self) -> Dict[str, Any]:
        return {
            "manifest_schema_version": 2,
            "experiment_id": self.config.experiment_id,
            "run_id": self.provenance["run_id"],
            "fingerprint": self.provenance["fingerprint"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "outputs": [],
            "provenance": self.provenance["identity"],
            "observed_repository_state": self.provenance.get("observed_repository_state", {}),
        }

    def _read_manifest(self) -> Dict[str, Any]:
        try:
            parsed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(f"Run directory has no manifest: {self.run_dir}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Run manifest is invalid JSON: {self.manifest_path}") from error
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Run manifest must be a JSON object: {self.manifest_path}")
        return parsed

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("experiment_id") != self.config.experiment_id:
            raise RuntimeError(f"Run directory belongs to another experiment: {self.run_dir}")
        if manifest.get("fingerprint") != self.provenance["fingerprint"]:
            raise RuntimeError(f"Refusing to reuse run directory with different provenance: {self.run_dir}")

    def prepare(self) -> RunDisposition:
        """Create, resume, or reuse a run without overwriting any successful output.

        If manifest writing fails, the empty run directory is left behind on
        purpose: a later call reaches `_read_manifest` and refuses it, so the
        partial state cannot be mistaken for a real run.
        """

        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        # Bare mkdir (no exist_ok) is the atomic create-or-detect-existing test.
        try:
            self.run_dir.mkdir()
        except FileExistsError:
            if not self.run_dir.is_dir():
                raise RuntimeError(f"Run path exists but is not a directory: {self.run_dir}")
            manifest = self._read_manifest()
            self._validate_manifest(manifest)
            return RunDisposition.REUSE if self.success_path.is_file() else RunDisposition.RESUME

        with self.manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(self._manifest(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return RunDisposition.CREATED

    def output_path(self, relative_path: Path, *, allow_existing: bool = False) -> Path:
        """Resolve one output path while preventing path escape and silent overwrite."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Output path must stay inside the run directory: {relative}")
        output = (self.run_dir / relative).resolve()
        try:
            output.relative_to(self.run_dir.resolve())
        except ValueError as error:
            raise ValueError(f"Output path escapes run directory: {relative}") from error
        if output.exists() and not allow_existing:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    def _output_inventory(self) -> list[Dict[str, Any]]:
        excluded = {self.manifest_path, self.success_path}
        inventory = []
        for path in sorted(self.run_dir.rglob("*")):
            if path in excluded or not path.is_file():
                continue
            if path.is_symlink():
                raise RuntimeError(f"Run output inventory refuses symbolic links: {path}")
            inventory.append(
                {
                    "path": path.relative_to(self.run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return inventory

    def _write_completed_manifest(self, completed_at_utc: str) -> None:
        manifest = self._read_manifest()
        self._validate_manifest(manifest)
        completed = dict(manifest)
        completed.update(
            {
                "manifest_schema_version": 2,
                "status": "success",
                "completed_at_utc": completed_at_utc,
                "outputs": self._output_inventory(),
            }
        )
        temporary_path = self.run_dir / ".manifest.json.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8") as handle:
                json.dump(completed, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            temporary_path.replace(self.manifest_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def backfill_completed_manifest(self) -> None:
        """Add completion metadata to a successful pre-schema-2 run."""

        manifest = self._read_manifest()
        self._validate_manifest(manifest)
        try:
            marker = json.loads(self.success_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(f"Run has no success marker: {self.run_dir}") from error
        if marker.get("fingerprint") != self.provenance["fingerprint"]:
            raise RuntimeError(f"Conflicting success marker: {self.success_path}")
        self._write_completed_manifest(str(marker["completed_at_utc"]))

    def mark_success(self, required_outputs: Iterable[Path] = ()) -> None:
        """Seal a run after checking its declared canonical outputs."""

        manifest = self._read_manifest()
        self._validate_manifest(manifest)
        if self.success_path.is_file():
            existing = json.loads(self.success_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.provenance["fingerprint"]:
                raise RuntimeError(f"Conflicting success marker: {self.success_path}")
            if manifest.get("status") != "success" or "outputs" not in manifest:
                self.backfill_completed_manifest()
            return

        missing = []
        for relative in required_outputs:
            candidate = self.output_path(relative, allow_existing=True)
            if not candidate.is_file():
                missing.append(str(relative))
        if missing:
            raise FileNotFoundError(f"Cannot seal run; missing required outputs: {missing}")

        completed_at_utc = datetime.now(timezone.utc).isoformat()
        marker = {
            "fingerprint": self.provenance["fingerprint"],
            "completed_at_utc": completed_at_utc,
        }
        self._write_completed_manifest(completed_at_utc)
        with self.success_path.open("x", encoding="utf-8") as handle:
            json.dump(marker, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
