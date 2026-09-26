"""Validated, serializable experiment configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{key} must be a non-empty mapping.")
    return value


def _mapping_sequence(mapping: Mapping[str, Any], key: str) -> Tuple[Mapping[str, Any], ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{key} must be a non-empty list.")
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or not item:
            raise ValueError(f"{key}[{index}] must be a non-empty mapping.")
        output.append(_freeze(item))
    return tuple(output)


def _text_sequence(mapping: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{key} must be a non-empty list.")
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key}[{index}] must be a non-empty string.")
        output.append(item.strip())
    return tuple(output)


def _validate_project_relative(path: Path, name: str) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe project-relative path; got {path}.")


@dataclass(frozen=True)
class ExperimentConfig:
    """A stable experiment definition independent of execution environment."""

    schema_version: int
    experiment_id: str
    description: str
    status: str
    random_seed: int
    artifact_root: Path
    dataset: Mapping[str, Any]
    split: Mapping[str, Any]
    representations: Tuple[Mapping[str, Any], ...]
    methods: Tuple[Mapping[str, Any], ...]
    estimands: Tuple[str, ...]
    primary_metrics: Tuple[str, ...]
    tags: Tuple[str, ...] = ()
    notes: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("Experiment configuration must be a mapping.")

        schema_version = raw.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
            raise ValueError("schema_version must be a positive integer.")

        experiment_id = _required_text(raw, "experiment_id")
        if not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError("experiment_id must contain only lowercase letters, digits, and hyphens.")

        random_seed = raw.get("random_seed")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer.")

        artifact_root = Path(_required_text(raw, "artifact_root"))
        _validate_project_relative(artifact_root, "artifact_root")

        tags_raw = raw.get("tags", [])
        if not isinstance(tags_raw, Sequence) or isinstance(tags_raw, (str, bytes)):
            raise ValueError("tags must be a list of strings.")
        tags = tuple(str(tag).strip() for tag in tags_raw)
        if any(not tag for tag in tags):
            raise ValueError("tags cannot contain empty values.")

        notes = raw.get("notes")
        if notes is not None and (not isinstance(notes, str) or not notes.strip()):
            raise ValueError("notes must be null or a non-empty string.")

        known = {
            "schema_version",
            "experiment_id",
            "description",
            "status",
            "random_seed",
            "artifact_root",
            "dataset",
            "split",
            "representations",
            "methods",
            "estimands",
            "primary_metrics",
            "tags",
            "notes",
        }
        extra = {str(key): value for key, value in raw.items() if key not in known}

        return cls(
            schema_version=schema_version,
            experiment_id=experiment_id,
            description=_required_text(raw, "description"),
            status=_required_text(raw, "status"),
            random_seed=random_seed,
            artifact_root=artifact_root,
            dataset=_freeze(_required_mapping(raw, "dataset")),
            split=_freeze(_required_mapping(raw, "split")),
            representations=_mapping_sequence(raw, "representations"),
            methods=_mapping_sequence(raw, "methods"),
            estimands=_text_sequence(raw, "estimands"),
            primary_metrics=_text_sequence(raw, "primary_metrics"),
            tags=tags,
            notes=notes.strip() if notes else None,
            extra=_freeze(extra),
        )

    def to_dict(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "status": self.status,
            "random_seed": self.random_seed,
            "artifact_root": self.artifact_root.as_posix(),
            "dataset": _thaw(self.dataset),
            "split": _thaw(self.split),
            "representations": _thaw(self.representations),
            "methods": _thaw(self.methods),
            "estimands": list(self.estimands),
            "primary_metrics": list(self.primary_metrics),
            "tags": list(self.tags),
        }
        if self.notes is not None:
            output["notes"] = self.notes
        output.update(_thaw(self.extra))
        return output


def _parse_yaml_or_json(text: str, path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{path} requires PyYAML because it is not JSON-compatible YAML. "
                "Install the project dependencies first."
            ) from error
    else:
        parsed = yaml.safe_load(text)

    if not isinstance(parsed, Mapping):
        raise ValueError(f"{path} must contain a mapping at the document root.")
    return parsed


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = _parse_yaml_or_json(config_path.read_text(encoding="utf-8"), config_path)
    return ExperimentConfig.from_mapping(raw)
