"""Versioned ConversionPlan model with strict YAML serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import yaml

from ..errors import PlanValidationError
from ..sources.types import EpisodeSelection

__all__ = [
    "PLAN_SCHEMA_VERSION",
    "ConversionPlan",
    "FeatureMapping",
    "FeatureSpec",
    "TargetConfig",
    "TaskProvider",
    "load_plan",
    "plan_from_dict",
    "save_plan",
]

PLAN_SCHEMA_VERSION = 1
_LEROBOT_AUTOMATIC_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_SPECIAL_DTYPES = {"image", "video", "string", "language"}


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Location and writing options for the LeRobot target dataset."""

    repo_id: str
    root: Path
    robot_type: str | None = None
    use_videos: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert target configuration into stable YAML/JSON fields."""

        return {
            "repo_id": self.repo_id,
            "root": str(self.root),
            "robot_type": self.robot_type,
            "use_videos": self.use_videos,
        }


@dataclass(frozen=True, slots=True)
class TaskProvider:
    """Explicit source of natural-language task text for each LeRobot frame."""

    kind: Literal["static", "metadata"]
    value: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the explicit static or metadata task source."""

        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Dtype, shape, and optional axis names for one target LeRobot feature."""

    dtype: str
    shape: tuple[int, ...]
    names: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize a target feature while preserving optional LeRobot axis names."""

        result: dict[str, Any] = {"dtype": self.dtype, "shape": list(self.shape)}
        if self.names is not None:
            result["names"] = self.names
        return result

    def to_lerobot(self) -> dict[str, Any]:
        """Return the dictionary accepted by the current LeRobot ``features`` argument."""

        return self.to_dict()


@dataclass(frozen=True, slots=True)
class FeatureMapping:
    """Mechanical mapping from source selectors to one target feature."""

    sources: tuple[str, ...]
    operation: Literal["direct", "concat"] = "direct"
    cast: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize ordered selectors, the mechanical operation, and an optional dtype cast."""

        result: dict[str, Any] = {
            "operation": self.operation,
            "sources": list(self.sources),
        }
        if self.cast is not None:
            result["cast"] = self.cast
        return result


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    """Complete reproducible semantics for one conversion."""

    adapter: str
    source: Path
    selection: EpisodeSelection
    target: TargetConfig
    fps: int
    task: TaskProvider
    features: Mapping[str, FeatureSpec]
    mappings: Mapping[str, FeatureMapping]
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze feature mappings and validate the complete structure at construction time."""

        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "mappings", MappingProxyType(dict(self.mappings)))
        self.validate()

    @property
    def source_selectors(self) -> tuple[str, ...]:
        """Deduplicate selectors in plan order to identify fields the adapter must read."""

        return tuple(
            dict.fromkeys(selector for mapping in self.mappings.values() for selector in mapping.sources)
        )

    def validate(self) -> None:
        """Validate plan structure without accessing the source or target directory."""

        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise PlanValidationError(
                "Unsupported ConversionPlan schema_version",
                context={"actual": self.schema_version, "supported": PLAN_SCHEMA_VERSION},
            )
        if not self.adapter.strip():
            raise PlanValidationError("adapter cannot be empty")
        if not str(self.source):
            raise PlanValidationError("source cannot be empty")
        if not self.target.repo_id.strip() or not str(self.target.root):
            raise PlanValidationError("target.repo_id and target.root cannot be empty")
        if not isinstance(self.fps, int) or isinstance(self.fps, bool) or self.fps <= 0:
            raise PlanValidationError("fps must be a positive integer")
        if self.task.kind not in {"static", "metadata"} or not self.task.value.strip():
            raise PlanValidationError("task must contain a non-empty static value or metadata selector")
        if not self.features:
            raise PlanValidationError("features cannot be empty")
        if set(self.features) != set(self.mappings):
            raise PlanValidationError(
                "features and mappings must contain exactly the same target names",
                context={
                    "without_mapping": sorted(set(self.features) - set(self.mappings)),
                    "without_feature": sorted(set(self.mappings) - set(self.features)),
                },
            )
        forbidden = sorted(set(self.features) & (_LEROBOT_AUTOMATIC_FEATURES | {"task"}))
        if forbidden:
            raise PlanValidationError(
                "Plans cannot declare LeRobot-managed fields or the special task field",
                context={"features": forbidden},
            )
        if "action" not in self.features:
            raise PlanValidationError("The plan must explicitly declare and map the target `action` feature")

        for target_name, spec in self.features.items():
            if not spec.dtype:
                raise PlanValidationError("feature dtype cannot be empty", context={"target": target_name})
            if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in spec.shape):
                raise PlanValidationError(
                    "Every feature shape dimension must be a positive integer",
                    context={"target": target_name, "shape": spec.shape},
                )
            if spec.dtype not in _SPECIAL_DTYPES:
                try:
                    np.dtype(spec.dtype)
                except TypeError as exc:
                    raise PlanValidationError(
                        "feature dtype is not a valid NumPy dtype",
                        context={"target": target_name, "dtype": spec.dtype},
                    ) from exc
            if spec.dtype in {"image", "video"} and len(spec.shape) != 3:
                raise PlanValidationError(
                    "Image and video features must declare a three-dimensional single-frame shape",
                    context={"target": target_name, "shape": spec.shape},
                )

            mapping = self.mappings[target_name]
            if not mapping.sources or any(not selector.strip() for selector in mapping.sources):
                raise PlanValidationError("mapping sources cannot be empty", context={"target": target_name})
            if mapping.operation == "direct" and len(mapping.sources) != 1:
                raise PlanValidationError(
                    "A direct mapping must contain exactly one source",
                    context={"target": target_name, "sources": mapping.sources},
                )
            if mapping.operation not in {"direct", "concat"}:
                raise PlanValidationError(
                    "Unsupported mapping operation",
                    context={"target": target_name, "operation": mapping.operation},
                )
            if mapping.cast is not None:
                try:
                    np.dtype(mapping.cast)
                except TypeError as exc:
                    raise PlanValidationError(
                        "mapping cast is not a valid NumPy dtype",
                        context={"target": target_name, "cast": mapping.cast},
                    ) from exc
                if spec.dtype in _SPECIAL_DTYPES or mapping.cast != spec.dtype:
                    raise PlanValidationError(
                        "cast applies only to numeric features and must match the target dtype",
                        context={"target": target_name, "cast": mapping.cast, "dtype": spec.dtype},
                    )

    def to_dict(self) -> dict[str, Any]:
        """Build a reproducible plan dictionary with stable top-level ordering."""

        return {
            "schema_version": self.schema_version,
            "adapter": self.adapter,
            "source": str(self.source),
            "selection": self.selection.to_dict(),
            "target": self.target.to_dict(),
            "fps": self.fps,
            "task": self.task.to_dict(),
            "features": {name: spec.to_dict() for name, spec in self.features.items()},
            "mappings": {name: mapping.to_dict() for name, mapping in self.mappings.items()},
        }


def plan_from_dict(raw: Any) -> ConversionPlan:
    """Construct a strict ConversionPlan from deserialized plain values."""

    root = _strict_mapping(
        raw,
        location="plan",
        required={
            "schema_version",
            "adapter",
            "source",
            "selection",
            "target",
            "fps",
            "task",
            "features",
            "mappings",
        },
    )
    selection_raw = _strict_mapping(
        root["selection"],
        location="selection",
        required={"episode_ids", "filter_key"},
    )
    target_raw = _strict_mapping(
        root["target"],
        location="target",
        required={"repo_id", "root", "robot_type", "use_videos"},
    )
    task_raw = _strict_mapping(root["task"], location="task", required={"kind", "value"})

    # YAML parses unquoted booleans and numbers into non-string scalars. Explicit type checks prevent
    # a visually plausible plan with changed semantics from entering conversion.
    if not isinstance(root["schema_version"], int) or isinstance(root["schema_version"], bool):
        raise PlanValidationError("schema_version must be an integer")
    if not isinstance(root["adapter"], str):
        raise PlanValidationError("adapter must be a string")
    if not isinstance(root["source"], str):
        raise PlanValidationError("source must be a string path")
    if not isinstance(root["fps"], int) or isinstance(root["fps"], bool):
        raise PlanValidationError("fps must be an integer")

    episode_ids_raw = selection_raw["episode_ids"]
    if not isinstance(episode_ids_raw, Sequence) or isinstance(episode_ids_raw, (str, bytes)):
        raise PlanValidationError("selection.episode_ids must be a list of strings")
    if any(not isinstance(item, str) for item in episode_ids_raw):
        raise PlanValidationError("Every selection.episode_ids value must be a string")
    if selection_raw["filter_key"] is not None and not isinstance(selection_raw["filter_key"], str):
        raise PlanValidationError("selection.filter_key must be a string or null")

    if not isinstance(target_raw["repo_id"], str):
        raise PlanValidationError("target.repo_id must be a string")
    if not isinstance(target_raw["root"], str):
        raise PlanValidationError("target.root must be a string path")
    if target_raw["robot_type"] is not None and not isinstance(target_raw["robot_type"], str):
        raise PlanValidationError("target.robot_type must be a string or null")
    if not isinstance(target_raw["use_videos"], bool):
        raise PlanValidationError("target.use_videos must be a boolean")

    if not isinstance(task_raw["kind"], str) or task_raw["kind"] not in {"static", "metadata"}:
        raise PlanValidationError("task.kind must be static or metadata")
    if not isinstance(task_raw["value"], str):
        raise PlanValidationError("task.value must be a string")

    if not isinstance(root["features"], Mapping):
        raise PlanValidationError("features must be a mapping")
    if not isinstance(root["mappings"], Mapping):
        raise PlanValidationError("mappings must be a mapping")
    features_raw = root["features"]
    mappings_raw = root["mappings"]

    features: dict[str, FeatureSpec] = {}
    for name, value in features_raw.items():
        if not isinstance(name, str):
            raise PlanValidationError("Feature target names must be strings")
        feature_raw = _strict_mapping(
            value,
            location=f"features.{name}",
            required={"dtype", "shape"},
            optional={"names"},
        )
        shape_raw = feature_raw["shape"]
        if not isinstance(shape_raw, Sequence) or isinstance(shape_raw, (str, bytes)):
            raise PlanValidationError(f"features.{name}.shape must be a list of integers")
        if not isinstance(feature_raw["dtype"], str):
            raise PlanValidationError(f"features.{name}.dtype must be a string")
        features[name] = FeatureSpec(
            dtype=feature_raw["dtype"],
            shape=tuple(shape_raw),
            names=feature_raw.get("names"),
        )

    mappings: dict[str, FeatureMapping] = {}
    for name, value in mappings_raw.items():
        if not isinstance(name, str):
            raise PlanValidationError("Mapping target names must be strings")
        mapping_raw = _strict_mapping(
            value,
            location=f"mappings.{name}",
            required={"operation", "sources"},
            optional={"cast"},
        )
        sources_raw = mapping_raw["sources"]
        if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
            raise PlanValidationError(f"mappings.{name}.sources must be a list of strings")
        if any(not isinstance(item, str) for item in sources_raw):
            raise PlanValidationError(f"Every mappings.{name}.sources value must be a string")
        if not isinstance(mapping_raw["operation"], str):
            raise PlanValidationError(f"mappings.{name}.operation must be a string")
        cast = mapping_raw.get("cast")
        if cast is not None and not isinstance(cast, str):
            raise PlanValidationError(f"mappings.{name}.cast must be a string or null")
        mappings[name] = FeatureMapping(
            operation=mapping_raw["operation"],  # type: ignore[arg-type]
            sources=tuple(sources_raw),
            cast=cast,
        )

    try:
        selection = EpisodeSelection(
            episode_ids=tuple(episode_ids_raw),
            filter_key=selection_raw["filter_key"],
        )
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("selection is invalid", context={"reason": str(exc)}) from exc

    return ConversionPlan(
        schema_version=root["schema_version"],
        adapter=root["adapter"],
        source=Path(root["source"]),
        selection=selection,
        target=TargetConfig(
            repo_id=target_raw["repo_id"],
            root=Path(target_raw["root"]),
            robot_type=target_raw["robot_type"],
            use_videos=target_raw["use_videos"],
        ),
        fps=root["fps"],
        task=TaskProvider(kind=task_raw["kind"], value=task_raw["value"]),  # type: ignore[arg-type]
        features=features,
        mappings=mappings,
    )


def load_plan(path: str | Path) -> ConversionPlan:
    """Load YAML safely and convert syntax failures into stable plan errors."""

    plan_path = Path(path)
    try:
        with plan_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise PlanValidationError(
            "Could not read ConversionPlan YAML",
            context={"path": str(plan_path), "reason": str(exc)},
        ) from exc
    return plan_from_dict(raw)


def save_plan(plan: ConversionPlan, path: str | Path) -> Path:
    """Save YAML with stable field ordering and reject implicit overwrite."""

    plan_path = Path(path)
    if plan_path.exists():
        raise PlanValidationError("Plan file already exists", context={"path": str(plan_path)})
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(plan.to_dict(), file, sort_keys=False, allow_unicode=True)
    return plan_path


def _strict_mapping(
    value: Any,
    *,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    """Require a YAML mapping with no missing or unknown keys."""

    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{location} must be a mapping")
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise PlanValidationError(
            f"{location} fields do not match the schema",
            context={"missing": sorted(missing), "unknown": sorted(unknown)},
        )
    return value
