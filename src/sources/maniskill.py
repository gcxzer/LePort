"""Source adapter for paired ManiSkill trajectory HDF5 and JSON files.

ManiSkill stores transitions in ``traj_<episode_id>`` groups. Actions and transition signals use
length ``T``, while observations and environment states include both transition endpoints and use
length ``T+1``. This adapter exposes explicit current and next selectors without interpreting robot
semantics or requiring the ManiSkill runtime.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..errors import OptionalDependencyError, SourceSchemaError
from .base import SOURCE_ADAPTER_API_VERSION
from .types import (
    DatasetInspection,
    EpisodeSelection,
    FieldInspection,
    ProbeResult,
    SourceEpisode,
    SourceFrame,
)

__all__ = ["ManiSkillAdapter"]

_TRAJECTORY_PATTERN = re.compile(r"traj_(\d+)$")


class ManiSkillAdapter:
    """Read standard ManiSkill trajectories from one HDF5 and JSON pair."""

    name: ClassVar[str] = "maniskill"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = "maniskill"

    def probe(self, source: Path) -> ProbeResult:
        """Recognize a complete ManiSkill pair without reading frame payloads."""

        h5py = _require_h5py()
        if not source.is_file():
            return ProbeResult(self.name, 0, "Source path is not a regular HDF5 file")
        try:
            metadata_path, metadata = _load_metadata(source)
            with h5py.File(source, "r") as h5_file:
                episode_ids, _, _ = _validated_catalog(
                    h5_file,
                    metadata,
                    source=source,
                    metadata_path=metadata_path,
                )
        except OSError as exc:
            return ProbeResult(self.name, 0, f"Source is not readable HDF5: {exc}")
        except SourceSchemaError as exc:
            return ProbeResult(self.name, 0, exc.message)
        return ProbeResult(
            self.name,
            100,
            f"Detected paired ManiSkill trajectory data with {len(episode_ids)} episodes",
        )

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Inspect projected field schemas, episode coverage, and JSON metadata."""

        h5py = _require_h5py()
        metadata_path, metadata = _load_metadata(source)
        try:
            h5_file = h5py.File(source, "r")
        except OSError as exc:
            raise SourceSchemaError(
                "Could not open ManiSkill HDF5 file",
                context={"source": str(source), "reason": str(exc)},
            ) from exc

        with h5_file:
            available_ids, episode_metadata, episode_lengths = _validated_catalog(
                h5_file,
                metadata,
                source=source,
                metadata_path=metadata_path,
            )
            selected_ids = _selected_episode_ids(
                available_ids,
                selection or EpisodeSelection(),
            )
            selected_lengths = {episode_id: episode_lengths[episode_id] for episode_id in selected_ids}
            field_records: dict[str, list[tuple[str, str, tuple[int, ...], int]]] = {}

            for episode_id in selected_ids:
                trajectory = h5_file[episode_id]
                bindings = _field_bindings(
                    trajectory,
                    h5py=h5py,
                    episode_id=episode_id,
                    action_length=episode_lengths[episode_id],
                )
                for selector, (dataset, _, _) in bindings.items():
                    field_records.setdefault(selector, []).append(
                        (
                            episode_id,
                            str(dataset.dtype),
                            tuple(int(size) for size in dataset.shape[1:]),
                            episode_lengths[episode_id],
                        )
                    )

            fields: list[FieldInspection] = []
            diagnostics: list[str] = []
            for selector in sorted(field_records):
                records = field_records[selector]
                present_ids = {record[0] for record in records}
                dtypes = tuple(sorted({record[1] for record in records}))
                shapes = tuple(sorted({record[2] for record in records}))
                field = FieldInspection(
                    selector=selector,
                    dtypes=dtypes,
                    shapes=shapes,
                    episode_lengths={record[0]: record[3] for record in records},
                    missing_episodes=tuple(
                        episode_id for episode_id in selected_ids if episode_id not in present_ids
                    ),
                    image_candidate=(
                        dtypes == ("uint8",) and bool(shapes) and all(len(shape) == 3 for shape in shapes)
                    ),
                )
                fields.append(field)
                if not field.schema_consistent:
                    diagnostics.append(
                        f"Field {selector!r} has an inconsistent schema across selected episodes"
                    )

            inspection_metadata: dict[str, Any] = {
                "env_info": metadata["env_info"],
                "hdf5_filename": source.name,
                "json_filename": metadata_path.name,
                "episode_metadata": {episode_id: episode_metadata[episode_id] for episode_id in selected_ids},
            }
            for key in ("source_type", "source_desc"):
                if key in metadata:
                    inspection_metadata[key] = metadata[key]
            return DatasetInspection(
                adapter=self.name,
                source=source,
                episode_ids=tuple(selected_ids),
                episode_lengths=selected_lengths,
                fields=tuple(fields),
                metadata=inspection_metadata,
                diagnostics=tuple(diagnostics),
            )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Yield selected transitions lazily while preserving exact source arrays."""

        h5py = _require_h5py()
        metadata_path, metadata = _load_metadata(source)
        requested_selectors = tuple(dict.fromkeys(selectors)) if selectors is not None else None
        try:
            h5_file = h5py.File(source, "r")
        except OSError as exc:
            raise SourceSchemaError(
                "Could not open ManiSkill HDF5 file",
                context={"source": str(source), "reason": str(exc)},
            ) from exc

        # The generator owns this handle. Exhaustion, explicit close, or an exception leaves the
        # context and releases it even when the current SourceEpisode was only partly consumed.
        with h5_file:
            available_ids, episode_metadata, episode_lengths = _validated_catalog(
                h5_file,
                metadata,
                source=source,
                metadata_path=metadata_path,
            )
            selected_ids = _selected_episode_ids(
                available_ids,
                selection or EpisodeSelection(),
            )
            for episode_id in selected_ids:
                action_length = episode_lengths[episode_id]
                bindings = _field_bindings(
                    h5_file[episode_id],
                    h5py=h5py,
                    episode_id=episode_id,
                    action_length=action_length,
                )
                episode_selectors = (
                    requested_selectors if requested_selectors is not None else tuple(sorted(bindings))
                )
                missing = [selector for selector in episode_selectors if selector not in bindings]
                if missing:
                    raise SourceSchemaError(
                        "Selected ManiSkill fields are missing",
                        context={
                            "episode": episode_id,
                            "missing": missing,
                            "available": sorted(bindings),
                        },
                    )
                selected_bindings = {selector: bindings[selector] for selector in episode_selectors}
                frame_metadata = dict(episode_metadata[episode_id])
                frame_metadata["hdf5_filename"] = source.name
                frame_metadata["json_filename"] = metadata_path.name

                yield SourceEpisode(
                    episode_id=episode_id,
                    length=action_length,
                    frames=(
                        SourceFrame(
                            index=frame_index,
                            fields={
                                selector: np.asarray(dataset[frame_index + offset])
                                for selector, (dataset, offset, _) in selected_bindings.items()
                            },
                        )
                        for frame_index in range(action_length)
                    ),
                    metadata=frame_metadata,
                )


def _require_h5py() -> Any:
    """Import HDF5 support only when the ManiSkill adapter is used."""

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "The maniskill adapter requires HDF5 support; run `uv sync --extra maniskill`",
            context={"adapter": "maniskill", "extra": "maniskill", "dependency": "h5py"},
        ) from exc
    return h5py


def _load_metadata(source: Path) -> tuple[Path, dict[str, Any]]:
    """Load the required companion JSON and validate its dataset-level structure."""

    metadata_path = source.with_suffix(".json")
    if not source.is_file():
        raise SourceSchemaError(
            "ManiSkill source must be a regular HDF5 file",
            context={"source": str(source)},
        )
    if not metadata_path.is_file():
        raise SourceSchemaError(
            "ManiSkill source is missing its same-basename JSON metadata file",
            context={"source": str(source), "metadata": str(metadata_path)},
        )
    try:
        loaded: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSchemaError(
            "Could not parse ManiSkill JSON metadata",
            context={"metadata": str(metadata_path), "reason": str(exc)},
        ) from exc
    if not isinstance(loaded, dict):
        raise SourceSchemaError(
            "ManiSkill JSON metadata root must be an object",
            context={"metadata": str(metadata_path)},
        )
    if not isinstance(loaded.get("env_info"), dict):
        raise SourceSchemaError(
            "ManiSkill JSON metadata requires an `env_info` object",
            context={"metadata": str(metadata_path), "path": "env_info"},
        )
    if not isinstance(loaded.get("episodes"), list):
        raise SourceSchemaError(
            "ManiSkill JSON metadata requires an `episodes` array",
            context={"metadata": str(metadata_path), "path": "episodes"},
        )
    return metadata_path, dict(loaded)


def _validated_catalog(
    h5_file: Any,
    metadata: dict[str, Any],
    *,
    source: Path,
    metadata_path: Path,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, int]]:
    """Cross-check JSON episodes, HDF5 groups, actions, and declared transition counts."""

    h5py = _require_h5py()
    hdf5_ids: dict[int, str] = {}
    for key in h5_file:
        match = _TRAJECTORY_PATTERN.fullmatch(key)
        if match is None:
            continue
        numeric_id = int(match.group(1))
        if key != f"traj_{numeric_id}":
            raise SourceSchemaError(
                "ManiSkill trajectory group uses a non-canonical numeric ID",
                context={"source": str(source), "group": key, "numeric_id": numeric_id},
            )
        if numeric_id in hdf5_ids:
            raise SourceSchemaError(
                "ManiSkill HDF5 contains duplicate numeric trajectory IDs",
                context={"source": str(source), "numeric_id": numeric_id},
            )
        if not isinstance(h5_file.get(key), h5py.Group):
            raise SourceSchemaError(
                "ManiSkill trajectory entry is not an HDF5 group",
                context={"source": str(source), "group": key},
            )
        hdf5_ids[numeric_id] = key
    if not hdf5_ids:
        raise SourceSchemaError(
            "ManiSkill HDF5 contains no numeric `traj_<episode_id>` groups",
            context={"source": str(source)},
        )

    json_ids: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(metadata["episodes"]):
        if not isinstance(item, dict):
            raise SourceSchemaError(
                "Each ManiSkill JSON episode must be an object",
                context={"metadata": str(metadata_path), "index": index},
            )
        episode_id = item.get("episode_id")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise SourceSchemaError(
                "ManiSkill JSON episode IDs must be integers",
                context={"metadata": str(metadata_path), "index": index, "episode_id": episode_id},
            )
        if episode_id in json_ids:
            raise SourceSchemaError(
                "ManiSkill JSON contains duplicate episode IDs",
                context={"metadata": str(metadata_path), "episode_id": episode_id},
            )
        json_ids[episode_id] = dict(item)

    missing_hdf5 = sorted(set(json_ids) - set(hdf5_ids))
    missing_json = sorted(set(hdf5_ids) - set(json_ids))
    if missing_hdf5 or missing_json:
        raise SourceSchemaError(
            "ManiSkill JSON and HDF5 episode catalogs do not match",
            context={
                "metadata_without_hdf5": [f"traj_{item}" for item in missing_hdf5],
                "hdf5_without_metadata": [f"traj_{item}" for item in missing_json],
            },
        )

    episode_ids: list[str] = []
    episode_metadata: dict[str, dict[str, Any]] = {}
    episode_lengths: dict[str, int] = {}
    for numeric_id in sorted(hdf5_ids):
        public_id = f"traj_{numeric_id}"
        trajectory = h5_file[hdf5_ids[numeric_id]]
        actions = trajectory.get("actions")
        if not isinstance(actions, h5py.Dataset) or len(actions.shape) < 1:
            raise SourceSchemaError(
                "ManiSkill episode is missing frame-addressable actions",
                context={"episode": public_id, "selector": "actions"},
            )
        action_length = int(actions.shape[0])
        declared_length = json_ids[numeric_id].get("elapsed_steps")
        if isinstance(declared_length, bool) or not isinstance(declared_length, int):
            raise SourceSchemaError(
                "ManiSkill episode requires an integer `elapsed_steps`",
                context={
                    "metadata": str(metadata_path),
                    "episode": public_id,
                    "elapsed_steps": declared_length,
                },
            )
        if declared_length != action_length:
            raise SourceSchemaError(
                "ManiSkill elapsed_steps does not match actions length",
                context={
                    "episode": public_id,
                    "elapsed_steps": declared_length,
                    "actions_length": action_length,
                },
            )
        episode_ids.append(public_id)
        episode_metadata[public_id] = json_ids[numeric_id]
        episode_lengths[public_id] = action_length
    return episode_ids, episode_metadata, episode_lengths


def _selected_episode_ids(available: list[str], selection: EpisodeSelection) -> list[str]:
    """Apply explicit selection while retaining the validated numeric catalog order."""

    if selection.filter_key is not None:
        raise SourceSchemaError(
            "ManiSkill filter keys are not supported in this release",
            context={"filter_key": selection.filter_key},
        )
    if not selection.episode_ids:
        return available
    unknown = sorted(set(selection.episode_ids) - set(available))
    if unknown:
        raise SourceSchemaError(
            "Explicit episode list contains unknown ManiSkill IDs",
            context={"unknown": unknown, "available": available},
        )
    selected = set(selection.episode_ids)
    return [episode_id for episode_id in available if episode_id in selected]


def _dataset_leaves(value: Any, *, h5py: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield dataset leaves in stable order without reading their contents."""

    if isinstance(value, h5py.Dataset):
        yield prefix, value
        return
    if not isinstance(value, h5py.Group):
        return
    for key in sorted(value.keys()):
        child_prefix = f"{prefix}/{key}" if prefix else key
        yield from _dataset_leaves(value[key], h5py=h5py, prefix=child_prefix)


def _field_bindings(
    trajectory: Any,
    *,
    h5py: Any,
    episode_id: str,
    action_length: int,
) -> dict[str, tuple[Any, int, str]]:
    """Map public selectors to datasets and endpoint offsets after strict length validation."""

    bindings: dict[str, tuple[Any, int, str]] = {}
    for raw_selector, dataset in _dataset_leaves(trajectory, h5py=h5py):
        if len(dataset.shape) < 1:
            raise SourceSchemaError(
                "ManiSkill field is not frame-addressable",
                context={"episode": episode_id, "selector": raw_selector},
            )
        public_selectors: tuple[tuple[str, int], ...]
        if raw_selector == "obs" or raw_selector.startswith("obs/"):
            expected_length = action_length + 1
            public_selectors = (
                (raw_selector, 0),
                (f"next_obs{raw_selector[len('obs') :]}", 1),
            )
        elif raw_selector == "env_states" or raw_selector.startswith("env_states/"):
            expected_length = action_length + 1
            public_selectors = (
                (raw_selector, 0),
                (f"next_env_states{raw_selector[len('env_states') :]}", 1),
            )
        else:
            expected_length = action_length
            public_selectors = ((raw_selector, 0),)
        actual_length = int(dataset.shape[0])
        if actual_length != expected_length:
            raise SourceSchemaError(
                "ManiSkill field length violates transition alignment",
                context={
                    "episode": episode_id,
                    "selector": raw_selector,
                    "actions_length": action_length,
                    "expected_length": expected_length,
                    "actual_length": actual_length,
                },
            )
        for public_selector, offset in public_selectors:
            if public_selector in bindings:
                raise SourceSchemaError(
                    "ManiSkill field selectors collide after transition projection",
                    context={"episode": episode_id, "selector": public_selector},
                )
            bindings[public_selector] = (dataset, offset, raw_selector)
    return bindings
