"""Source adapter for standard robomimic HDF5 datasets.

This module understands only the robomimic source structure. It does not choose LeRobot feature names
or interpret action coordinates. ``h5py`` is imported only when the adapter is used, so the LePort core
and other adapters remain available without the HDF5 extra.
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

__all__ = ["RobomimicAdapter"]

_DEMO_PATTERN = re.compile(r"demo_(\d+)$")


class RobomimicAdapter:
    """Read standard ``data/demo_*`` robomimic HDF5 files."""

    name: ClassVar[str] = "robomimic"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = "robomimic"

    def probe(self, source: Path) -> ProbeResult:
        """Probe actual HDF5 groups, demos, and actions without relying on the file extension."""

        h5py = _require_h5py()
        if not source.is_file():
            return ProbeResult(self.name, 0, "Source path is not a regular file")
        try:
            with h5py.File(source, "r") as h5_file:
                data_group = h5_file.get("data")
                if not isinstance(data_group, h5py.Group):
                    return ProbeResult(self.name, 0, "HDF5 file is missing the root `data` group")
                episode_ids = _episode_ids(data_group)
                if not episode_ids:
                    return ProbeResult(self.name, 0, "`data` contains no `demo_<integer>` episodes")
                missing_actions = [
                    episode_id for episode_id in episode_ids if "actions" not in data_group[episode_id]
                ]
                if missing_actions:
                    return ProbeResult(
                        self.name,
                        0,
                        f"The following episodes are missing actions: {', '.join(missing_actions)}",
                    )
                # LIBERO is structurally robomimic-compatible. Its dedicated adapter has richer task
                # semantics, so generic auto-detection yields only when the complete signature is valid.
                if source.name.endswith("_demo.hdf5"):
                    problem_info = _metadata_value(data_group.attrs.get("problem_info"))
                    bddl_identity = any(
                        isinstance(_metadata_value(data_group.attrs.get(key)), str)
                        and bool(_metadata_value(data_group.attrs.get(key)).strip())
                        for key in ("bddl_file_name", "bddl_file_content")
                    )
                    try:
                        parsed_problem = json.loads(problem_info) if isinstance(problem_info, str) else None
                    except json.JSONDecodeError:
                        parsed_problem = None
                    instruction = (
                        parsed_problem.get("language_instruction")
                        if isinstance(parsed_problem, dict)
                        else None
                    )
                    if bddl_identity and isinstance(instruction, str) and instruction.strip():
                        return ProbeResult(
                            self.name,
                            80,
                            "Detected robomimic-compatible LIBERO signature; "
                            "specialized adapter has priority",
                        )
        except OSError as exc:
            return ProbeResult(self.name, 0, f"File is not readable HDF5: {exc}")
        return ProbeResult(
            self.name,
            100,
            f"Detected standard `data/demo_*` structure with {len(episode_ids)} episodes",
        )

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Summarize field schemas, lengths, attributes, and masks across selected demos."""

        h5py = _require_h5py()
        resolved_selection = selection or EpisodeSelection()
        try:
            h5_file = h5py.File(source, "r")
        except OSError as exc:
            raise SourceSchemaError(
                "Could not open robomimic HDF5 file",
                context={"source": str(source), "reason": str(exc)},
            ) from exc

        with h5_file:
            selected_ids = _select_episode_ids(h5_file, resolved_selection)
            data_group = h5_file["data"]
            episode_lengths: dict[str, int] = {}
            field_records: dict[str, list[tuple[str, str, tuple[int, ...], int]]] = {}
            diagnostics: list[str] = []

            for episode_id in selected_ids:
                episode_group = data_group[episode_id]
                actions = episode_group.get("actions")
                if not isinstance(actions, h5py.Dataset) or len(actions.shape) < 1:
                    raise SourceSchemaError(
                        "Episode is missing frame-addressable actions",
                        context={"episode": episode_id, "selector": "actions"},
                    )
                action_length = int(actions.shape[0])
                episode_lengths[episode_id] = action_length
                if "num_samples" in episode_group.attrs:
                    declared_length = int(episode_group.attrs["num_samples"])
                    if declared_length != action_length:
                        raise SourceSchemaError(
                            "Episode num_samples does not match actions length",
                            context={
                                "episode": episode_id,
                                "num_samples": declared_length,
                                "actions_length": action_length,
                            },
                        )

                for selector in _field_selectors(episode_group):
                    dataset = episode_group[selector]
                    field_records.setdefault(selector, []).append(
                        (
                            episode_id,
                            str(dataset.dtype),
                            tuple(int(size) for size in dataset.shape[1:]),
                            int(dataset.shape[0]),
                        )
                    )

            fields: list[FieldInspection] = []
            selected_set = set(selected_ids)
            for selector in sorted(field_records):
                records = field_records[selector]
                present_ids = {record[0] for record in records}
                dtypes = tuple(sorted({record[1] for record in records}))
                shapes = tuple(sorted({record[2] for record in records}))
                missing = tuple(episode_id for episode_id in selected_ids if episode_id not in present_ids)
                lengths = {record[0]: record[3] for record in records}
                image_candidate = (
                    dtypes == ("uint8",) and bool(shapes) and all(len(shape) == 3 for shape in shapes)
                )
                field = FieldInspection(
                    selector=selector,
                    dtypes=dtypes,
                    shapes=shapes,
                    episode_lengths=lengths,
                    missing_episodes=missing,
                    image_candidate=image_candidate,
                )
                fields.append(field)
                if not field.schema_consistent:
                    diagnostics.append(
                        f"Field {selector!r} has an inconsistent schema across selected episodes"
                    )
                if set(lengths) - selected_set:
                    raise AssertionError("Internal error: field inspection includes an unselected episode")

            root_attributes = {key: _metadata_value(value) for key, value in h5_file.attrs.items()}
            data_attributes = {key: _metadata_value(value) for key, value in data_group.attrs.items()}
            env_args = data_attributes.get("env_args")
            if isinstance(env_args, str):
                try:
                    data_attributes["env_args"] = json.loads(env_args)
                except json.JSONDecodeError:
                    diagnostics.append("data.attrs['env_args'] is not valid JSON and was preserved as text")
            mask_group = h5_file.get("mask")
            filter_keys = sorted(mask_group.keys()) if isinstance(mask_group, h5py.Group) else []

            return DatasetInspection(
                adapter=self.name,
                source=source,
                episode_ids=tuple(selected_ids),
                episode_lengths=episode_lengths,
                fields=tuple(fields),
                metadata={
                    "root_attributes": root_attributes,
                    "data_attributes": data_attributes,
                    "filter_keys": filter_keys,
                },
                diagnostics=tuple(diagnostics),
            )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Yield demos in numeric order while lazily reading only requested fields."""

        h5py = _require_h5py()
        resolved_selection = selection or EpisodeSelection()
        requested_selectors = tuple(dict.fromkeys(selectors)) if selectors is not None else None

        with h5py.File(source, "r") as h5_file:
            selected_ids = _select_episode_ids(h5_file, resolved_selection)
            data_group = h5_file["data"]
            env_args = _metadata_value(data_group.attrs.get("env_args"))

            for episode_id in selected_ids:
                episode_group = data_group[episode_id]
                actions = episode_group.get("actions")
                if not isinstance(actions, h5py.Dataset) or len(actions.shape) < 1:
                    raise SourceSchemaError(
                        "Episode is missing frame-addressable actions",
                        context={"episode": episode_id, "selector": "actions"},
                    )
                action_length = int(actions.shape[0])
                episode_selectors = (
                    requested_selectors
                    if requested_selectors is not None
                    else _field_selectors(episode_group)
                )
                for selector in episode_selectors:
                    dataset = episode_group.get(selector)
                    if not isinstance(dataset, h5py.Dataset):
                        raise SourceSchemaError(
                            "Selected source field is missing or is not a dataset",
                            context={"episode": episode_id, "selector": selector},
                        )
                    if len(dataset.shape) < 1 or int(dataset.shape[0]) != action_length:
                        actual_length = int(dataset.shape[0]) if len(dataset.shape) >= 1 else None
                        raise SourceSchemaError(
                            "Selected field length differs from actions; "
                            "LePort does not truncate or shift fields",
                            context={
                                "episode": episode_id,
                                "selector": selector,
                                "actions_length": action_length,
                                "field_length": actual_length,
                            },
                        )

                metadata = {key: _metadata_value(value) for key, value in episode_group.attrs.items()}
                if env_args is not None:
                    metadata["env_args"] = env_args
                datasets = {selector: episode_group[selector] for selector in episode_selectors}

                # The outer iterator remains suspended at this yield, keeping the HDF5 handle valid
                # while the caller consumes the current episode as required by SourceEpisode.
                yield SourceEpisode(
                    episode_id=episode_id,
                    length=action_length,
                    frames=(
                        SourceFrame(
                            index=frame_index,
                            fields={
                                selector: np.asarray(dataset[frame_index])
                                for selector, dataset in datasets.items()
                            },
                        )
                        for frame_index in range(action_length)
                    ),
                    metadata=metadata,
                )


def _require_h5py() -> Any:
    """Import h5py on demand and convert import failures into actionable dependency errors."""

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "The robomimic adapter requires HDF5 support; run `uv sync --extra robomimic`",
            context={"adapter": "robomimic", "extra": "robomimic"},
        ) from exc
    return h5py


def _metadata_value(value: Any) -> Any:
    """Convert HDF5 attribute values into plain Python values suitable for YAML/JSON."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    return value


def _episode_ids(data_group: Any) -> list[str]:
    """Accept only ``demo_<integer>`` names and sort by numeric rather than lexical order."""

    matched = []
    for key in data_group:
        match = _DEMO_PATTERN.fullmatch(key)
        if match is not None:
            matched.append((int(match.group(1)), key))
    return [key for _, key in sorted(matched)]


def _select_episode_ids(h5_file: Any, selection: EpisodeSelection) -> list[str]:
    """Resolve all, explicit-list, or mask selection with strict reference diagnostics."""

    h5py = _require_h5py()
    data_group = h5_file.get("data")
    if not isinstance(data_group, h5py.Group):
        raise SourceSchemaError("robomimic file is missing the root `data` group", context={"path": "data"})

    available = _episode_ids(data_group)
    if not available:
        raise SourceSchemaError(
            "robomimic `data` group contains no `demo_<integer>` episodes",
            context={"path": "data"},
        )

    if selection.episode_ids:
        unknown = sorted(set(selection.episode_ids) - set(available))
        if unknown:
            raise SourceSchemaError(
                "Explicit episode list contains unknown demos",
                context={"unknown": unknown, "available": available},
            )
        selected_set = set(selection.episode_ids)
        return [episode_id for episode_id in available if episode_id in selected_set]

    if selection.filter_key is not None:
        mask_group = h5_file.get("mask")
        if not isinstance(mask_group, h5py.Group) or selection.filter_key not in mask_group:
            available_masks = sorted(mask_group.keys()) if isinstance(mask_group, h5py.Group) else []
            raise SourceSchemaError(
                "Unknown robomimic filter key",
                context={"filter_key": selection.filter_key, "available": available_masks},
            )
        raw_ids = mask_group[selection.filter_key][...].tolist()
        mask_ids = [
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            for item in raw_ids
        ]
        duplicates = sorted({item for item in mask_ids if mask_ids.count(item) > 1})
        if duplicates:
            raise SourceSchemaError(
                "robomimic mask contains duplicate demos",
                context={"filter_key": selection.filter_key, "duplicates": duplicates},
            )
        unknown = sorted(set(mask_ids) - set(available))
        if unknown:
            raise SourceSchemaError(
                "robomimic mask references unknown demos",
                context={"filter_key": selection.filter_key, "unknown": unknown},
            )
        mask_set = set(mask_ids)
        return [episode_id for episode_id in available if episode_id in mask_set]

    return available


def _field_selectors(episode_group: Any) -> tuple[str, ...]:
    """Enumerate dataset selectors addressable by frame along their first dimension."""

    h5py = _require_h5py()
    selectors: list[str] = []
    for key, value in episode_group.items():
        if isinstance(value, h5py.Dataset) and len(value.shape) >= 1:
            selectors.append(key)
        elif key in {"obs", "next_obs"} and isinstance(value, h5py.Group):
            nested_names: list[str] = []
            value.visit(nested_names.append)
            for nested_name in nested_names:
                nested_value = value.get(nested_name)
                if isinstance(nested_value, h5py.Dataset) and len(nested_value.shape) >= 1:
                    selectors.append(f"{key}/{nested_name}")
    return tuple(sorted(selectors))
