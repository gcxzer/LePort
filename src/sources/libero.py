"""Source adapter for official LIBERO task HDF5 files.

LIBERO stores several demonstrations for one task in each ``*_demo.hdf5`` file. This adapter also
accepts a flat suite directory containing several task files and exposes globally qualified episode
identifiers so equally named demonstrations never collide across tasks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Integral
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

__all__ = ["LiberoAdapter"]

_TASK_FILE_SUFFIX = "_demo.hdf5"
_DEMO_PATTERN = re.compile(r"demo_(\d+)$")


@dataclass(frozen=True, slots=True)
class _EpisodeInfo:
    """Validated catalog information for one demonstration."""

    public_id: str
    group_name: str
    length: int
    attributes: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TaskInfo:
    """Validated catalog information for one task file."""

    path: Path
    task_name: str
    metadata: dict[str, Any]
    episodes: tuple[_EpisodeInfo, ...]


class LiberoAdapter:
    """Read one official LIBERO task file or a flat directory of task files."""

    name: ClassVar[str] = "libero"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = "libero"

    def probe(self, source: Path) -> ProbeResult:
        """Recognize the complete LIBERO signature without reading frame payloads."""

        h5py = _require_h5py()
        try:
            task_paths = _task_paths(source)
            with h5py.File(task_paths[0], "r") as h5_file:
                task = _read_task_catalog(h5_file, task_paths[0])
        except (OSError, SourceSchemaError) as exc:
            message = exc.message if isinstance(exc, SourceSchemaError) else str(exc)
            return ProbeResult(self.name, 0, message)
        return ProbeResult(
            self.name,
            100,
            f"Detected official LIBERO task structure for {task.task_name!r}",
        )

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Inspect selected demonstrations, task metadata, and cross-task field schemas."""

        h5py = _require_h5py()
        tasks, selected_ids = _selected_catalog(source, selection or EpisodeSelection())
        selected_set = set(selected_ids)
        episode_lengths: dict[str, int] = {}
        field_records: dict[str, list[tuple[str, str, tuple[int, ...], int]]] = {}

        for task in tasks:
            task_episode_ids = {episode.public_id for episode in task.episodes} & selected_set
            if not task_episode_ids:
                continue
            try:
                h5_file = h5py.File(task.path, "r")
            except OSError as exc:
                raise SourceSchemaError(
                    "Could not open LIBERO task file",
                    context={"source": str(task.path), "reason": str(exc)},
                ) from exc
            with h5_file:
                data_group = h5_file["data"]
                for episode in task.episodes:
                    if episode.public_id not in task_episode_ids:
                        continue
                    episode_group = data_group[episode.group_name]
                    episode_lengths[episode.public_id] = episode.length
                    for selector in _field_selectors(episode_group):
                        dataset = episode_group[selector]
                        field_records.setdefault(selector, []).append(
                            (
                                episode.public_id,
                                str(dataset.dtype),
                                tuple(int(size) for size in dataset.shape[1:]),
                                int(dataset.shape[0]),
                            )
                        )

        fields: list[FieldInspection] = []
        diagnostics: list[str] = []
        for selector in sorted(field_records):
            records = field_records[selector]
            present_ids = {record[0] for record in records}
            lengths = {record[0]: record[3] for record in records}
            field = FieldInspection(
                selector=selector,
                dtypes=tuple(sorted({record[1] for record in records})),
                shapes=tuple(sorted({record[2] for record in records})),
                episode_lengths=lengths,
                missing_episodes=tuple(
                    episode_id for episode_id in selected_ids if episode_id not in present_ids
                ),
                image_candidate=(
                    {record[1] for record in records} == {"uint8"}
                    and all(len(record[2]) == 3 for record in records)
                ),
            )
            fields.append(field)
            if not field.schema_consistent:
                diagnostics.append(
                    f"Field {selector!r} has an inconsistent schema across selected demonstrations"
                )
            mismatched = {
                episode_id: field_length
                for episode_id, field_length in lengths.items()
                if field_length != episode_lengths[episode_id]
            }
            if mismatched:
                diagnostics.append(
                    f"Field {selector!r} is not aligned with actions in {sorted(mismatched)!r}"
                )

        return DatasetInspection(
            adapter=self.name,
            source=source,
            episode_ids=tuple(selected_ids),
            episode_lengths=episode_lengths,
            fields=tuple(fields),
            metadata={
                "ordering": "lexical task filename, then numeric demo identifier",
                "task_files": [task.path.name for task in tasks],
                "tasks": {task.task_name: task.metadata for task in tasks},
                "episode_files": {
                    episode.public_id: task.path.name
                    for task in tasks
                    for episode in task.episodes
                    if episode.public_id in selected_set
                },
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
        """Yield selected frames unchanged while owning at most one task-file handle."""

        h5py = _require_h5py()
        tasks, selected_ids = _selected_catalog(source, selection or EpisodeSelection())
        selected_set = set(selected_ids)
        requested_selectors = tuple(dict.fromkeys(selectors)) if selectors is not None else None

        for task in tasks:
            task_episodes = [episode for episode in task.episodes if episode.public_id in selected_set]
            if not task_episodes:
                continue
            try:
                h5_file = h5py.File(task.path, "r")
            except OSError as exc:
                raise SourceSchemaError(
                    "Could not open LIBERO task file",
                    context={"source": str(task.path), "reason": str(exc)},
                ) from exc
            with h5_file:
                data_group = h5_file["data"]
                for episode in task_episodes:
                    episode_group = data_group[episode.group_name]
                    episode_selectors = (
                        requested_selectors
                        if requested_selectors is not None
                        else _field_selectors(episode_group)
                    )
                    datasets: dict[str, Any] = {}
                    for selector in episode_selectors:
                        dataset = episode_group.get(selector)
                        if not isinstance(dataset, h5py.Dataset):
                            raise SourceSchemaError(
                                "Selected LIBERO field is missing or is not a dataset",
                                context={"episode": episode.public_id, "selector": selector},
                            )
                        actual_length = int(dataset.shape[0]) if len(dataset.shape) >= 1 else None
                        if actual_length != episode.length:
                            raise SourceSchemaError(
                                "Selected field length differs from actions; LePort does not alter alignment",
                                context={
                                    "episode": episode.public_id,
                                    "selector": selector,
                                    "actions_length": episode.length,
                                    "field_length": actual_length,
                                },
                            )
                        datasets[selector] = dataset

                    metadata = dict(episode.attributes)
                    metadata.update(task.metadata)
                    # The existing metadata task provider reads this stable, format-independent key.
                    metadata["instruction"] = task.metadata["instruction"]
                    yield SourceEpisode(
                        episode_id=episode.public_id,
                        length=episode.length,
                        frames=(
                            SourceFrame(
                                index=frame_index,
                                fields={
                                    selector: np.asarray(dataset[frame_index])
                                    for selector, dataset in datasets.items()
                                },
                            )
                            for frame_index in range(episode.length)
                        ),
                        metadata=metadata,
                    )


def _require_h5py() -> Any:
    """Import HDF5 support only when the LIBERO adapter is used."""

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "The LIBERO adapter requires HDF5 support; run `uv sync --extra libero`",
            context={"adapter": "libero", "extra": "libero", "dependency": "h5py"},
        ) from exc
    return h5py


def _metadata_value(value: Any) -> Any:
    """Normalize HDF5 scalar and array attributes into JSON-compatible Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    return value


def _task_paths(source: Path) -> tuple[Path, ...]:
    """Resolve one official task file or direct matching children of a suite directory."""

    if source.is_file():
        if not source.name.endswith(_TASK_FILE_SUFFIX):
            raise SourceSchemaError(
                "A LIBERO task filename must end with `_demo.hdf5`",
                context={"source": str(source)},
            )
        return (source,)
    if source.is_dir():
        candidates = tuple(
            sorted(
                (
                    path
                    for path in source.iterdir()
                    if path.is_file() and path.name.endswith(_TASK_FILE_SUFFIX)
                ),
                key=lambda path: path.name,
            )
        )
        if candidates:
            return candidates
        raise SourceSchemaError(
            "LIBERO suite directory contains no direct `*_demo.hdf5` task files",
            context={"source": str(source)},
        )
    raise SourceSchemaError("LIBERO source does not exist", context={"source": str(source)})


def _read_task_catalog(h5_file: Any, path: Path) -> _TaskInfo:
    """Validate task metadata, canonical demos, and declared catalog counts."""

    h5py = _require_h5py()
    data_group = h5_file.get("data")
    if not isinstance(data_group, h5py.Group):
        raise SourceSchemaError(
            "LIBERO task file is missing the root `data` group",
            context={"source": str(path), "path": "data"},
        )

    raw_problem_info = _metadata_value(data_group.attrs.get("problem_info"))
    if not isinstance(raw_problem_info, str):
        raise SourceSchemaError(
            "LIBERO data attributes require JSON text `problem_info`",
            context={"source": str(path), "attribute": "problem_info"},
        )
    try:
        problem_info = json.loads(raw_problem_info)
    except json.JSONDecodeError as exc:
        raise SourceSchemaError(
            "LIBERO `problem_info` is not valid JSON",
            context={"source": str(path), "reason": str(exc)},
        ) from exc
    instruction = problem_info.get("language_instruction") if isinstance(problem_info, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise SourceSchemaError(
            "LIBERO `problem_info` requires a non-empty language_instruction",
            context={"source": str(path), "attribute": "problem_info"},
        )

    bddl_file_name = _metadata_value(data_group.attrs.get("bddl_file_name"))
    bddl_file_content = _metadata_value(data_group.attrs.get("bddl_file_content"))
    if not any(isinstance(value, str) and value.strip() for value in (bddl_file_name, bddl_file_content)):
        raise SourceSchemaError(
            "LIBERO task metadata requires BDDL file identity or content",
            context={"source": str(path)},
        )

    task_name = path.name[: -len(_TASK_FILE_SUFFIX)]
    metadata: dict[str, Any] = {
        "instruction": instruction,
        "task_name": task_name,
        "source_filename": path.name,
        "problem_info": problem_info,
    }
    for attribute in (
        "bddl_file_name",
        "bddl_file_content",
        "macros_image_convention",
        "num_demos",
        "total",
    ):
        if attribute in data_group.attrs:
            metadata[attribute] = _metadata_value(data_group.attrs[attribute])
    if "env_args" in data_group.attrs:
        raw_env_args = _metadata_value(data_group.attrs["env_args"])
        if not isinstance(raw_env_args, str):
            raise SourceSchemaError(
                "LIBERO `env_args` must contain JSON text",
                context={"source": str(path), "attribute": "env_args"},
            )
        try:
            metadata["env_args"] = json.loads(raw_env_args)
        except json.JSONDecodeError as exc:
            raise SourceSchemaError(
                "LIBERO `env_args` is not valid JSON",
                context={"source": str(path), "reason": str(exc)},
            ) from exc

    demos: list[tuple[int, str]] = []
    for name, value in data_group.items():
        if not name.startswith("demo_"):
            continue
        match = _DEMO_PATTERN.fullmatch(name)
        if match is None or name != f"demo_{int(match.group(1))}":
            raise SourceSchemaError(
                "LIBERO demo names must use canonical `demo_<integer>` identifiers",
                context={"source": str(path), "demo": name},
            )
        if not isinstance(value, h5py.Group):
            raise SourceSchemaError(
                "LIBERO demo catalog entry is not an HDF5 group",
                context={"source": str(path), "demo": name},
            )
        demos.append((int(match.group(1)), name))
    if not demos:
        raise SourceSchemaError(
            "LIBERO `data` group contains no canonical demonstrations",
            context={"source": str(path)},
        )

    episodes: list[_EpisodeInfo] = []
    for _, group_name in sorted(demos):
        episode_group = data_group[group_name]
        actions = episode_group.get("actions")
        if not isinstance(actions, h5py.Dataset) or len(actions.shape) < 1:
            raise SourceSchemaError(
                "LIBERO demonstration is missing frame-addressable actions",
                context={"source": str(path), "demo": group_name, "selector": "actions"},
            )
        length = int(actions.shape[0])
        if "num_samples" in episode_group.attrs:
            declared_length = _metadata_value(episode_group.attrs["num_samples"])
            if not isinstance(declared_length, Integral) or int(declared_length) != length:
                raise SourceSchemaError(
                    "LIBERO demo num_samples does not match actions length",
                    context={
                        "source": str(path),
                        "demo": f"{task_name}/{group_name}",
                        "num_samples": declared_length,
                        "actions_length": length,
                    },
                )
        episodes.append(
            _EpisodeInfo(
                public_id=f"{task_name}/{group_name}",
                group_name=group_name,
                length=length,
                attributes={
                    key: _metadata_value(value)
                    for key, value in episode_group.attrs.items()
                    if key != "model_file"
                },
            )
        )

    declared_demos = metadata.get("num_demos")
    if declared_demos is not None and (
        not isinstance(declared_demos, Integral) or int(declared_demos) != len(episodes)
    ):
        raise SourceSchemaError(
            "LIBERO num_demos does not match the task catalog",
            context={"source": str(path), "num_demos": declared_demos, "actual": len(episodes)},
        )
    declared_total = metadata.get("total")
    actual_total = sum(episode.length for episode in episodes)
    if declared_total is not None and (
        not isinstance(declared_total, Integral) or int(declared_total) != actual_total
    ):
        raise SourceSchemaError(
            "LIBERO total does not match summed actions lengths",
            context={"source": str(path), "total": declared_total, "actual": actual_total},
        )
    return _TaskInfo(path=path, task_name=task_name, metadata=metadata, episodes=tuple(episodes))


def _selected_catalog(source: Path, selection: EpisodeSelection) -> tuple[tuple[_TaskInfo, ...], list[str]]:
    """Catalog only relevant task files and return selected IDs in canonical order."""

    if selection.filter_key is not None:
        raise SourceSchemaError(
            "LIBERO does not support robomimic filter keys; use qualified episode IDs",
            context={"filter_key": selection.filter_key},
        )
    all_paths = _task_paths(source)
    paths_by_task = {path.name[: -len(_TASK_FILE_SUFFIX)]: path for path in all_paths}
    if len(paths_by_task) != len(all_paths):
        raise SourceSchemaError(
            "LIBERO suite contains duplicate derived task names",
            context={"source": str(source)},
        )

    requested = set(selection.episode_ids)
    if requested:
        malformed = sorted(
            episode_id
            for episode_id in requested
            if len(episode_id.split("/")) != 2 or _DEMO_PATTERN.fullmatch(episode_id.split("/", 1)[1]) is None
        )
        if malformed:
            raise SourceSchemaError(
                "LIBERO episode IDs must use `<task-name>/demo_<integer>`",
                context={"invalid": malformed},
            )
        requested_tasks = {episode_id.split("/", 1)[0] for episode_id in requested}
        unknown_tasks = sorted(requested_tasks - set(paths_by_task))
        if unknown_tasks:
            raise SourceSchemaError(
                "Explicit LIBERO episode list contains unknown tasks",
                context={"unknown_tasks": unknown_tasks, "available_tasks": sorted(paths_by_task)},
            )
        catalog_paths = tuple(
            path for path in all_paths if path.name[: -len(_TASK_FILE_SUFFIX)] in requested_tasks
        )
    else:
        catalog_paths = all_paths

    h5py = _require_h5py()
    tasks: list[_TaskInfo] = []
    for path in catalog_paths:
        try:
            with h5py.File(path, "r") as h5_file:
                tasks.append(_read_task_catalog(h5_file, path))
        except OSError as exc:
            raise SourceSchemaError(
                "Could not open LIBERO task file",
                context={"source": str(path), "reason": str(exc)},
            ) from exc

    available_ids = [episode.public_id for task in tasks for episode in task.episodes]
    if requested:
        unknown = sorted(requested - set(available_ids))
        if unknown:
            raise SourceSchemaError(
                "Explicit LIBERO episode list contains unknown demonstrations",
                context={"unknown": unknown, "available": available_ids},
            )
        return tuple(tasks), [episode_id for episode_id in available_ids if episode_id in requested]
    return tuple(tasks), available_ids


def _field_selectors(episode_group: Any) -> tuple[str, ...]:
    """Enumerate frame-addressable root datasets and all nested observation leaves."""

    h5py = _require_h5py()
    selectors: list[str] = []
    for name, value in episode_group.items():
        if isinstance(value, h5py.Dataset) and len(value.shape) >= 1:
            selectors.append(name)
        elif name == "obs" and isinstance(value, h5py.Group):
            nested_names: list[str] = []
            value.visit(nested_names.append)
            for nested_name in nested_names:
                nested_value = value.get(nested_name)
                if isinstance(nested_value, h5py.Dataset) and len(nested_value.shape) >= 1:
                    selectors.append(f"obs/{nested_name}")
    return tuple(sorted(selectors))
