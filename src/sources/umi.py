"""Source adapter for processed Universal Manipulation Interface Zarr replay buffers.

The official UMI processing pipeline stores all frame arrays in a flat ``data`` group and cumulative
episode boundaries in ``meta/episode_ends``. Zarr and imagecodecs remain runtime imports so unrelated
LePort workflows do not require the UMI optional dependency group.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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

__all__ = ["UmiAdapter"]

_REQUIRED_ROBOT_FIELDS = {
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (3,),
    "robot0_gripper_width": (1,),
}
_CAMERA_FIELD_PATTERN = re.compile(r"camera\d+_rgb$")


@dataclass(frozen=True, slots=True)
class _EpisodeInfo:
    """One deterministic episode slice derived from cumulative replay-buffer boundaries."""

    episode_id: str
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of frames in the half-open global slice."""

        return self.end - self.start


@dataclass(frozen=True, slots=True)
class _Catalog:
    """Validated UMI metadata that can be reused without retaining frame payloads."""

    episodes: tuple[_EpisodeInfo, ...]
    field_names: tuple[str, ...]

    @property
    def total_frames(self) -> int:
        """Return the final cumulative boundary after validation."""

        return self.episodes[-1].end


class UmiAdapter:
    """Read processed UMI Zarr v2 ZipStore files and directory stores."""

    name: ClassVar[str] = "umi"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = "umi"

    def probe(self, source: Path) -> ProbeResult:
        """Recognize the UMI-specific replay-buffer signature without reading frame payloads."""

        try:
            with _open_umi_group(source) as (zarr, root, store_kind):
                catalog = _validated_catalog(zarr, root)
        except SourceSchemaError as exc:
            return ProbeResult(self.name, 0, exc.message)
        return ProbeResult(
            self.name,
            100,
            f"Detected processed UMI {store_kind} with {len(catalog.episodes)} episode(s)",
        )

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Inspect replay-buffer array metadata and episode boundaries without decoding frames."""

        with _open_umi_group(source) as (zarr, root, store_kind):
            catalog = _validated_catalog(zarr, root)
            selected = _selected_episodes(catalog, selection or EpisodeSelection())
            data_group = root["data"]
            episode_lengths = {episode.episode_id: episode.length for episode in selected}
            fields = tuple(
                FieldInspection(
                    selector=field_name,
                    dtypes=(str(data_group[field_name].dtype),),
                    shapes=(tuple(int(size) for size in data_group[field_name].shape[1:]),),
                    episode_lengths=episode_lengths,
                    image_candidate=_is_image_array(data_group[field_name]),
                )
                for field_name in catalog.field_names
            )
            return DatasetInspection(
                adapter=self.name,
                source=source,
                episode_ids=tuple(episode.episode_id for episode in selected),
                episode_lengths=episode_lengths,
                fields=fields,
                metadata={
                    "store_kind": store_kind,
                    "ordering": "cumulative meta/episode_ends order",
                    "total_frames": catalog.total_frames,
                    "episode_slices": {
                        episode.episode_id: [episode.start, episode.end] for episode in selected
                    },
                },
            )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Yield selected UMI slices while indexing only requested Zarr arrays frame by frame."""

        with _open_umi_group(source) as (zarr, root, _store_kind):
            catalog = _validated_catalog(zarr, root)
            selected = _selected_episodes(catalog, selection or EpisodeSelection())
            requested = (
                tuple(dict.fromkeys(selectors)) if selectors is not None else catalog.field_names
            )
            unknown = tuple(selector for selector in requested if selector not in catalog.field_names)
            if unknown:
                raise SourceSchemaError(
                    "Selected UMI field is missing",
                    context={"selectors": unknown, "available": catalog.field_names},
                )
            arrays = {selector: root["data"][selector] for selector in requested}

            for episode in selected:
                yield SourceEpisode(
                    episode_id=episode.episode_id,
                    length=episode.length,
                    frames=(
                        SourceFrame(
                            index=frame_index,
                            fields={
                                selector: np.asarray(array[episode.start + frame_index])
                                for selector, array in arrays.items()
                            },
                        )
                        for frame_index in range(episode.length)
                    ),
                    metadata={
                        "episode_index": episode.index,
                        "source_start": episode.start,
                        "source_end": episode.end,
                    },
                )


def _require_zarr() -> Any:
    """Load Zarr v2 and register the codec identifier used by official UMI RGB arrays."""

    try:
        import zarr
        from imagecodecs.numcodecs import register_codecs
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "The UMI adapter requires Zarr v2 and image codecs; run `uv sync --extra umi`",
            context={"adapter": "umi", "extra": "umi", "dependency": "zarr,imagecodecs"},
        ) from exc
    register_codecs()
    return zarr


@contextmanager
def _open_umi_group(source: Path) -> Iterator[tuple[Any, Any, str]]:
    """Open one read-only Zarr store and close its filesystem resources deterministically."""

    zarr = _require_zarr()
    if source.is_file():
        store_kind = "ZipStore"
        try:
            store = zarr.ZipStore(str(source), mode="r")
        except Exception as exc:
            raise SourceSchemaError(
                "Could not open UMI source as a Zarr ZipStore",
                context={"source": str(source), "reason": f"{type(exc).__name__}: {exc}"},
            ) from exc
    elif source.is_dir():
        store_kind = "DirectoryStore"
        store = zarr.DirectoryStore(str(source))
    else:
        raise SourceSchemaError(
            "UMI source must be an existing Zarr ZipStore file or directory store",
            context={"source": str(source)},
        )

    try:
        try:
            root = zarr.open_group(store=store, mode="r")
        except Exception as exc:
            raise SourceSchemaError(
                "Could not open UMI Zarr root group",
                context={"source": str(source), "reason": f"{type(exc).__name__}: {exc}"},
            ) from exc
        yield zarr, root, store_kind
    finally:
        store.close()


def _validated_catalog(zarr: Any, root: Any) -> _Catalog:
    """Validate the UMI signature, cumulative boundaries, and every flat data array."""

    try:
        data_group = root["data"]
        meta_group = root["meta"]
    except KeyError as exc:
        raise SourceSchemaError(
            "Processed UMI stores require `data` and `meta` groups",
            context={"missing": str(exc)},
        ) from exc
    if not isinstance(data_group, zarr.hierarchy.Group) or not isinstance(
        meta_group, zarr.hierarchy.Group
    ):
        raise SourceSchemaError("Processed UMI `data` and `meta` members must be Zarr groups")

    try:
        episode_ends_array = meta_group["episode_ends"]
    except KeyError as exc:
        raise SourceSchemaError("Processed UMI stores require `meta/episode_ends`") from exc
    if not isinstance(episode_ends_array, zarr.core.Array):
        raise SourceSchemaError("UMI `meta/episode_ends` must be a Zarr array")

    episode_ends = np.asarray(episode_ends_array[:])
    if episode_ends.ndim != 1 or episode_ends.size == 0 or not np.issubdtype(
        episode_ends.dtype, np.integer
    ):
        raise SourceSchemaError(
            "UMI `meta/episode_ends` must be a non-empty one-dimensional integer array",
            context={"dtype": str(episode_ends.dtype), "shape": episode_ends.shape},
        )
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise SourceSchemaError(
            "UMI `meta/episode_ends` must contain strictly increasing positive boundaries",
            context={"episode_ends": episode_ends.tolist()},
        )

    nested_groups = tuple(sorted(data_group.group_keys()))
    if nested_groups:
        raise SourceSchemaError(
            "Processed UMI `data` must contain only flat frame arrays",
            context={"groups": nested_groups},
        )
    field_names = tuple(sorted(data_group.array_keys()))
    total_frames = int(episode_ends[-1])
    for field_name in field_names:
        array = data_group[field_name]
        if len(array.shape) == 0:
            raise SourceSchemaError(
                "Processed UMI fields must be frame-addressable",
                context={"selector": field_name, "shape": array.shape},
            )
        if int(array.shape[0]) != total_frames:
            raise SourceSchemaError(
                "UMI field length differs from the final episode boundary",
                context={
                    "selector": field_name,
                    "expected_length": total_frames,
                    "field_length": int(array.shape[0]),
                },
            )

    missing_robot_fields = tuple(
        field_name for field_name in _REQUIRED_ROBOT_FIELDS if field_name not in field_names
    )
    if missing_robot_fields:
        raise SourceSchemaError(
            "Processed UMI store is missing required robot fields",
            context={"missing": missing_robot_fields},
        )
    for field_name, expected_shape in _REQUIRED_ROBOT_FIELDS.items():
        array = data_group[field_name]
        frame_shape = tuple(int(size) for size in array.shape[1:])
        if frame_shape != expected_shape or not np.issubdtype(array.dtype, np.number):
            raise SourceSchemaError(
                "Processed UMI robot field has an invalid schema",
                context={
                    "selector": field_name,
                    "expected_shape": expected_shape,
                    "shape": frame_shape,
                    "dtype": str(array.dtype),
                },
            )

    camera_fields = tuple(
        field_name for field_name in field_names if _CAMERA_FIELD_PATTERN.fullmatch(field_name)
    )
    if not camera_fields:
        raise SourceSchemaError("Processed UMI store requires at least one `camera<integer>_rgb` field")
    for field_name in camera_fields:
        array = data_group[field_name]
        if str(array.dtype) != "uint8" or len(array.shape) != 4 or int(array.shape[-1]) != 3:
            raise SourceSchemaError(
                "Processed UMI RGB field must use HWC uint8 frames",
                context={"selector": field_name, "dtype": str(array.dtype), "shape": array.shape},
            )

    starts = np.concatenate((np.asarray([0], dtype=episode_ends.dtype), episode_ends[:-1]))
    episodes = tuple(
        _EpisodeInfo(
            episode_id=f"episode_{index}",
            index=index,
            start=int(start),
            end=int(end),
        )
        for index, (start, end) in enumerate(zip(starts, episode_ends, strict=True))
    )
    return _Catalog(episodes=episodes, field_names=field_names)


def _selected_episodes(catalog: _Catalog, selection: EpisodeSelection) -> tuple[_EpisodeInfo, ...]:
    """Apply supported selection modes while retaining cumulative source order."""

    available = tuple(episode.episode_id for episode in catalog.episodes)
    if selection.filter_key is not None:
        raise SourceSchemaError(
            "UMI replay buffers do not provide filter keys",
            context={"filter_key": selection.filter_key},
        )
    if not selection.episode_ids:
        return catalog.episodes
    unknown = tuple(sorted(set(selection.episode_ids) - set(available)))
    if unknown:
        raise SourceSchemaError(
            "UMI episode selection contains unknown identifiers",
            context={"unknown": unknown, "available": available},
        )
    requested = set(selection.episode_ids)
    return tuple(episode for episode in catalog.episodes if episode.episode_id in requested)


def _is_image_array(array: Any) -> bool:
    """Identify conventional decoded HWC image arrays from metadata only."""

    return (
        str(array.dtype) == "uint8"
        and len(array.shape) == 4
        and int(array.shape[-1]) in {1, 3, 4}
    )
