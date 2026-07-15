"""Source adapter for standard ALOHA episode HDF5 files.

Each matching file contributes one episode. HDF5 and Pillow stay behind runtime imports so importing
LePort or using another adapter never requires the ALOHA optional dependency group.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from io import BytesIO
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

__all__ = ["AlohaAdapter"]

_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.hdf5$")
_IMAGE_PREFIX = "observations/images/"


class AlohaAdapter:
    """Read one ALOHA episode file or a flat directory of episode files."""

    name: ClassVar[str] = "aloha"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = "aloha"

    def probe(self, source: Path) -> ProbeResult:
        """Recognize ALOHA from required datasets without reading frame payloads."""

        h5py = _require_h5py()
        try:
            episode_files = _resolve_episode_files(source)
        except SourceSchemaError as exc:
            return ProbeResult(self.name, 0, exc.message)

        episode_id, episode_path = episode_files[0]
        try:
            with h5py.File(episode_path, "r") as h5_file:
                _validate_required_structure(h5_file, episode_id)
        except OSError as exc:
            return ProbeResult(self.name, 0, f"Episode file is not readable HDF5: {exc}")
        except SourceSchemaError as exc:
            return ProbeResult(self.name, 0, exc.message)
        return ProbeResult(
            self.name,
            100,
            f"Detected standard ALOHA structure with {len(episode_files)} episode file(s)",
        )

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Inspect schemas, decoded camera shapes, lengths, and per-file metadata."""

        h5py = _require_h5py()
        selected_files = _select_episode_files(source, selection or EpisodeSelection())
        selected_ids = tuple(episode_id for episode_id, _ in selected_files)
        episode_lengths: dict[str, int] = {}
        field_records: dict[str, list[tuple[str, str, tuple[int, ...], int, bool]]] = {}
        episode_attributes: dict[str, dict[str, Any]] = {}
        episode_files: dict[str, str] = {}
        compression_lengths: dict[str, Any] = {}

        for episode_id, episode_path in selected_files:
            try:
                h5_file = h5py.File(episode_path, "r")
            except OSError as exc:
                raise SourceSchemaError(
                    "Could not open ALOHA episode file",
                    context={"episode": episode_id, "source": str(episode_path), "reason": str(exc)},
                ) from exc

            with h5_file:
                actions, _ = _validate_required_structure(h5_file, episode_id)
                action_length = int(actions.shape[0])
                episode_lengths[episode_id] = action_length
                episode_files[episode_id] = episode_path.name
                episode_attributes[episode_id] = {
                    key: _metadata_value(value) for key, value in h5_file.attrs.items()
                }
                compress_len = h5_file.get("compress_len")
                if isinstance(compress_len, h5py.Dataset):
                    compression_lengths[episode_id] = _metadata_value(compress_len[...])

                for selector in _field_selectors(h5_file):
                    dataset = h5_file[selector]
                    field_length = int(dataset.shape[0])
                    image_candidate = selector.startswith(_IMAGE_PREFIX)
                    if image_candidate and field_length > 0:
                        frame = _read_camera_frame(
                            dataset,
                            episode_id=episode_id,
                            selector=selector,
                            frame_index=0,
                        )
                        dtype = str(frame.dtype)
                        shape = tuple(int(size) for size in frame.shape)
                    elif image_candidate:
                        if (
                            len(dataset.shape) != 4
                            or str(dataset.dtype) != "uint8"
                            or int(dataset.shape[-1]) not in {1, 3, 4}
                        ):
                            raise SourceSchemaError(
                                "An empty compressed or unsupported camera field has no decodable schema",
                                context={"episode": episode_id, "selector": selector},
                            )
                        dtype = "uint8"
                        shape = tuple(int(size) for size in dataset.shape[1:])
                    else:
                        dtype = str(dataset.dtype)
                        shape = tuple(int(size) for size in dataset.shape[1:])
                    field_records.setdefault(selector, []).append(
                        (episode_id, dtype, shape, field_length, image_candidate)
                    )

        fields: list[FieldInspection] = []
        diagnostics: list[str] = []
        for selector in sorted(field_records):
            records = field_records[selector]
            present_ids = {record[0] for record in records}
            field = FieldInspection(
                selector=selector,
                dtypes=tuple(sorted({record[1] for record in records})),
                shapes=tuple(sorted({record[2] for record in records})),
                episode_lengths={record[0]: record[3] for record in records},
                missing_episodes=tuple(
                    episode_id for episode_id in selected_ids if episode_id not in present_ids
                ),
                image_candidate=all(record[4] for record in records),
            )
            fields.append(field)
            if not field.schema_consistent:
                diagnostics.append(f"Field {selector!r} has an inconsistent schema across selected episodes")

        metadata: dict[str, Any] = {
            "episode_files": episode_files,
            "episode_attributes": episode_attributes,
        }
        if compression_lengths:
            metadata["compression_lengths"] = compression_lengths
        return DatasetInspection(
            adapter=self.name,
            source=source,
            episode_ids=selected_ids,
            episode_lengths=episode_lengths,
            fields=tuple(fields),
            metadata=metadata,
            diagnostics=tuple(diagnostics),
        )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Open one selected file at a time and read only requested frame fields."""

        h5py = _require_h5py()
        selected_files = _select_episode_files(source, selection or EpisodeSelection())
        requested_selectors = tuple(dict.fromkeys(selectors)) if selectors is not None else None

        for episode_id, episode_path in selected_files:
            try:
                h5_file = h5py.File(episode_path, "r")
            except OSError as exc:
                raise SourceSchemaError(
                    "Could not open ALOHA episode file",
                    context={"episode": episode_id, "source": str(episode_path), "reason": str(exc)},
                ) from exc

            # Suspending this context at yield keeps exactly the current episode handle open. The
            # handle closes when its frames are consumed and the caller requests the next episode.
            with h5_file:
                actions, _ = _validate_required_structure(h5_file, episode_id)
                action_length = int(actions.shape[0])
                episode_selectors = (
                    requested_selectors if requested_selectors is not None else _field_selectors(h5_file)
                )
                datasets: dict[str, Any] = {}
                for selector in episode_selectors:
                    dataset = h5_file.get(selector)
                    if not isinstance(dataset, h5py.Dataset) or len(dataset.shape) < 1:
                        raise SourceSchemaError(
                            "Selected ALOHA field is missing or is not frame-addressable",
                            context={"episode": episode_id, "selector": selector},
                        )
                    field_length = int(dataset.shape[0])
                    if field_length != action_length:
                        raise SourceSchemaError(
                            "Selected field length differs from action; "
                            "LePort does not truncate or pad fields",
                            context={
                                "episode": episode_id,
                                "selector": selector,
                                "action_length": action_length,
                                "field_length": field_length,
                            },
                        )
                    datasets[selector] = dataset

                metadata = {key: _metadata_value(value) for key, value in h5_file.attrs.items()}
                metadata["source_filename"] = episode_path.name
                compress_len = h5_file.get("compress_len")
                if isinstance(compress_len, h5py.Dataset):
                    metadata["compress_len"] = _metadata_value(compress_len[...])

                yield SourceEpisode(
                    episode_id=episode_id,
                    length=action_length,
                    frames=(
                        SourceFrame(
                            index=frame_index,
                            fields={
                                selector: (
                                    _read_camera_frame(
                                        dataset,
                                        episode_id=episode_id,
                                        selector=selector,
                                        frame_index=frame_index,
                                    )
                                    if selector.startswith(_IMAGE_PREFIX)
                                    else np.asarray(dataset[frame_index])
                                )
                                for selector, dataset in datasets.items()
                            },
                        )
                        for frame_index in range(action_length)
                    ),
                    metadata=metadata,
                )


def _require_h5py() -> Any:
    """Import HDF5 support on demand with stable local setup guidance."""

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "The aloha adapter requires HDF5 support; run `uv sync --extra aloha`",
            context={"adapter": "aloha", "extra": "aloha", "dependency": "h5py"},
        ) from exc
    return h5py


def _resolve_episode_files(source: Path) -> list[tuple[str, Path]]:
    """Resolve matching files non-recursively and reject duplicate numeric identifiers."""

    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = [path for path in source.iterdir() if path.is_file()]
    else:
        raise SourceSchemaError(
            "ALOHA source must be a matching episode file or directory",
            context={"source": str(source)},
        )

    matched: list[tuple[int, str, Path]] = []
    seen_numbers: dict[int, str] = {}
    for candidate in candidates:
        match = _EPISODE_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        number = int(match.group(1))
        episode_id = candidate.stem
        if number in seen_numbers:
            raise SourceSchemaError(
                "ALOHA source contains duplicate numeric episode IDs",
                context={
                    "numeric_id": number,
                    "episodes": sorted((seen_numbers[number], episode_id)),
                },
            )
        seen_numbers[number] = episode_id
        matched.append((number, episode_id, candidate))

    if not matched:
        raise SourceSchemaError(
            "ALOHA source contains no `episode_<integer>.hdf5` files",
            context={"source": str(source)},
        )
    return [(episode_id, path) for _, episode_id, path in sorted(matched)]


def _select_episode_files(source: Path, selection: EpisodeSelection) -> list[tuple[str, Path]]:
    """Apply explicit episode selection while retaining canonical numeric order."""

    available_files = _resolve_episode_files(source)
    available_ids = [episode_id for episode_id, _ in available_files]
    if selection.filter_key is not None:
        raise SourceSchemaError(
            "ALOHA sources do not support filter keys because they have no mask table",
            context={"filter_key": selection.filter_key},
        )
    if not selection.episode_ids:
        return available_files
    unknown = sorted(set(selection.episode_ids) - set(available_ids))
    if unknown:
        raise SourceSchemaError(
            "Explicit episode list contains unknown ALOHA IDs",
            context={"unknown": unknown, "available": available_ids},
        )
    selected = set(selection.episode_ids)
    return [item for item in available_files if item[0] in selected]


def _validate_required_structure(h5_file: Any, episode_id: str) -> tuple[Any, Any]:
    """Require frame-addressable action and observations/qpos datasets."""

    h5py = _require_h5py()
    actions = h5_file.get("action")
    if not isinstance(actions, h5py.Dataset) or len(actions.shape) < 1:
        raise SourceSchemaError(
            "ALOHA episode is missing frame-addressable `action`",
            context={"episode": episode_id, "selector": "action"},
        )
    observations = h5_file.get("observations")
    qpos = h5_file.get("observations/qpos")
    if not isinstance(observations, h5py.Group) or not isinstance(qpos, h5py.Dataset) or len(qpos.shape) < 1:
        raise SourceSchemaError(
            "ALOHA episode is missing frame-addressable `observations/qpos`",
            context={"episode": episode_id, "selector": "observations/qpos"},
        )
    return actions, qpos


def _field_selectors(h5_file: Any) -> tuple[str, ...]:
    """Enumerate public root and observation datasets with slash-separated selectors."""

    h5py = _require_h5py()
    selectors: list[str] = []
    for key, value in h5_file.items():
        if key == "compress_len":
            continue
        if isinstance(value, h5py.Dataset) and len(value.shape) >= 1:
            selectors.append(key)
        elif key == "observations" and isinstance(value, h5py.Group):
            nested_names: list[str] = []
            value.visit(nested_names.append)
            for nested_name in nested_names:
                nested_value = value.get(nested_name)
                if (
                    nested_name.rsplit("/", 1)[-1] != "compress_len"
                    and isinstance(nested_value, h5py.Dataset)
                    and len(nested_value.shape) >= 1
                ):
                    selectors.append(f"observations/{nested_name}")
    return tuple(sorted(selectors))


def _read_camera_frame(
    dataset: Any,
    *,
    episode_id: str,
    selector: str,
    frame_index: int,
) -> np.ndarray[Any, Any]:
    """Return raw HWC uint8 data or decode one padded or variable-length JPEG buffer."""

    if len(dataset.shape) == 4:
        if str(dataset.dtype) != "uint8" or int(dataset.shape[-1]) not in {1, 3, 4}:
            raise SourceSchemaError(
                "Raw ALOHA camera must use HWC uint8 frames with 1, 3, or 4 channels",
                context={"episode": episode_id, "frame": frame_index, "selector": selector},
            )
        frame = np.asarray(dataset[frame_index])
        if frame.ndim != 3:
            raise SourceSchemaError(
                "Raw ALOHA camera frame is not three-dimensional",
                context={"episode": episode_id, "frame": frame_index, "selector": selector},
            )
        return frame

    if len(dataset.shape) not in {1, 2}:
        raise SourceSchemaError(
            "Compressed ALOHA camera must contain one byte buffer per frame",
            context={"episode": episode_id, "frame": frame_index, "selector": selector},
        )
    value = dataset[frame_index]
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        buffer = np.asarray(value)
        if buffer.dtype != np.uint8:
            raise SourceSchemaError(
                "Compressed ALOHA camera buffer must contain uint8 bytes",
                context={"episode": episode_id, "frame": frame_index, "selector": selector},
            )
        payload = buffer.reshape(-1).tobytes()
    jpeg_end = payload.find(b"\xff\xd9")
    if jpeg_end >= 0:
        payload = payload[: jpeg_end + 2]

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "The aloha adapter requires JPEG support; run `uv sync --extra aloha`",
            context={"adapter": "aloha", "extra": "aloha", "dependency": "pillow"},
        ) from exc
    try:
        with Image.open(BytesIO(payload)) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise SourceSchemaError(
            "Could not decode ALOHA JPEG camera frame",
            context={
                "episode": episode_id,
                "frame": frame_index,
                "selector": selector,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        ) from exc
    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise SourceSchemaError(
            "Decoded ALOHA JPEG frame is not HWC RGB uint8",
            context={"episode": episode_id, "frame": frame_index, "selector": selector},
        )
    return decoded


def _metadata_value(value: Any) -> Any:
    """Normalize HDF5 scalar and array metadata into JSON-compatible Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    return value
