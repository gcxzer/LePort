"""Read-only source data models independent of any concrete format.

These types describe the normalized data exposed by every source adapter. Reusing them keeps
format-specific field conventions out of the conversion core.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "DatasetInspection",
    "EpisodeSelection",
    "FieldInspection",
    "ProbeResult",
    "SourceEpisode",
    "SourceFrame",
]


@dataclass(frozen=True, slots=True)
class EpisodeSelection:
    """Mutually exclusive episode selection criteria.

    Empty ``episode_ids`` and a ``None`` ``filter_key`` select every episode. The two selectors cannot
    be combined because adapters could otherwise interpret the same plan differently.
    """

    episode_ids: tuple[str, ...] = ()
    filter_key: str | None = None

    def __post_init__(self) -> None:
        """Reject conflicting or duplicate selectors before adapters interpret them."""

        if self.episode_ids and self.filter_key is not None:
            raise ValueError("episode_ids and filter_key cannot be used together")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode_ids cannot contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        """Return plain values that can be serialized safely to YAML or JSON."""

        return {"episode_ids": list(self.episode_ids), "filter_key": self.filter_key}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Lightweight adapter probe result for one source."""

    adapter: str
    confidence: int
    reason: str

    def __post_init__(self) -> None:
        """Enforce the shared confidence range used by all adapters."""

        if not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")

    @property
    def matched(self) -> bool:
        """Return whether the adapter considers the source a possible match."""

        return self.confidence > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a probe result suitable for CLI serialization."""

        return {
            "adapter": self.adapter,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FieldInspection:
    """Schema summary for one source field across all selected episodes."""

    selector: str
    dtypes: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    episode_lengths: Mapping[str, int]
    missing_episodes: tuple[str, ...] = ()
    image_candidate: bool = False

    def __post_init__(self) -> None:
        """Copy and freeze per-episode lengths so inspection remains read-only."""

        object.__setattr__(self, "episode_lengths", _readonly_mapping(self.episode_lengths))

    @property
    def schema_consistent(self) -> bool:
        """Return whether dtype and single-frame shape match across selected episodes."""

        return len(self.dtypes) == 1 and len(self.shapes) == 1 and not self.missing_episodes

    def to_dict(self) -> dict[str, Any]:
        """Expand tuples and read-only mappings into JSON/YAML-compatible values."""

        return {
            "selector": self.selector,
            "dtypes": list(self.dtypes),
            "shapes": [list(shape) for shape in self.shapes],
            "episode_lengths": dict(self.episode_lengths),
            "missing_episodes": list(self.missing_episodes),
            "image_candidate": self.image_candidate,
            "schema_consistent": self.schema_consistent,
        }


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    """Complete read-only inspection returned by an adapter."""

    adapter: str
    source: Path
    episode_ids: tuple[str, ...]
    episode_lengths: Mapping[str, int]
    fields: tuple[FieldInspection, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze lengths and metadata to preserve the inspection snapshot."""

        object.__setattr__(self, "episode_lengths", _readonly_mapping(self.episode_lengths))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    @property
    def total_frames(self) -> int:
        """Return the total frame count across selected episodes."""

        return sum(int(length) for length in self.episode_lengths.values())

    def field(self, selector: str) -> FieldInspection | None:
        """Find a source field by stable selector, returning ``None`` when absent."""

        return next((item for item in self.fields if item.selector == selector), None)

    def to_dict(self) -> dict[str, Any]:
        """Build an inspection report suitable for readers and machine processing."""

        return {
            "adapter": self.adapter,
            "source": str(self.source),
            "episode_ids": list(self.episode_ids),
            "episode_lengths": dict(self.episode_lengths),
            "total_frames": self.total_frames,
            "fields": [item.to_dict() for item in self.fields],
            "metadata": dict(self.metadata),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One source frame whose field names preserve adapter selectors."""

    index: int
    fields: Mapping[str, Any]
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Freeze the field mapping without copying potentially large NumPy frames."""

        object.__setattr__(self, "fields", _readonly_mapping(self.fields))


@dataclass(frozen=True, slots=True)
class SourceEpisode:
    """A source episode with lazily iterable frames.

    ``frames`` may be a single-use generator or an iterable that reopens the source for each pass.
    Callers must consume the current episode before requesting another one so adapters do not need to
    keep multiple large source handles open.
    """

    episode_id: str
    length: int
    frames: Iterable[SourceFrame]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze episode metadata while preserving lazy frame iteration."""

        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    def iter_frames(self) -> Iterable[SourceFrame]:
        """Return the lazy frame iterable without copying episode data."""

        return iter(self.frames)


def _readonly_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy and freeze a mapping so adapters and callers do not share mutable state."""

    return MappingProxyType(dict(value))
