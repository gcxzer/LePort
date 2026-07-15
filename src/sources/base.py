"""Source Adapter API v1."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from .types import DatasetInspection, EpisodeSelection, ProbeResult, SourceEpisode

__all__ = ["SOURCE_ADAPTER_API_VERSION", "SourceAdapter"]

SOURCE_ADAPTER_API_VERSION = 1


@runtime_checkable
class SourceAdapter(Protocol):
    """Minimum protocol implemented by every source-format adapter.

    ``probe`` and ``inspect`` are read-only. ``iter_episodes`` returns a lazy iterator. A
    ``selectors`` value of ``None`` exposes every source field; otherwise the adapter reads only
    explicitly selected fields so large image datasets avoid unrelated I/O.
    """

    name: ClassVar[str]
    api_version: ClassVar[int]
    extra: ClassVar[str | None]

    def probe(self, source: Path) -> ProbeResult:
        """Identify a likely source format without modifying the source or creating a target."""

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        """Inspect selected episode fields and metadata without reading all frame content."""

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        """Yield episodes lazily in stable order while reading only requested selectors."""
