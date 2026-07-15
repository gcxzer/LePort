"""Test the Source Adapter protocol, registry, and plugin isolation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from leport.errors import AdapterAmbiguousError, AdapterNotFoundError, OptionalDependencyError
from leport.sources.base import SOURCE_ADAPTER_API_VERSION
from leport.sources.registry import AdapterRegistry
from leport.sources.types import (
    DatasetInspection,
    EpisodeSelection,
    FieldInspection,
    ProbeResult,
    SourceEpisode,
    SourceFrame,
)


class ArbitraryAdapter:
    """Use field names without state/action terminology to verify schema neutrality."""

    name: ClassVar[str] = "arbitrary"
    api_version: ClassVar[int] = SOURCE_ADAPTER_API_VERSION
    extra: ClassVar[str | None] = None

    def __init__(self, confidence: int = 80) -> None:
        self.confidence = confidence

    def probe(self, source: Path) -> ProbeResult:
        return ProbeResult(self.name, self.confidence, "synthetic match")

    def inspect(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
    ) -> DatasetInspection:
        del selection
        return DatasetInspection(
            adapter=self.name,
            source=source,
            episode_ids=("segment-a",),
            episode_lengths={"segment-a": 1},
            fields=(
                FieldInspection(
                    selector="sensor/custom",
                    dtypes=("float32",),
                    shapes=((2,),),
                    episode_lengths={"segment-a": 1},
                ),
            ),
        )

    def iter_episodes(
        self,
        source: Path,
        *,
        selection: EpisodeSelection | None = None,
        selectors: Sequence[str] | None = None,
    ) -> Iterator[SourceEpisode]:
        del source, selection, selectors
        yield SourceEpisode(
            episode_id="segment-a",
            length=1,
            frames=(SourceFrame(0, {"sensor/custom": [1.0, 2.0]}),),
        )


class SecondAdapter(ArbitraryAdapter):
    name: ClassVar[str] = "second"


class MissingDependencyAdapter(ArbitraryAdapter):
    name: ClassVar[str] = "missing-dependency"

    def probe(self, source: Path) -> ProbeResult:
        del source
        raise OptionalDependencyError("dependency missing")


def test_registry_selects_unique_highest_confidence(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("arbitrary", lambda: ArbitraryAdapter(90))
    registry.register("second", lambda: SecondAdapter(30))
    assert registry.select(tmp_path / "source").name == "arbitrary"


def test_registry_reports_ambiguous_highest_confidence(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("arbitrary", lambda: ArbitraryAdapter(80))
    registry.register("second", lambda: SecondAdapter(80))
    with pytest.raises(AdapterAmbiguousError) as error:
        registry.select(tmp_path / "source")
    assert error.value.context["candidates"] == ["arbitrary", "second"]


def test_registry_reports_all_failed_attempts(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("arbitrary", lambda: ArbitraryAdapter(0))
    registry.register("missing-dependency", MissingDependencyAdapter)
    with pytest.raises(AdapterNotFoundError) as error:
        registry.select(tmp_path / "source")
    assert set(error.value.context["attempts"]) == {"arbitrary", "missing-dependency"}


def test_missing_optional_dependency_does_not_block_other_adapter(tmp_path: Path) -> None:
    registry = AdapterRegistry()
    registry.register("missing-dependency", MissingDependencyAdapter)
    registry.register("arbitrary", ArbitraryAdapter)
    assert registry.select(tmp_path / "source").name == "arbitrary"


def test_domain_model_preserves_arbitrary_fields(tmp_path: Path) -> None:
    inspection = ArbitraryAdapter().inspect(tmp_path / "source")
    episode = next(ArbitraryAdapter().iter_episodes(tmp_path / "source"))
    frame = next(iter(episode.iter_frames()))
    assert inspection.fields[0].selector == "sensor/custom"
    assert frame.fields["sensor/custom"] == [1.0, 2.0]


def test_plugin_failures_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incompatible plugin cannot prevent a compatible plugin from registering."""

    class FakeEntryPoint:
        def __init__(self, name: str, loaded: object) -> None:
            self.name = name
            self._loaded = loaded

        def load(self) -> object:
            return self._loaded

    class IncompatibleAdapter(ArbitraryAdapter):
        name: ClassVar[str] = "bad"
        api_version: ClassVar[int] = 99

    monkeypatch.setattr(
        "leport.sources.registry.metadata.entry_points",
        lambda **kwargs: [
            FakeEntryPoint("bad", IncompatibleAdapter),
            FakeEntryPoint("arbitrary", ArbitraryAdapter),
        ],
    )
    registry = AdapterRegistry()
    registry.discover_plugins()
    assert registry.names == ("arbitrary",)
    assert any("API v99" in diagnostic for diagnostic in registry.plugin_diagnostics)
