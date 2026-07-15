"""Registration and deterministic selection for built-in and third-party source adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from ..errors import (
    AdapterAmbiguousError,
    AdapterNotFoundError,
    LePortError,
    OptionalDependencyError,
)
from .base import SOURCE_ADAPTER_API_VERSION, SourceAdapter

__all__ = ["AdapterFactory", "AdapterRegistration", "AdapterRegistry", "create_default_registry"]

AdapterFactory = Callable[[], SourceAdapter]


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    """Lightweight registry entry that does not import an adapter until needed."""

    name: str
    factory: AdapterFactory
    api_version: int
    extra: str | None = None


class AdapterRegistry:
    """Manage adapter factories, plugin diagnostics, and deterministic source selection."""

    def __init__(self) -> None:
        """Create an empty registry with diagnostics for isolated plugin failures."""

        self._registrations: dict[str, AdapterRegistration] = {}
        self.plugin_diagnostics: list[str] = []

    @property
    def names(self) -> tuple[str, ...]:
        """Return adapters by name so probe order is independent of registration order."""

        return tuple(sorted(self._registrations))

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        *,
        api_version: int = SOURCE_ADAPTER_API_VERSION,
        extra: str | None = None,
    ) -> None:
        """Register a lazy factory and reject duplicate names or incompatible APIs immediately."""

        if api_version != SOURCE_ADAPTER_API_VERSION:
            raise ValueError(
                f"Adapter {name!r} uses API v{api_version}; only v{SOURCE_ADAPTER_API_VERSION} is supported"
            )
        if name in self._registrations:
            raise ValueError(f"Duplicate adapter name: {name}")
        self._registrations[name] = AdapterRegistration(name, factory, api_version, extra)

    def get(self, name: str) -> SourceAdapter:
        """Instantiate an adapter by name and verify its declarations against the registry entry."""

        registration = self._registrations.get(name)
        if registration is None:
            raise AdapterNotFoundError(
                f"Adapter {name!r} is not registered",
                context={"available": self.names},
            )
        adapter = registration.factory()
        if not isinstance(adapter, SourceAdapter):
            raise AdapterNotFoundError(
                f"Adapter {name!r} does not implement SourceAdapter API v1",
                context={"adapter": name},
            )
        if adapter.name != name or adapter.api_version != registration.api_version:
            raise AdapterNotFoundError(
                f"Adapter {name!r} declarations do not match its registry entry",
                context={
                    "registered_name": name,
                    "instance_name": adapter.name,
                    "registered_api": registration.api_version,
                    "instance_api": adapter.api_version,
                },
            )
        return adapter

    def select(self, source: Path, *, name: str | None = None) -> SourceAdapter:
        """Select an adapter explicitly or by a unique highest confidence score."""

        if name is not None:
            return self.get(name)

        matched: list[tuple[int, str, SourceAdapter]] = []
        attempts: dict[str, str] = {}
        for adapter_name in self.names:
            try:
                adapter = self.get(adapter_name)
                result = adapter.probe(source)
                attempts[adapter_name] = result.reason
                if result.matched:
                    matched.append((result.confidence, adapter_name, adapter))
            except OptionalDependencyError as exc:
                attempts[adapter_name] = str(exc)
            except LePortError as exc:
                attempts[adapter_name] = str(exc)
            except Exception as exc:  # A broken plugin must not prevent other adapters from probing.
                attempts[adapter_name] = f"probe failed: {type(exc).__name__}: {exc}"

        if not matched:
            raise AdapterNotFoundError(
                f"No adapter can read {source}",
                context={"attempts": attempts},
            )

        highest = max(item[0] for item in matched)
        winners = [(adapter_name, adapter) for score, adapter_name, adapter in matched if score == highest]
        if len(winners) != 1:
            raise AdapterAmbiguousError(
                f"Multiple adapters match {source} with the same confidence",
                context={"confidence": highest, "candidates": [item[0] for item in winners]},
            )
        return winners[0][1]

    def discover_plugins(self) -> None:
        """Load ``leport.source_adapters`` entry points while isolating individual failures."""

        for entry_point in metadata.entry_points(group="leport.source_adapters"):
            if entry_point.name in self._registrations:
                self.plugin_diagnostics.append(
                    f"Skipped plugin {entry_point.name!r}: adapter name is already registered"
                )
                continue
            try:
                loaded: Any = entry_point.load()
                factory: AdapterFactory
                if isinstance(loaded, SourceAdapter):
                    # Instance entry points are stateless and may be requested repeatedly, so the
                    # registry factory returns the same loaded instance.
                    def loaded_factory(loaded_adapter: SourceAdapter = loaded) -> SourceAdapter:
                        """Wrap a stateless adapter instance exported directly by an entry point."""

                        return loaded_adapter

                    factory = loaded_factory
                elif callable(loaded):
                    factory = loaded
                else:
                    raise TypeError("entry point must return an adapter instance or a zero-argument factory")

                adapter = factory()
                if not isinstance(adapter, SourceAdapter):
                    raise TypeError("plugin does not implement SourceAdapter API v1")
                if adapter.api_version != SOURCE_ADAPTER_API_VERSION:
                    raise ValueError(
                        f"plugin API v{adapter.api_version} is incompatible; "
                        f"v{SOURCE_ADAPTER_API_VERSION} is supported"
                    )
                if adapter.name != entry_point.name:
                    raise ValueError(
                        f"entry point name {entry_point.name!r} does not match adapter name {adapter.name!r}"
                    )
                self.register(
                    adapter.name,
                    factory,
                    api_version=adapter.api_version,
                    extra=adapter.extra,
                )
            except Exception as exc:
                self.plugin_diagnostics.append(
                    f"Skipped plugin {entry_point.name!r}: {type(exc).__name__}: {exc}"
                )


def create_default_registry(*, discover_plugins: bool = True) -> AdapterRegistry:
    """Create a new registry containing the built-in source adapters.

    A fresh registry lets tests and embedded callers add temporary adapters without shared state.
    Built-in modules do not import format dependencies here, so their extras do not affect core
    imports or adapters for other formats.
    """

    registry = AdapterRegistry()

    def aloha_factory() -> SourceAdapter:
        """Import the ALOHA adapter only after registry selection requests it."""

        from .aloha import AlohaAdapter

        return AlohaAdapter()

    def robomimic_factory() -> SourceAdapter:
        """Import the built-in adapter only when selected to preserve dependency isolation."""

        from .robomimic import RobomimicAdapter

        return RobomimicAdapter()

    def maniskill_factory() -> SourceAdapter:
        """Import the paired trajectory adapter without importing HDF5 during core startup."""

        from .maniskill import ManiSkillAdapter

        return ManiSkillAdapter()

    def libero_factory() -> SourceAdapter:
        """Import the LIBERO adapter only when registry selection requests it."""

        from .libero import LiberoAdapter

        return LiberoAdapter()

    def umi_factory() -> SourceAdapter:
        """Import the UMI adapter without loading its storage dependencies during core startup."""

        from .umi import UmiAdapter

        return UmiAdapter()

    registry.register("aloha", aloha_factory, extra="aloha")
    registry.register("libero", libero_factory, extra="libero")
    registry.register("maniskill", maniskill_factory, extra="maniskill")
    registry.register("robomimic", robomimic_factory, extra="robomimic")
    registry.register("umi", umi_factory, extra="umi")
    if discover_plugins:
        registry.discover_plugins()
    return registry
