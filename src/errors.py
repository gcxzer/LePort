"""Stable LePort error types.

The CLI and Python API share these exceptions. Each exception exposes a stable ``code`` so callers
can handle failures without parsing human-readable messages, while ``context`` carries structured
diagnostics such as episode, frame, and selector. Keeping these errors at the package boundary avoids
reverse dependencies between sources, conversion logic, and targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = [
    "AdapterAmbiguousError",
    "AdapterNotFoundError",
    "ConversionError",
    "LePortError",
    "MergeError",
    "OptionalDependencyError",
    "PlanValidationError",
    "ReplayError",
    "SourceSchemaError",
    "TargetValidationError",
]


class LePortError(Exception):
    """Base class for expected LePort failures.

    Args:
        message: Human-readable failure description.
        context: Optional structured context, copied to prevent later caller mutation.
    """

    code: ClassVar[str] = "leport_error"

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        """Store an immutable snapshot of the supplied diagnostic context."""

        super().__init__(message)
        self.message = message
        self.context = dict(context or {})

    def __str__(self) -> str:
        """Render the stable error code, message, and optional diagnostic context."""

        if not self.context:
            return f"[{self.code}] {self.message}"
        rendered_context = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"[{self.code}] {self.message} ({rendered_context})"


class AdapterNotFoundError(LePortError):
    """No registered adapter can read the requested source."""

    code = "adapter_not_found"


class AdapterAmbiguousError(LePortError):
    """Multiple adapters match the source at the same highest confidence."""

    code = "adapter_ambiguous"


class OptionalDependencyError(LePortError):
    """A format-specific optional dependency is unavailable."""

    code = "optional_dependency_missing"


class SourceSchemaError(LePortError):
    """The source structure violates the selected adapter contract."""

    code = "source_schema_error"


class PlanValidationError(LePortError):
    """The conversion plan is incomplete, invalid, or incompatible with the source."""

    code = "plan_validation_error"


class ConversionError(LePortError):
    """Source-frame mapping or target writing failed."""

    code = "conversion_error"


class ReplayError(LePortError):
    """A requested source replay could not produce one validated trajectory pair."""

    code = "replay_error"


class MergeError(LePortError):
    """LeRobot merge arguments, compatibility checks, or output writing failed."""

    code = "merge_error"


class TargetValidationError(LePortError):
    """The written LeRobot dataset failed reload validation."""

    code = "target_validation_error"
