"""Public interface for source dataset readers.

This package normalizes source data into a shared episode/frame model. It does not map fields or write
LeRobot datasets. Third-party adapter authors should import the stable protocol and domain types here.
"""

from .base import SOURCE_ADAPTER_API_VERSION, SourceAdapter
from .libero import LiberoAdapter
from .registry import AdapterRegistry, create_default_registry
from .types import (
    DatasetInspection,
    EpisodeSelection,
    FieldInspection,
    ProbeResult,
    SourceEpisode,
    SourceFrame,
)
from .umi import UmiAdapter

__all__ = [
    "SOURCE_ADAPTER_API_VERSION",
    "AdapterRegistry",
    "DatasetInspection",
    "EpisodeSelection",
    "FieldInspection",
    "LiberoAdapter",
    "ProbeResult",
    "SourceAdapter",
    "SourceEpisode",
    "SourceFrame",
    "UmiAdapter",
    "create_default_registry",
]
