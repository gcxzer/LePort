"""Target dataset output layer.

Each module writes and validates one target format. Target implementations remain independent of source
readers.
"""

from .lerobot import LeRobotDatasetWriter, ValidationReport, validate_lerobot_dataset

__all__ = ["LeRobotDatasetWriter", "ValidationReport", "validate_lerobot_dataset"]
