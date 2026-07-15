"""Exclusive boundary for official LeRobot Dataset writing and reload APIs."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.version import Version

from ..conversion.plan import ConversionPlan
from ..errors import ConversionError, MergeError, OptionalDependencyError, TargetValidationError

__all__ = [
    "LeRobotDatasetWriter",
    "MergeResult",
    "ValidationReport",
    "merge_lerobot_datasets",
    "validate_lerobot_dataset",
]

_MIN_LEROBOT = Version("0.6.0")
_MAX_LEROBOT = Version("0.7.0")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured validation summary produced by reloading with the LeRobot API."""

    root: Path
    repo_id: str
    total_episodes: int
    total_frames: int
    episode_lengths: tuple[int, ...]
    tasks: tuple[str, ...]
    features: dict[str, dict[str, Any]]
    decoded_visual_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation result into plain CLI-serializable values."""

        return {
            "root": str(self.root),
            "repo_id": self.repo_id,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "episode_lengths": list(self.episode_lengths),
            "tasks": list(self.tasks),
            "features": self.features,
            "decoded_visual_features": list(self.decoded_visual_features),
        }


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Structured result after multiple LeRobot inputs are merged and committed."""

    sources: tuple[Path, ...]
    target: Path
    validation: ValidationReport

    def to_dict(self) -> dict[str, Any]:
        """Convert the merge result into plain CLI-serializable values."""

        return {
            "sources": [str(source) for source in self.sources],
            "target": str(self.target),
            "total_episodes": self.validation.total_episodes,
            "total_frames": self.validation.total_frames,
            "validation": self.validation.to_dict(),
        }


def validate_lerobot_dataset(
    root: str | Path,
    *,
    repo_id: str | None = None,
    expected_episode_ids: tuple[str, ...] | None = None,
    expected_episode_lengths: tuple[int, ...] | None = None,
    expected_features: dict[str, dict[str, Any]] | None = None,
    expected_tasks: tuple[str, ...] | None = None,
) -> ValidationReport:
    """Reload a dataset with the current LeRobot API and validate structure and visual samples."""

    dataset_class = _lerobot_dataset_class()
    target_root = Path(root)
    resolved_repo_id = repo_id or f"local/{target_root.name}"
    try:
        dataset = dataset_class(repo_id=resolved_repo_id, root=target_root)
        episode_lengths = tuple(
            int(dataset.meta.episodes[index]["length"]) for index in range(dataset.meta.total_episodes)
        )
        tasks = tuple(sorted(str(item) for item in dataset.meta.tasks.index.tolist()))
        features = {name: dict(spec) for name, spec in dataset.meta.features.items()}
        visual_keys = tuple(
            name for name, spec in dataset.meta.features.items() if spec["dtype"] in {"image", "video"}
        )
        decoded: set[str] = set()
        if dataset.meta.total_frames > 0 and visual_keys:
            # Decoding the global first frame and every episode's final frame detects video segments
            # that are shorter than their metadata before the dataset is committed.
            sample_indices = {0}
            for episode_index in range(dataset.meta.total_episodes):
                episode = dataset.meta.episodes[episode_index]
                if int(episode["length"]) > 0:
                    sample_indices.add(int(episode["dataset_to_index"]) - 1)
            for sample_index in sorted(sample_indices):
                sample = dataset[sample_index]
                for key in visual_keys:
                    value = sample[key]
                    actual_shape = tuple(int(size) for size in getattr(value, "shape", ()))
                    metadata_shape = tuple(int(size) for size in dataset.meta.features[key]["shape"])
                    # LeRobot commonly decodes CHW tensors while metadata may record HWC. Both standard
                    # layouts are valid, but dimensions and channel count must match the schema.
                    allowed_shapes = {metadata_shape}
                    if len(metadata_shape) == 3 and metadata_shape[-1] in {1, 3, 4}:
                        allowed_shapes.add((metadata_shape[-1], metadata_shape[0], metadata_shape[1]))
                    if len(metadata_shape) == 3 and metadata_shape[0] in {1, 3, 4}:
                        allowed_shapes.add((metadata_shape[1], metadata_shape[2], metadata_shape[0]))
                    if actual_shape not in allowed_shapes:
                        raise TargetValidationError(
                            "Decoded visual feature dimensions or channels do not match metadata",
                            context={
                                "target": key,
                                "frame": sample_index,
                                "actual_shape": actual_shape,
                                "metadata_shape": metadata_shape,
                                "allowed_shapes": sorted(allowed_shapes),
                            },
                        )
                    decoded.add(key)
    except TargetValidationError:
        raise
    except Exception as exc:
        raise TargetValidationError(
            "LeRobot dataset reload failed",
            context={"root": str(target_root), "repo_id": resolved_repo_id, "reason": str(exc)},
        ) from exc

    report = ValidationReport(
        root=target_root,
        repo_id=resolved_repo_id,
        total_episodes=int(dataset.meta.total_episodes),
        total_frames=int(dataset.meta.total_frames),
        episode_lengths=episode_lengths,
        tasks=tasks,
        features=features,
        decoded_visual_features=tuple(sorted(decoded)),
    )
    if expected_episode_lengths is not None and report.episode_lengths != expected_episode_lengths:
        if expected_episode_ids is not None and len(expected_episode_ids) != len(expected_episode_lengths):
            raise TargetValidationError(
                "Source episode ID count does not match expected length count",
                context={
                    "episode_ids": expected_episode_ids,
                    "episode_lengths": expected_episode_lengths,
                },
            )
        episode_length_mismatches: dict[str, dict[str, int | None]] = {}
        for index in range(max(len(expected_episode_lengths), len(report.episode_lengths))):
            episode_id = (
                expected_episode_ids[index]
                if expected_episode_ids is not None and index < len(expected_episode_ids)
                else f"position-{index}"
            )
            expected_length = (
                expected_episode_lengths[index] if index < len(expected_episode_lengths) else None
            )
            actual_length = report.episode_lengths[index] if index < len(report.episode_lengths) else None
            if expected_length != actual_length:
                episode_length_mismatches[episode_id] = {
                    "expected": expected_length,
                    "actual": actual_length,
                }
        raise TargetValidationError(
            "Target episode lengths do not match source expectations",
            context={"mismatches": episode_length_mismatches},
        )
    if expected_tasks is not None and set(report.tasks) != set(expected_tasks):
        raise TargetValidationError(
            "Target task set does not match plan expectations",
            context={"expected": sorted(expected_tasks), "actual": report.tasks},
        )
    if expected_features is not None:
        mismatches: dict[str, Any] = {}
        for name, expected in expected_features.items():
            actual = report.features.get(name)
            if actual is None:
                mismatches[name] = {"expected": expected, "actual": None}
                continue
            if actual.get("dtype") != expected.get("dtype") or tuple(actual.get("shape", ())) != tuple(
                expected.get("shape", ())
            ):
                mismatches[name] = {"expected": expected, "actual": actual}
        if mismatches:
            raise TargetValidationError(
                "Target feature schema does not match the plan", context={"mismatches": mismatches}
            )
    return report


def merge_lerobot_datasets(
    sources: Sequence[str | Path],
    *,
    target_root: str | Path,
    repo_id: str,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
) -> MergeResult:
    """Merge local datasets with the official LeRobot merger and commit after validation.

    ``sources`` order determines target episode order. Inputs must be complete dataset directories
    that LeRobot can reload. This function never reads HDF5 or modifies an input directory.
    """

    resolved_sources = tuple(Path(source).expanduser().resolve() for source in sources)
    target = Path(target_root).expanduser().resolve()
    resolved_repo_id = repo_id.strip()

    if len(resolved_sources) < 2:
        raise MergeError(
            "Merge requires at least two LeRobot datasets",
            context={"sources": [str(source) for source in resolved_sources]},
        )
    if not resolved_repo_id:
        raise MergeError("repo_id cannot be empty")
    if len(set(resolved_sources)) != len(resolved_sources):
        raise MergeError(
            "Merge inputs cannot contain duplicate directories",
            context={"sources": [str(source) for source in resolved_sources]},
        )
    for source in resolved_sources:
        if source == target:
            raise MergeError(
                "Target directory cannot also be a merge input",
                context={"source": str(source), "target": str(target)},
            )
        if not source.is_dir():
            raise MergeError("Merge input is not an existing directory", context={"source": str(source)})

    if target.exists() and not target.is_dir():
        raise MergeError("Target path exists and is not a directory", context={"target": str(target)})
    if target.exists() and any(target.iterdir()):
        raise MergeError("Target directory exists and is not empty", context={"target": str(target)})

    # Reloading every input before staging prevents invalid inputs from leaving temporary output and
    # provides exact episode-length, task, and feature expectations for the merged result.
    source_reports: list[ValidationReport] = []
    for source in resolved_sources:
        try:
            source_reports.append(validate_lerobot_dataset(source))
        except TargetValidationError as exc:
            raise MergeError(
                "Input LeRobot dataset failed validation",
                context={"source": str(source), "reason": str(exc)},
            ) from exc

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Staging shares the target parent so the final os.replace stays on one filesystem. The dataset
        # child passed to LeRobot remains nonexistent to satisfy its creation contract.
        staging_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.leport-merge-", dir=target.parent))
    except OSError as exc:
        raise MergeError(
            "Could not create merge staging directory",
            context={"target": str(target), "reason": str(exc)},
        ) from exc

    temp_root = staging_root / "dataset"
    try:
        dataset_class = _lerobot_dataset_class()
        try:
            from lerobot.datasets.dataset_tools import merge_datasets

            # repo_id identifies a local dataset while root locates its files. Validation reports supply
            # stable local IDs so callers do not need to maintain a second path-to-ID mapping.
            datasets = [
                dataset_class(repo_id=report.repo_id, root=source)
                for source, report in zip(resolved_sources, source_reports, strict=True)
            ]
            merge_datasets(
                datasets,
                output_repo_id=resolved_repo_id,
                output_dir=temp_root,
                concatenate_videos=concatenate_videos,
                concatenate_data=concatenate_data,
            )
        except Exception as exc:
            raise MergeError(
                "Official LeRobot merge failed",
                context={
                    "sources": [str(source) for source in resolved_sources],
                    "target": str(target),
                    "reason": str(exc),
                },
            ) from exc

        expected_episode_lengths = tuple(
            length for report in source_reports for length in report.episode_lengths
        )
        expected_tasks = tuple(sorted({task for report in source_reports for task in report.tasks}))
        validation = validate_lerobot_dataset(
            temp_root,
            repo_id=resolved_repo_id,
            expected_episode_lengths=expected_episode_lengths,
            expected_features=source_reports[0].features,
            expected_tasks=expected_tasks,
        )

        # Constructing the final report before os.replace ensures no fallible domain logic runs after
        # the irreversible commit, avoiding a committed target reported as failed.
        committed_validation = ValidationReport(
            root=target,
            repo_id=validation.repo_id,
            total_episodes=validation.total_episodes,
            total_frames=validation.total_frames,
            episode_lengths=validation.episode_lengths,
            tasks=validation.tasks,
            features=validation.features,
            decoded_visual_features=validation.decoded_visual_features,
        )
        if target.exists():
            # Only an empty directory can reach this point; non-empty targets are rejected before writes.
            target.rmdir()
        os.replace(temp_root, target)
        # Best-effort cleanup prevents an unrelated staging-directory error from masking a successful
        # commit.
        shutil.rmtree(staging_root, ignore_errors=True)
        return MergeResult(
            sources=resolved_sources,
            target=target,
            validation=committed_validation,
        )
    except Exception:
        # The official merger may have written partial Parquet or video files. Failure removes the private
        # staging container without touching inputs or a non-empty final target.
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


class LeRobotDatasetWriter:
    """Wrap the official writer lifecycle and attach consistent source context to failures."""

    def __init__(self, plan: ConversionPlan, root: Path) -> None:
        """Create a new staged dataset from plan features through the official API."""

        dataset_class = _lerobot_dataset_class()
        try:
            create_options: dict[str, Any] = {}
            if plan.target.use_videos:
                from lerobot.configs.video import RGBEncoderConfig

                # LeRobot 0.6.0 may hang while closing SVT-AV1 subprocesses for many short videos on
                # macOS. libx264 remains inside the official DatasetWriter/PyAV boundary and has broader
                # runtime compatibility.
                create_options["rgb_encoder"] = RGBEncoderConfig(
                    # LeRobot uses the canonical h264 name and resolves it to libx264 through PyAV.
                    vcodec="h264",
                    pix_fmt="yuv420p",
                    g=plan.fps,
                    crf=23,
                    preset="veryfast",
                    video_backend="pyav",
                )
                create_options["video_backend"] = "pyav"
            self._dataset = dataset_class.create(
                repo_id=plan.target.repo_id,
                root=root,
                robot_type=plan.target.robot_type,
                fps=plan.fps,
                features={name: spec.to_lerobot() for name, spec in plan.features.items()},
                use_videos=plan.target.use_videos,
                **create_options,
            )
        except Exception as exc:
            raise ConversionError(
                "Could not create LeRobot target dataset",
                context={"root": str(root), "repo_id": plan.target.repo_id, "reason": str(exc)},
            ) from exc
        self._finalized = False

    def add_frame(self, frame: dict[str, Any], *, episode_id: str, frame_index: int) -> None:
        """Write one mapped frame and attach source episode/frame context on failure."""

        try:
            self._dataset.add_frame(frame)
        except Exception as exc:
            raise ConversionError(
                "LeRobot add_frame failed",
                context={"episode": episode_id, "frame": frame_index, "reason": str(exc)},
            ) from exc

    def save_episode(self, *, episode_id: str) -> None:
        """Commit buffered frames and statistics for the current episode."""

        try:
            self._dataset.save_episode()
        except Exception as exc:
            raise ConversionError(
                "LeRobot save_episode failed",
                context={"episode": episode_id, "reason": str(exc)},
            ) from exc

    def finalize(self) -> None:
        """Finalize once to complete videos, metadata, and indexes."""

        if self._finalized:
            return
        try:
            self._dataset.finalize()
            self._finalized = True
        except Exception as exc:
            raise ConversionError("LeRobot finalize failed", context={"reason": str(exc)}) from exc

    def abort(self) -> None:
        """Release encoder and Parquet resources best-effort before staging cleanup."""

        if self._finalized:
            return
        try:
            writer = getattr(self._dataset, "writer", None)
            if writer is not None:
                writer.cancel_pending_videos()
            self._dataset.finalize()
        except Exception:
            # Cleanup failures must not replace the original conversion error.
            pass
        finally:
            self._finalized = True


def _lerobot_dataset_class() -> Any:
    """Import and validate LeRobot centrally so other modules avoid its internal layout."""

    try:
        installed = Version(metadata.version("lerobot"))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except (metadata.PackageNotFoundError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "Conversion and target validation require the LeRobot Dataset runtime; run `uv sync`",
            context={"dependency": "lerobot[dataset]"},
        ) from exc
    if not _MIN_LEROBOT <= installed < _MAX_LEROBOT:
        raise OptionalDependencyError(
            "Installed LeRobot version is outside the range validated by LePort",
            context={"installed": str(installed), "supported": ">=0.6.0,<0.7.0"},
        )
    return LeRobotDataset
