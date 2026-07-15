"""Shared orchestration for inspection, preflight, conversion, and validation."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConversionError, PlanValidationError, TargetValidationError
from ..sources.registry import AdapterRegistry, create_default_registry
from ..sources.types import DatasetInspection
from ..targets.lerobot import LeRobotDatasetWriter, ValidationReport, validate_lerobot_dataset
from .mapping import map_frame, resolve_task
from .plan import ConversionPlan

__all__ = ["ConversionResult", "PreflightReport", "convert_dataset", "preflight"]


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Deterministic expectations derived from every selected episode before writing."""

    inspection: DatasetInspection
    episode_lengths: tuple[int, ...]
    tasks: tuple[str, ...]

    @property
    def total_frames(self) -> int:
        """Return the total source frame count confirmed by preflight."""

        return sum(self.episode_lengths)


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Conversion result after a successful atomic commit."""

    target: Path
    preflight: PreflightReport
    validation: ValidationReport


def preflight(
    plan: ConversionPlan,
    *,
    registry: AdapterRegistry | None = None,
) -> PreflightReport:
    """Validate every episode schema and map the first frame of each episode."""

    plan.validate()
    resolved_registry = registry or create_default_registry()
    adapter = resolved_registry.select(plan.source, name=plan.adapter)
    inspection = adapter.inspect(plan.source, selection=plan.selection)
    if not inspection.episode_ids:
        raise PlanValidationError("The plan does not select any episodes")

    for selector in plan.source_selectors:
        field = inspection.field(selector)
        if field is None:
            raise PlanValidationError(
                "The plan references a missing source field", context={"selector": selector}
            )
        if field.missing_episodes or len(field.dtypes) != 1 or len(field.shapes) != 1:
            raise PlanValidationError(
                "A source field referenced by the plan has an inconsistent episode schema",
                context={
                    "selector": selector,
                    "dtypes": field.dtypes,
                    "shapes": field.shapes,
                    "missing_episodes": field.missing_episodes,
                },
            )
        mismatched_lengths = {
            episode_id: length
            for episode_id, length in field.episode_lengths.items()
            if length != inspection.episode_lengths[episode_id]
        }
        if mismatched_lengths:
            raise PlanValidationError(
                "A source field referenced by the plan has a different length from actions",
                context={"selector": selector, "lengths": mismatched_lengths},
            )

    tasks: list[str] = []
    seen_episodes: list[str] = []
    for episode in adapter.iter_episodes(
        plan.source,
        selection=plan.selection,
        selectors=plan.source_selectors,
    ):
        if episode.length <= 0:
            raise PlanValidationError(
                "Empty episodes are not supported", context={"episode": episode.episode_id}
            )
        frame_iterator = iter(episode.iter_frames())
        try:
            first_frame = next(frame_iterator)
        except StopIteration as exc:
            raise PlanValidationError(
                "Episode declares a non-zero length but exposes no readable frames",
                context={"episode": episode.episode_id, "length": episode.length},
            ) from exc
        map_frame(plan, episode, first_frame)
        tasks.append(resolve_task(plan, episode))
        seen_episodes.append(episode.episode_id)

    if tuple(seen_episodes) != inspection.episode_ids:
        raise PlanValidationError(
            "Episode order differs between inspect and iter_episodes",
            context={"inspected": inspection.episode_ids, "iterated": seen_episodes},
        )
    return PreflightReport(
        inspection=inspection,
        episode_lengths=tuple(inspection.episode_lengths[item] for item in inspection.episode_ids),
        tasks=tuple(sorted(set(tasks))),
    )


def convert_dataset(
    plan: ConversionPlan,
    *,
    registry: AdapterRegistry | None = None,
) -> ConversionResult:
    """Stream conversion output and commit the final directory only after reload validation."""

    resolved_registry = registry or create_default_registry()
    report = preflight(plan, registry=resolved_registry)
    target = plan.target.root
    if target.exists() and not target.is_dir():
        raise ConversionError("Target path exists and is not a directory", context={"target": str(target)})
    if target.exists() and any(target.iterdir()):
        raise ConversionError("Target directory exists and is not empty", context={"target": str(target)})
    target.parent.mkdir(parents=True, exist_ok=True)
    # LeRobotDataset.create requires a nonexistent root. The staging container may exist, but its
    # dataset child must not. Keeping staging beside the target allows an atomic os.replace commit.
    staging_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.leport-", dir=target.parent))
    temp_root = staging_root / "dataset"
    writer: LeRobotDatasetWriter | None = None

    try:
        writer = LeRobotDatasetWriter(plan, temp_root)
        adapter = resolved_registry.select(plan.source, name=plan.adapter)
        converted_ids: list[str] = []
        for episode in adapter.iter_episodes(
            plan.source,
            selection=plan.selection,
            selectors=plan.source_selectors,
        ):
            frame_count = 0
            for frame in episode.iter_frames():
                writer.add_frame(
                    map_frame(plan, episode, frame),
                    episode_id=episode.episode_id,
                    frame_index=frame.index,
                )
                frame_count += 1
            if frame_count != episode.length:
                raise ConversionError(
                    "Adapter frame count does not match the declared episode length",
                    context={
                        "episode": episode.episode_id,
                        "expected": episode.length,
                        "actual": frame_count,
                    },
                )
            writer.save_episode(episode_id=episode.episode_id)
            converted_ids.append(episode.episode_id)
        writer.finalize()

        if tuple(converted_ids) != report.inspection.episode_ids:
            raise TargetValidationError(
                "Converted episode order differs from preflight",
                context={"expected": report.inspection.episode_ids, "actual": converted_ids},
            )
        validation = validate_lerobot_dataset(
            temp_root,
            repo_id=plan.target.repo_id,
            expected_episode_ids=report.inspection.episode_ids,
            expected_episode_lengths=report.episode_lengths,
            expected_features={name: spec.to_lerobot() for name, spec in plan.features.items()},
            expected_tasks=report.tasks,
        )

        if target.exists():
            target.rmdir()
        os.replace(temp_root, target)
        staging_root.rmdir()
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
        return ConversionResult(target=target, preflight=report, validation=committed_validation)
    except Exception:
        if writer is not None:
            writer.abort()
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
