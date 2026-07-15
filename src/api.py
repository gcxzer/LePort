"""Thin public API for Python callers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .conversion.pipeline import ConversionResult, convert_dataset, preflight
from .conversion.plan import (
    ConversionPlan,
    FeatureMapping,
    FeatureSpec,
    TargetConfig,
    TaskProvider,
    load_plan,
)
from .errors import PlanValidationError
from .maniskill_replay import (
    ManiSkillReplayOptions,
    ManiSkillReplayResult,
    run_maniskill_replay,
)
from .sources.registry import AdapterRegistry, create_default_registry
from .sources.types import DatasetInspection, EpisodeSelection, FieldInspection
from .targets.lerobot import (
    MergeResult,
    ValidationReport,
    merge_lerobot_datasets,
    validate_lerobot_dataset,
)

__all__ = ["convert", "create_plan", "inspect", "merge", "replay_maniskill", "validate"]


def inspect(
    source: str | Path,
    *,
    adapter: str | None = None,
    selection: EpisodeSelection | None = None,
    registry: AdapterRegistry | None = None,
) -> DatasetInspection:
    """Inspect a source without creating a plan or target directory."""

    source_path = Path(source)
    resolved_registry = registry or create_default_registry()
    resolved_adapter = resolved_registry.select(source_path, name=adapter)
    return resolved_adapter.inspect(source_path, selection=selection)


def create_plan(
    source: str | Path,
    *,
    target_root: str | Path,
    repo_id: str,
    fps: int,
    action_source: str,
    task: str | None = None,
    task_metadata: str | None = None,
    state_sources: Sequence[str] = (),
    image_sources: Mapping[str, str] | None = None,
    action_dtype: str | None = None,
    state_dtype: str | None = None,
    robot_type: str | None = None,
    use_videos: bool = True,
    adapter: str | None = None,
    selection: EpisodeSelection | None = None,
    registry: AdapterRegistry | None = None,
) -> ConversionPlan:
    """Build a conversion plan from explicit arguments and source inspection.

    The caller must provide ``action_source`` because field names and shapes do not establish action
    semantics. Each ``image_sources`` key is a source selector and each value is a complete target
    feature name.
    """

    if (task is None) == (task_metadata is None):
        raise PlanValidationError("Provide exactly one of task and task_metadata")
    resolved_registry = registry or create_default_registry()
    source_path = Path(source)
    resolved_adapter = resolved_registry.select(source_path, name=adapter)
    resolved_selection = selection or EpisodeSelection()
    inspection = resolved_adapter.inspect(source_path, selection=resolved_selection)

    features: dict[str, FeatureSpec] = {}
    mappings: dict[str, FeatureMapping] = {}
    if state_sources:
        state_fields = [_uniform_field(inspection, selector) for selector in state_sources]
        resolved_state_dtype = state_dtype or state_fields[0].dtypes[0]
        if state_dtype is None and any(field.dtypes[0] != resolved_state_dtype for field in state_fields):
            raise PlanValidationError(
                "Selected state fields have different dtypes; provide state_dtype explicitly",
                context={"selectors": list(state_sources)},
            )
        state_size = sum(int(np.prod(field.shapes[0])) for field in state_fields)
        features["observation.state"] = FeatureSpec(resolved_state_dtype, (state_size,))
        mappings["observation.state"] = FeatureMapping(
            sources=tuple(state_sources),
            operation="concat",
            cast=(
                resolved_state_dtype
                if any(field.dtypes[0] != resolved_state_dtype for field in state_fields)
                else None
            ),
        )

    for source_selector, target_name in (image_sources or {}).items():
        image_field = _uniform_field(inspection, source_selector)
        if not image_field.image_candidate:
            raise PlanValidationError(
                "The selected image field is not a consistent three-dimensional uint8 observation",
                context={"selector": source_selector},
            )
        features[target_name] = FeatureSpec(
            dtype="video" if use_videos else "image",
            shape=image_field.shapes[0],
        )
        mappings[target_name] = FeatureMapping(sources=(source_selector,))

    action_field = _uniform_field(inspection, action_source)
    resolved_action_dtype = action_dtype or action_field.dtypes[0]
    features["action"] = FeatureSpec(resolved_action_dtype, action_field.shapes[0])
    mappings["action"] = FeatureMapping(
        sources=(action_source,),
        cast=(resolved_action_dtype if action_field.dtypes[0] != resolved_action_dtype else None),
    )

    return ConversionPlan(
        adapter=resolved_adapter.name,
        source=source_path,
        selection=resolved_selection,
        target=TargetConfig(
            repo_id=repo_id,
            root=Path(target_root),
            robot_type=robot_type,
            use_videos=use_videos,
        ),
        fps=fps,
        task=(
            TaskProvider("static", task)
            if task is not None
            else TaskProvider("metadata", task_metadata or "")
        ),
        features=features,
        mappings=mappings,
    )


def convert(
    plan: ConversionPlan | str | Path,
    *,
    registry: AdapterRegistry | None = None,
) -> ConversionResult:
    """Run preflight, stream frames, validate the output, and commit atomically."""

    resolved_plan = load_plan(plan) if isinstance(plan, (str, Path)) else plan
    return convert_dataset(resolved_plan, registry=registry)


def merge(
    sources: Sequence[str | Path],
    *,
    target_root: str | Path,
    repo_id: str,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
) -> MergeResult:
    """Merge existing LeRobot Dataset v3 inputs into one validated dataset."""

    return merge_lerobot_datasets(
        sources,
        target_root=target_root,
        repo_id=repo_id,
        concatenate_videos=concatenate_videos,
        concatenate_data=concatenate_data,
    )


def replay_maniskill(
    source: str | Path,
    *,
    obs_mode: str = "rgb",
    use_env_states: bool = False,
    target_control_mode: str | None = None,
    sim_backend: str | None = None,
    count: int | None = None,
    num_envs: int = 1,
    record_rewards: bool = False,
    reward_mode: str | None = None,
    allow_failure: bool = False,
) -> ManiSkillReplayResult:
    """Materialize a new ManiSkill trajectory pair through the optional simulator runtime."""

    return run_maniskill_replay(
        source,
        options=ManiSkillReplayOptions(
            obs_mode=obs_mode,
            use_env_states=use_env_states,
            target_control_mode=target_control_mode,
            sim_backend=sim_backend,
            count=count,
            num_envs=num_envs,
            record_rewards=record_rewards,
            reward_mode=reward_mode,
            allow_failure=allow_failure,
        ),
    )


def validate(
    target: str | Path,
    *,
    repo_id: str | None = None,
    plan: ConversionPlan | str | Path | None = None,
    registry: AdapterRegistry | None = None,
) -> ValidationReport:
    """Validate a target and optionally compare it with the source expectations in a plan."""

    if plan is None:
        return validate_lerobot_dataset(target, repo_id=repo_id)
    resolved_plan = load_plan(plan) if isinstance(plan, (str, Path)) else plan
    report = preflight(resolved_plan, registry=registry)
    return validate_lerobot_dataset(
        target,
        repo_id=repo_id or resolved_plan.target.repo_id,
        expected_episode_ids=report.inspection.episode_ids,
        expected_episode_lengths=report.episode_lengths,
        expected_features={name: spec.to_lerobot() for name, spec in resolved_plan.features.items()},
        expected_tasks=report.tasks,
    )


def _uniform_field(inspection: DatasetInspection, selector: str) -> FieldInspection:
    """Return a source field with a consistent schema across all selected episodes."""

    field = inspection.field(selector)
    if field is None:
        raise PlanValidationError("Source field does not exist", context={"selector": selector})
    if not field.schema_consistent:
        raise PlanValidationError(
            "Source field schema is inconsistent across selected episodes",
            context={
                "selector": selector,
                "dtypes": field.dtypes,
                "shapes": field.shapes,
                "missing_episodes": field.missing_episodes,
            },
        )
    return field
