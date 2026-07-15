"""Test ConversionPlan validation, mechanical mapping, and full-episode preflight."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from leport.conversion.mapping import map_frame
from leport.conversion.pipeline import preflight
from leport.conversion.plan import (
    ConversionPlan,
    FeatureMapping,
    FeatureSpec,
    TargetConfig,
    TaskProvider,
    load_plan,
    plan_from_dict,
    save_plan,
)
from leport.errors import ConversionError, PlanValidationError
from leport.sources.types import EpisodeSelection, SourceEpisode, SourceFrame


def test_yaml_round_trip_is_stable(numeric_plan: ConversionPlan, tmp_path: Path) -> None:
    path = save_plan(numeric_plan, tmp_path / "plan.yaml")
    loaded = load_plan(path)
    assert loaded == numeric_plan
    assert loaded.to_dict() == numeric_plan.to_dict()


def test_yaml_rejects_unknown_and_wrong_typed_fields(numeric_plan: ConversionPlan) -> None:
    raw = numeric_plan.to_dict()
    raw["unknown"] = True
    with pytest.raises(PlanValidationError, match="fields do not match"):
        plan_from_dict(raw)

    raw = numeric_plan.to_dict()
    raw["fps"] = "20"
    with pytest.raises(PlanValidationError, match="fps must be an integer"):
        plan_from_dict(raw)

    # Strict YAML typing rejects a scalar episode list and numeric selector instead of hiding
    # configuration mistakes until source access.
    raw = numeric_plan.to_dict()
    raw["selection"]["episode_ids"] = "demo_0"
    with pytest.raises(PlanValidationError, match="episode_ids must be a list of strings"):
        plan_from_dict(raw)

    raw = numeric_plan.to_dict()
    raw["mappings"]["action"]["sources"] = [123]
    with pytest.raises(PlanValidationError, match=r"Every .* value must be a string"):
        plan_from_dict(raw)


def test_plan_requires_action_task_and_positive_fps(numeric_plan: ConversionPlan) -> None:
    with pytest.raises(PlanValidationError, match="action"):
        ConversionPlan(
            adapter=numeric_plan.adapter,
            source=numeric_plan.source,
            selection=numeric_plan.selection,
            target=numeric_plan.target,
            fps=20,
            task=numeric_plan.task,
            features={"observation.state": FeatureSpec("float32", (5,))},
            mappings={
                "observation.state": FeatureMapping(
                    ("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
                    "concat",
                    "float32",
                )
            },
        )
    with pytest.raises(PlanValidationError, match="fps"):
        ConversionPlan(
            adapter=numeric_plan.adapter,
            source=numeric_plan.source,
            selection=numeric_plan.selection,
            target=numeric_plan.target,
            fps=0,
            task=numeric_plan.task,
            features=numeric_plan.features,
            mappings=numeric_plan.mappings,
        )


def test_concat_and_explicit_cast_preserve_order(numeric_plan: ConversionPlan) -> None:
    episode = SourceEpisode("demo", 1, ())
    frame = SourceFrame(
        0,
        {
            "obs/robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "obs/robot0_gripper_qpos": np.asarray([4.0, 5.0], dtype=np.float64),
            "actions": np.arange(7, dtype=np.float64),
        },
    )
    mapped = map_frame(numeric_plan, episode, frame)
    np.testing.assert_array_equal(
        mapped["observation.state"],
        np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
    )
    assert mapped["observation.state"].dtype == np.float32
    assert mapped["action"].dtype == np.float32
    assert mapped["task"] == "lift the cube"
    assert "timestamp" not in mapped and "frame_index" not in mapped


def test_image_channel_first_and_last_are_compatible(tmp_path: Path) -> None:
    plan = ConversionPlan(
        adapter="test",
        source=tmp_path / "source",
        selection=EpisodeSelection(),
        target=TargetConfig("tests/image", tmp_path / "target"),
        fps=10,
        task=TaskProvider("static", "task"),
        features={
            "observation.images.cam": FeatureSpec("video", (16, 16, 3)),
            "action": FeatureSpec("float32", (1,)),
        },
        mappings={
            "observation.images.cam": FeatureMapping(("camera",)),
            "action": FeatureMapping(("actions",)),
        },
    )
    episode = SourceEpisode("episode", 1, ())
    for image in (
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((3, 16, 16), dtype=np.uint8),
    ):
        mapped = map_frame(
            plan,
            episode,
            SourceFrame(0, {"camera": image, "actions": np.zeros(1, dtype=np.float32)}),
        )
        assert mapped["observation.images.cam"].shape == image.shape


def test_metadata_task_provider_and_error_context(numeric_plan: ConversionPlan) -> None:
    metadata_plan = ConversionPlan(
        adapter=numeric_plan.adapter,
        source=numeric_plan.source,
        selection=numeric_plan.selection,
        target=numeric_plan.target,
        fps=numeric_plan.fps,
        task=TaskProvider("metadata", "instruction"),
        features=numeric_plan.features,
        mappings=numeric_plan.mappings,
    )
    frame = SourceFrame(
        0,
        {
            "obs/robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "obs/robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
            "actions": np.zeros(7, dtype=np.float64),
        },
    )
    episode = SourceEpisode("demo", 1, (), {"instruction": "pick object"})
    assert map_frame(metadata_plan, episode, frame)["task"] == "pick object"

    with pytest.raises(ConversionError) as error:
        map_frame(metadata_plan, SourceEpisode("missing", 1, ()), frame)
    assert error.value.context["episode"] == "missing"
    assert error.value.context["selector"] == "instruction"


def test_preflight_checks_all_episode_schemas(
    numeric_plan: ConversionPlan,
    robomimic_file: Path,
) -> None:
    with h5py.File(robomimic_file, "r+") as h5_file:
        del h5_file["data/demo_10/obs/robot0_gripper_qpos"]
    with pytest.raises(PlanValidationError, match="inconsistent episode schema") as error:
        preflight(numeric_plan)
    assert error.value.context["missing_episodes"] == ("demo_10",)


def test_mapping_error_contains_episode_frame_selector_and_target(numeric_plan: ConversionPlan) -> None:
    episode = SourceEpisode("demo_7", 1, ())
    frame = SourceFrame(
        12,
        {
            "obs/robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "obs/robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
            "actions": np.zeros(6, dtype=np.float64),
        },
    )
    with pytest.raises(ConversionError) as error:
        map_frame(numeric_plan, episode, frame)
    assert error.value.context["episode"] == "demo_7"
    assert error.value.context["frame"] == 12
    assert error.value.context["selector"] == ("actions",)
    assert error.value.context["target"] == "action"


def test_dtype_mismatch_reports_expected_and_actual_context(tmp_path: Path) -> None:
    plan = ConversionPlan(
        adapter="memory",
        source=tmp_path / "source",
        selection=EpisodeSelection(),
        target=TargetConfig("tests/dtype-context", tmp_path / "target", use_videos=False),
        fps=10,
        task=TaskProvider("static", "move"),
        features={"action": FeatureSpec("float32", (2,))},
        mappings={"action": FeatureMapping(("raw-command",))},
    )
    episode = SourceEpisode("segment-4", 1, ())
    frame = SourceFrame(9, {"raw-command": np.zeros(2, dtype=np.float64)})
    with pytest.raises(ConversionError) as error:
        map_frame(plan, episode, frame)
    assert error.value.context == {
        "adapter": "memory",
        "episode": "segment-4",
        "frame": 9,
        "selector": ("raw-command",),
        "target": "action",
        "expected_dtype": "float32",
        "actual_dtype": "float64",
        "expected_shape": (2,),
        "actual_shape": (2,),
    }
