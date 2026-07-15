"""Test paired ManiSkill discovery, alignment, lazy reads, and LeRobot conversion."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from leport.api import convert, create_plan, inspect, validate
from leport.cli import main
from leport.conversion.pipeline import preflight
from leport.errors import (
    ConversionError,
    OptionalDependencyError,
    PlanValidationError,
    SourceSchemaError,
)
from leport.sources.maniskill import ManiSkillAdapter
from leport.sources.registry import create_default_registry
from leport.sources.robomimic import RobomimicAdapter
from leport.sources.types import EpisodeSelection


def test_probe_requires_a_structural_hdf5_json_pair(
    maniskill_file: Path,
    malformed_maniskill_pairs: dict[str, Path],
    tmp_path: Path,
) -> None:
    adapter = ManiSkillAdapter()
    matched = adapter.probe(maniskill_file)
    assert matched.confidence == 100
    assert "3 episodes" in matched.reason

    missing_json = adapter.probe(malformed_maniskill_pairs["missing_json"])
    assert missing_json.confidence == 0
    assert "same-basename JSON" in missing_json.reason

    generic = tmp_path / "generic.h5"
    with h5py.File(generic, "w") as h5_file:
        h5_file.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    generic.with_suffix(".json").write_text(json.dumps({"env_info": {}, "episodes": []}), encoding="utf-8")
    generic_result = adapter.probe(generic)
    assert generic_result.confidence == 0
    assert "traj_<episode_id>" in generic_result.reason


def test_catalog_validation_numeric_order_and_selection(
    maniskill_file: Path,
    malformed_maniskill_pairs: dict[str, Path],
) -> None:
    adapter = ManiSkillAdapter()
    inspection = adapter.inspect(maniskill_file)
    assert inspection.episode_ids == ("traj_0", "traj_2", "traj_10")
    assert inspection.episode_lengths == {"traj_0": 3, "traj_2": 4, "traj_10": 2}

    explicit = adapter.inspect(
        maniskill_file,
        selection=EpisodeSelection(episode_ids=("traj_10", "traj_0")),
    )
    assert explicit.episode_ids == ("traj_0", "traj_10")
    with pytest.raises(SourceSchemaError, match="unknown ManiSkill IDs") as unknown_error:
        adapter.inspect(
            maniskill_file,
            selection=EpisodeSelection(episode_ids=("traj_999",)),
        )
    assert unknown_error.value.context == {
        "unknown": ["traj_999"],
        "available": ["traj_0", "traj_2", "traj_10"],
    }
    with pytest.raises(SourceSchemaError, match="filter keys") as filter_error:
        adapter.inspect(maniskill_file, selection=EpisodeSelection(filter_key="success"))
    assert filter_error.value.context == {"filter_key": "success"}

    expected_messages = {
        "duplicate_json_id": "duplicate episode IDs",
        "json_without_hdf5": "catalogs do not match",
        "hdf5_without_json_episode": "catalogs do not match",
        "declared_length": "elapsed_steps",
        "missing_actions": "frame-addressable actions",
    }
    for case, message in expected_messages.items():
        with pytest.raises(SourceSchemaError, match=message):
            adapter.inspect(malformed_maniskill_pairs[case])


def test_malformed_metadata_reports_the_companion_path(
    malformed_maniskill_pairs: dict[str, Path],
) -> None:
    source = malformed_maniskill_pairs["invalid_json"]
    with pytest.raises(SourceSchemaError, match="parse ManiSkill JSON") as parse_error:
        ManiSkillAdapter().inspect(source)
    assert parse_error.value.context["metadata"] == str(source.with_suffix(".json"))

    source.with_suffix(".json").write_text(
        json.dumps({"episodes": []}),
        encoding="utf-8",
    )
    with pytest.raises(SourceSchemaError, match="env_info") as structure_error:
        ManiSkillAdapter().inspect(source)
    assert structure_error.value.context["path"] == "env_info"


def test_inspection_projects_nested_fields_and_preserves_metadata(maniskill_file: Path) -> None:
    inspection = ManiSkillAdapter().inspect(maniskill_file)
    assert inspection.total_frames == 9
    assert inspection.field("actions").shapes == ((7,),)  # type: ignore[union-attr]
    assert inspection.field("terminated").dtypes == ("bool",)  # type: ignore[union-attr]
    assert inspection.field("rewards").episode_lengths == {  # type: ignore[union-attr]
        "traj_0": 3,
        "traj_2": 4,
        "traj_10": 2,
    }

    for selector in (
        "obs/agent/qpos",
        "next_obs/agent/qpos",
        "env_states/actors/cube",
        "next_env_states/actors/cube",
    ):
        field = inspection.field(selector)
        assert field is not None and field.schema_consistent
        assert field.episode_lengths == inspection.episode_lengths

    for selector in (
        "obs/sensor_data/base_camera/rgb",
        "next_obs/sensor_data/base_camera/rgb",
        "obs/sensor_data/wrist_camera/rgb",
    ):
        field = inspection.field(selector)
        assert field is not None and field.image_candidate
        assert field.dtypes == ("uint8",)
        assert field.shapes == ((16, 20, 3),)
    depth = inspection.field("obs/sensor_data/base_camera/depth")
    assert depth is not None and not depth.image_candidate and depth.dtypes == ("uint16",)

    assert inspection.metadata["env_info"]["env_id"] == "PickCube-v1"
    assert inspection.metadata["source_type"] == "motionplanning"
    assert inspection.metadata["source_desc"] == "Deterministic ManiSkill test trajectories"
    assert inspection.metadata["hdf5_filename"] == maniskill_file.name
    assert inspection.metadata["json_filename"] == maniskill_file.with_suffix(".json").name
    assert inspection.metadata["episode_metadata"]["traj_2"]["control_mode"] == ("pd_joint_delta_pos")
    assert not {"fps", "robot_type", "action_meaning", "task"} & set(inspection.metadata)
    json.dumps(inspection.to_dict())


def test_raw_and_dataset_root_observations_are_inspectable(raw_maniskill_file: Path) -> None:
    raw = ManiSkillAdapter().inspect(raw_maniskill_file)
    assert raw.field("obs") is None and raw.field("next_obs") is None
    assert raw.field("env_states/actors/cube") is not None
    assert raw.field("next_env_states/actors/cube") is not None

    with h5py.File(raw_maniskill_file, "r+") as h5_file:
        h5_file["traj_0"].create_dataset(
            "obs",
            data=np.arange(12, dtype=np.float32).reshape(3, 4),
        )
    materialized = ManiSkillAdapter().inspect(raw_maniskill_file)
    assert materialized.field("obs").shapes == ((4,),)  # type: ignore[union-attr]
    assert materialized.field("next_obs").shapes == ((4,),)  # type: ignore[union-attr]


def test_schema_drift_is_reported_across_episodes(
    malformed_maniskill_pairs: dict[str, Path],
) -> None:
    inspection = ManiSkillAdapter().inspect(malformed_maniskill_pairs["schema_drift"])
    current = inspection.field("obs/state")
    following = inspection.field("next_obs/state")
    assert current is not None and following is not None
    assert current.shapes == ((6,), (7,)) and not current.schema_consistent
    assert following.shapes == ((6,), (7,)) and not following.schema_consistent
    assert any("obs/state" in diagnostic for diagnostic in inspection.diagnostics)


@pytest.mark.parametrize(
    ("case", "selector", "expected_length", "actual_length"),
    [
        ("bad_transition_length", "rewards", 2, 3),
        ("bad_observation_length", "obs/state", 3, 2),
        ("bad_environment_state_length", "env_states/state", 3, 2),
    ],
)
def test_invalid_transition_alignment_is_never_adjusted(
    malformed_maniskill_pairs: dict[str, Path],
    case: str,
    selector: str,
    expected_length: int,
    actual_length: int,
) -> None:
    with pytest.raises(SourceSchemaError, match="transition alignment") as caught:
        ManiSkillAdapter().inspect(malformed_maniskill_pairs[case])
    assert caught.value.context == {
        "episode": "traj_0",
        "selector": selector,
        "actions_length": 2,
        "expected_length": expected_length,
        "actual_length": actual_length,
    }


def test_lazy_iteration_preserves_current_next_values_and_metadata(maniskill_file: Path) -> None:
    episodes = ManiSkillAdapter().iter_episodes(
        maniskill_file,
        selection=EpisodeSelection(episode_ids=("traj_10", "traj_0")),
        selectors=(
            "actions",
            "obs/agent/qpos",
            "next_obs/agent/qpos",
            "env_states/actors/cube",
            "next_env_states/actors/cube",
            "obs/sensor_data/base_camera/rgb",
            "next_obs/sensor_data/base_camera/rgb",
        ),
    )
    episode = next(episodes)
    assert episode.episode_id == "traj_0" and episode.length == 3
    assert episode.metadata["episode_id"] == 0
    assert episode.metadata["instruction"] == "pick up the cube"
    assert episode.metadata["hdf5_filename"] == maniskill_file.name
    frame = next(iter(episode.iter_frames()))
    np.testing.assert_array_equal(frame.fields["actions"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(frame.fields["obs/agent/qpos"], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(frame.fields["next_obs/agent/qpos"], np.arange(6, 12, dtype=np.float32))
    np.testing.assert_array_equal(frame.fields["env_states/actors/cube"], np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(
        frame.fields["next_env_states/actors/cube"], np.arange(3, 6, dtype=np.float32)
    )
    current_rgb = frame.fields["obs/sensor_data/base_camera/rgb"]
    next_rgb = frame.fields["next_obs/sensor_data/base_camera/rgb"]
    assert current_rgb.dtype == next_rgb.dtype == np.uint8
    assert current_rgb.shape == next_rgb.shape == (16, 20, 3)
    np.testing.assert_array_equal(current_rgb[0, 0], np.asarray([10, 30, 50], dtype=np.uint8))
    np.testing.assert_array_equal(next_rgb[0, 0], np.asarray([11, 30, 51], dtype=np.uint8))
    episodes.close()


def test_iteration_reads_only_requested_datasets(maniskill_file: Path) -> None:
    read_paths: list[str] = []
    real_getitem = h5py.Dataset.__getitem__

    def tracked_getitem(dataset: h5py.Dataset, key: object) -> object:
        """Record payload reads while delegating unchanged to h5py."""

        read_paths.append(dataset.name)
        return real_getitem(dataset, key)

    with patch.object(h5py.Dataset, "__getitem__", tracked_getitem):
        episodes = ManiSkillAdapter().iter_episodes(
            maniskill_file,
            selection=EpisodeSelection(episode_ids=("traj_0",)),
            selectors=("actions", "next_obs/agent/qpos"),
        )
        frame = next(iter(next(episodes).iter_frames()))
        assert set(frame.fields) == {"actions", "next_obs/agent/qpos"}
        episodes.close()
    assert read_paths == ["/traj_0/actions", "/traj_0/obs/agent/qpos"]


def test_missing_selector_and_iterator_close_are_precise(
    maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ManiSkillAdapter().iter_episodes(maniskill_file, selectors=("obs/missing",))
    with pytest.raises(SourceSchemaError, match="fields are missing") as missing_error:
        next(missing)
    assert missing_error.value.context["episode"] == "traj_0"
    assert missing_error.value.context["missing"] == ["obs/missing"]

    real_file = h5py.File
    opened: list[h5py.File] = []

    def tracked_file(*args: object, **kwargs: object) -> h5py.File:
        """Retain opened handles so the test can assert generator ownership."""

        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", tracked_file)
    episodes = ManiSkillAdapter().iter_episodes(maniskill_file, selectors=("actions",))
    next(episodes)
    assert opened[-1].id.valid
    episodes.close()
    assert not opened[-1].id.valid


def test_registry_api_and_cli_inspection_are_integrated(
    maniskill_file: Path,
    aloha_directory: Path,
    robomimic_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = create_default_registry(discover_plugins=False)
    assert registry.names == ("aloha", "maniskill", "robomimic")
    assert registry.select(maniskill_file).name == "maniskill"
    assert registry.select(aloha_directory).name == "aloha"
    assert registry.select(robomimic_file).name == "robomimic"

    api_result = inspect(maniskill_file)
    assert api_result.adapter == "maniskill"
    json.dumps(api_result.to_dict())
    assert main(["inspect", str(maniskill_file), "--adapter", "maniskill", "--json"]) == 0
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["adapter"] == "maniskill"
    assert cli_result["episode_ids"] == ["traj_0", "traj_2", "traj_10"]


def test_plan_and_preflight_keep_alignment_and_semantics_explicit(
    maniskill_file: Path,
    tmp_path: Path,
) -> None:
    plan = create_plan(
        maniskill_file,
        target_root=tmp_path / "planned-target",
        repo_id="tests/maniskill-plan",
        robot_type="panda-test",
        fps=30,
        task="pick up the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=(
            "obs/agent/qpos",
            "next_obs/agent/qpos",
            "env_states/actors/cube",
            "next_env_states/actors/cube",
        ),
        state_dtype="float32",
        image_sources={
            "obs/sensor_data/base_camera/rgb": "observation.images.base",
        },
        use_videos=False,
        adapter="maniskill",
        selection=EpisodeSelection(episode_ids=("traj_10", "traj_0")),
    )
    assert plan.adapter == "maniskill" and plan.fps == 30
    assert plan.task.value == "pick up the cube"
    assert plan.target.robot_type == "panda-test"
    assert plan.mappings["action"].sources == ("actions",)
    assert plan.mappings["observation.state"].sources == (
        "obs/agent/qpos",
        "next_obs/agent/qpos",
        "env_states/actors/cube",
        "next_env_states/actors/cube",
    )
    assert plan.features["observation.state"].shape == (18,)
    assert plan.features["observation.images.base"].shape == (16, 20, 3)
    report = preflight(plan)
    assert report.episode_lengths == (3, 2)
    assert report.tasks == ("pick up the cube",)

    with pytest.raises(PlanValidationError, match="exactly one of task"):
        create_plan(
            maniskill_file,
            target_root=tmp_path / "missing-task",
            repo_id="tests/maniskill-missing-task",
            fps=30,
            action_source="actions",
            adapter="maniskill",
        )


def test_numeric_conversion_preserves_order_counts_tasks_and_values(
    maniskill_file: Path,
    tmp_path: Path,
) -> None:
    plan = create_plan(
        maniskill_file,
        target_root=tmp_path / "numeric-target",
        repo_id="tests/maniskill-numeric",
        fps=30,
        task="pick up the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/agent/qpos", "next_env_states/actors/cube"),
        state_dtype="float32",
        use_videos=False,
        adapter="maniskill",
    )
    result = convert(plan)
    assert result.validation.total_episodes == 3
    assert result.validation.total_frames == 9
    assert result.validation.episode_lengths == (3, 4, 2)
    assert result.validation.tasks == ("pick up the cube",)
    assert validate(result.target, plan=plan).episode_lengths == (3, 4, 2)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=plan.target.root)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(
        dataset[0]["observation.state"].numpy(),
        np.asarray([0, 1, 2, 3, 4, 5, 3, 4, 5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset[3]["action"].numpy(),
        np.arange(7, dtype=np.float32) + 200,
    )


@pytest.mark.parametrize("use_videos", [False, True])
def test_image_and_video_conversion_preserve_resolution_rgb_order_and_alignment(
    maniskill_file: Path,
    tmp_path: Path,
    use_videos: bool,
) -> None:
    kind = "video" if use_videos else "image"
    plan = create_plan(
        maniskill_file,
        target_root=tmp_path / f"visual-{kind}",
        repo_id=f"tests/maniskill-{kind}",
        fps=30,
        task="pick up the cube",
        action_source="actions",
        action_dtype="float32",
        image_sources={
            "obs/sensor_data/base_camera/rgb": "observation.images.base",
            "obs/sensor_data/wrist_camera/rgb": "observation.images.wrist",
        },
        use_videos=use_videos,
        adapter="maniskill",
        selection=EpisodeSelection(episode_ids=("traj_0",)),
    )
    result = convert(plan)
    assert result.validation.episode_lengths == (3,)
    assert result.validation.features["observation.images.base"]["dtype"] == kind
    assert result.validation.features["observation.images.base"]["shape"] == (16, 20, 3)
    assert set(result.validation.decoded_visual_features) == {
        "observation.images.base",
        "observation.images.wrist",
    }

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=plan.target.root)
    tolerance = 0.05 if use_videos else 1 / 255
    for frame_index, expected in (
        (0, np.asarray([10, 30, 50], dtype=np.float32) / 255),
        (1, np.asarray([11, 30, 51], dtype=np.float32) / 255),
    ):
        image = dataset[frame_index]["observation.images.base"].numpy()
        pixel = image[:, 0, 0] if image.shape[0] == 3 else image[0, 0, :]
        np.testing.assert_allclose(pixel, expected, atol=tolerance)


def test_preflight_and_target_failures_never_commit_partial_output(
    raw_maniskill_file: Path,
    malformed_maniskill_pairs: dict[str, Path],
    maniskill_file: Path,
    tmp_path: Path,
) -> None:
    raw_target = tmp_path / "raw-target"
    with pytest.raises(PlanValidationError, match="does not exist"):
        create_plan(
            raw_maniskill_file,
            target_root=raw_target,
            repo_id="tests/maniskill-raw",
            fps=30,
            task="pick up the cube",
            action_source="actions",
            state_sources=("obs/agent/qpos",),
            adapter="maniskill",
        )
    assert not raw_target.exists()

    with pytest.raises(PlanValidationError, match="inconsistent"):
        create_plan(
            malformed_maniskill_pairs["schema_drift"],
            target_root=tmp_path / "drift-target",
            repo_id="tests/maniskill-drift",
            fps=30,
            task="pick up the cube",
            action_source="actions",
            state_sources=("obs/state",),
            adapter="maniskill",
        )
    with pytest.raises(SourceSchemaError, match="transition alignment"):
        ManiSkillAdapter().inspect(malformed_maniskill_pairs["bad_observation_length"])

    occupied_target = tmp_path / "occupied-target"
    occupied_target.mkdir()
    marker = occupied_target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    plan = create_plan(
        maniskill_file,
        target_root=occupied_target,
        repo_id="tests/maniskill-occupied",
        fps=30,
        task="pick up the cube",
        action_source="actions",
        adapter="maniskill",
    )
    with pytest.raises(ConversionError, match="not empty"):
        convert(plan)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".occupied-target.leport-*"))


def test_maniskill_dependency_isolation_does_not_break_other_adapters(
    maniskill_file: Path,
    robomimic_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = create_default_registry(discover_plugins=False)
    monkeypatch.setitem(sys.modules, "h5py", None)
    assert registry.get("maniskill").name == "maniskill"
    with pytest.raises(OptionalDependencyError, match="uv sync --extra maniskill") as caught:
        registry.get("maniskill").probe(maniskill_file)
    assert caught.value.context == {
        "adapter": "maniskill",
        "extra": "maniskill",
        "dependency": "h5py",
    }
    assert registry.get("aloha").name == "aloha"
    monkeypatch.undo()
    assert RobomimicAdapter().inspect(robomimic_file).episode_ids == (
        "demo_0",
        "demo_2",
        "demo_10",
    )


def test_maniskill_documentation_and_notebook_cover_the_supported_workflow() -> None:
    repository = Path(__file__).parents[1]
    guide = (repository / "docs/maniskill.md").read_text(encoding="utf-8")
    readme = (repository / "README.md").read_text(encoding="utf-8")
    notebook_text = (repository / "notebooks/maniskill.ipynb").read_text(encoding="utf-8")
    for selector in (
        "actions",
        "obs/agent/qpos",
        "next_obs/agent/qpos",
        "env_states/actors/cube",
        "next_env_states/actors/cube",
        "obs/sensor_data/base_camera/rgb",
    ):
        assert selector in guide
    for selector in (
        "actions",
        "env_states/articulations/panda",
        "env_states/actors/cube",
        "obs/sensor_data/base_camera/rgb",
    ):
        assert selector in notebook_text
    for command in ("inspect", "plan", "convert", "validate", "replay-maniskill", "merge"):
        assert f"uv run leport {command}" in guide
        assert f"uv run leport {command}" in notebook_text
    assert "## Equivalent CLI commands" in notebook_text
    assert "T+1" in guide and "i + 1" in guide
    assert "same basename" in guide and "raw" in guide and "replay" in guide
    assert "does not infer" in guide and "filter_key" in guide and "unsupported" in guide
    assert "--extra maniskill" in guide
    assert "--extra maniskill-replay" in guide
    assert "docs/maniskill.md" in readme and "notebooks/maniskill.ipynb" in readme
    assert "uv run" not in readme


def test_maniskill_notebook_uses_downloaded_source_and_compiles_optional_replay() -> None:
    notebook_path = Path(__file__).parents[1] / "notebooks/maniskill.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"{notebook_path.name}:{cell['id']}", "exec")

    assert "data/maniskill/PickCube-v1-teleop/trajectory.h5" in notebook_source
    assert notebook_source.count("RUN_REPLAY = False") == 1
    assert notebook_source.count("if RUN_REPLAY:") == 2
    assert notebook_source.count("RUN_MERGE = False") == 1
    assert notebook_source.count("if RUN_MERGE:") == 1
    assert "replay_maniskill(" in notebook_source
    assert "merge(" in notebook_source
    assert 'EpisodeSelection(episode_ids=("traj_0",))' in notebook_source
    assert 'EpisodeSelection(episode_ids=("traj_1",))' in notebook_source
    assert "RUN_WORKFLOW" not in notebook_source
    assert "notebook-fixture" not in notebook_source
    assert "create_dataset(" not in notebook_source
    assert "import h5py" not in notebook_source
    assert "import numpy" not in notebook_source


def test_every_maniskill_requirement_and_scenario_has_an_automated_test_mapping() -> None:
    probe = "test_probe_requires_a_structural_hdf5_json_pair"
    catalog = "test_catalog_validation_numeric_order_and_selection"
    metadata_errors = "test_malformed_metadata_reports_the_companion_path"
    inspection = "test_inspection_projects_nested_fields_and_preserves_metadata"
    raw = "test_raw_and_dataset_root_observations_are_inspectable"
    drift = "test_schema_drift_is_reported_across_episodes"
    alignment = "test_invalid_transition_alignment_is_never_adjusted"
    iteration = "test_lazy_iteration_preserves_current_next_values_and_metadata"
    selected_reads = "test_iteration_reads_only_requested_datasets"
    lifecycle = "test_missing_selector_and_iterator_close_are_precise"
    integration = "test_registry_api_and_cli_inspection_are_integrated"
    planning = "test_plan_and_preflight_keep_alignment_and_semantics_explicit"
    numeric = "test_numeric_conversion_preserves_order_counts_tasks_and_values"
    visual = "test_image_and_video_conversion_preserve_resolution_rgb_order_and_alignment"
    failures = "test_preflight_and_target_failures_never_commit_partial_output"
    dependencies = "test_maniskill_dependency_isolation_does_not_break_other_adapters"
    documentation = "test_maniskill_documentation_and_notebook_cover_the_supported_workflow"
    notebook = "test_maniskill_notebook_uses_downloaded_source_and_compiles_optional_replay"
    requirement_tests = {
        "Recognize paired ManiSkill trajectory sources": probe,
        "Validate and order the episode catalog": catalog,
        "Select episodes deterministically": catalog,
        "Inspect nested ManiSkill fields": inspection,
        "Expose explicit transition alignment": alignment,
        "Support raw and replayed trajectory variants": raw,
        "Preserve values through lazy reads": iteration,
        "Preserve metadata without inferring semantics": inspection,
        "Integrate with the existing conversion pipeline": integration,
        "Isolate ManiSkill dependencies": dependencies,
        "Document the ManiSkill workflow in English": documentation,
    }
    scenario_tests = {
        "Standard trajectory pair is recognized": probe,
        "Companion JSON is missing": probe,
        "Metadata JSON is malformed": metadata_errors,
        "Generic HDF5 is rejected": probe,
        "File order differs from numeric order": catalog,
        "JSON episode has no HDF5 group": catalog,
        "HDF5 trajectory has no JSON episode": catalog,
        "Declared length conflicts with actions": catalog,
        "Explicit subset is selected": catalog,
        "Explicit ID is unknown": catalog,
        "Filter key is requested": catalog,
        "Transition fields are inspected": inspection,
        "Nested observations are inspected": inspection,
        "RGB observation is an image candidate": inspection,
        "Field coverage differs by episode": drift,
        "Current observation is read": iteration,
        "Next observation is read": iteration,
        "Current and next environment states are read": iteration,
        "Observation does not have T plus one values": alignment,
        "Transition field does not have T values": alignment,
        "Raw trajectory has no observations": raw,
        "Plan requests an unavailable observation": failures,
        "Replayed trajectory contains state and RGB observations": inspection,
        "Small episode subset is converted": selected_reads,
        "One camera is mapped": selected_reads,
        "Current and next pixels are preserved": iteration,
        "Iterator is closed early": lifecycle,
        "Dataset metadata is inspected": inspection,
        "Episode metadata is yielded": iteration,
        "Environment ID is available but no task text is supplied": planning,
        "Automatic adapter selection is unambiguous": integration,
        "Numeric conversion succeeds": numeric,
        "RGB conversion succeeds": visual,
        "Preflight fails": failures,
        "ManiSkill extra is installed": integration,
        "HDF5 dependency is absent": dependencies,
        "Another adapter is used": dependencies,
        "User starts with a raw official demonstration": documentation,
        "User maps transition data": documentation,
        "Notebook default workflow is inspected": notebook,
        "Notebook replay workflow is selected": notebook,
    }
    repository = Path(__file__).parents[1]
    candidates = (
        repository / "openspec/specs/maniskill-source-adapter/spec.md",
        repository / "openspec/changes/add-maniskill-hdf5-adapter/specs/maniskill-source-adapter/spec.md",
    )
    specification_path = next(path for path in candidates if path.is_file())
    specification = specification_path.read_text(encoding="utf-8")
    assert set(requirement_tests) == set(re.findall(r"^### Requirement: (.+)$", specification, re.MULTILINE))
    assert set(scenario_tests) == set(re.findall(r"^#### Scenario: (.+)$", specification, re.MULTILINE))
    for test_name in (*requirement_tests.values(), *scenario_tests.values()):
        assert callable(globals().get(test_name)), test_name
