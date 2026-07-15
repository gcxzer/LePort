"""Test official LIBERO discovery, metadata, lazy reads, and LeRobot conversion."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from leport.api import convert, create_plan, inspect, validate
from leport.cli import main
from leport.conversion.pipeline import preflight
from leport.errors import OptionalDependencyError, PlanValidationError, SourceSchemaError
from leport.sources import LiberoAdapter
from leport.sources.registry import create_default_registry
from leport.sources.robomimic import RobomimicAdapter
from leport.sources.types import EpisodeSelection


def test_probe_requires_complete_libero_identity_and_direct_candidates(
    libero_file: Path,
    libero_directory: Path,
    malformed_libero_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    adapter = LiberoAdapter()
    assert adapter.probe(libero_file).confidence == 100
    assert adapter.probe(libero_directory).confidence == 100
    assert adapter.probe(malformed_libero_files["missing_metadata"]).confidence == 0
    assert adapter.probe(malformed_libero_files["missing_bddl"]).confidence == 0

    empty = tmp_path / "empty-suite"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a task\n", encoding="utf-8")
    nested = empty / "nested"
    nested.mkdir()
    (nested / "hidden_demo.hdf5").write_bytes(b"not hdf5")
    result = adapter.probe(empty)
    assert result.confidence == 0
    assert "no direct" in result.reason


def test_catalog_uses_qualified_canonical_order_and_validates_conflicts(
    libero_directory: Path,
    malformed_libero_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    inspection = LiberoAdapter().inspect(libero_directory)
    assert inspection.episode_ids == (
        "alpha_task/demo_0",
        "alpha_task/demo_2",
        "alpha_task/demo_10",
        "beta_task/demo_0",
    )
    assert inspection.episode_lengths == {
        "alpha_task/demo_0": 3,
        "alpha_task/demo_2": 2,
        "alpha_task/demo_10": 2,
        "beta_task/demo_0": 4,
    }
    expected_messages = {
        "noncanonical_demo": "canonical",
        "num_demos_mismatch": "num_demos",
        "num_samples_mismatch": "num_samples",
        "missing_actions": "frame-addressable actions",
    }
    for case, message in expected_messages.items():
        with pytest.raises(SourceSchemaError, match=message):
            LiberoAdapter().inspect(malformed_libero_files[case])

    mixed = tmp_path / "mixed-suite"
    mixed.mkdir()
    valid = mixed / "a_valid_demo.hdf5"
    invalid = mixed / "z_invalid_demo.hdf5"
    valid.write_bytes((libero_directory / "alpha_task_demo.hdf5").read_bytes())
    invalid.write_bytes(malformed_libero_files["invalid_problem_json"].read_bytes())
    with pytest.raises(SourceSchemaError, match="problem_info") as caught:
        LiberoAdapter().inspect(mixed)
    assert caught.value.context["source"] == str(invalid)


def test_selection_is_canonical_precise_and_skips_unselected_task_catalogs(
    libero_directory: Path,
) -> None:
    adapter = LiberoAdapter()
    selected = EpisodeSelection(episode_ids=("beta_task/demo_0", "alpha_task/demo_10", "alpha_task/demo_0"))
    inspection = adapter.inspect(libero_directory, selection=selected)
    assert inspection.episode_ids == (
        "alpha_task/demo_0",
        "alpha_task/demo_10",
        "beta_task/demo_0",
    )
    with pytest.raises(SourceSchemaError, match="unknown demonstrations") as caught:
        adapter.inspect(
            libero_directory,
            selection=EpisodeSelection(episode_ids=("alpha_task/demo_999",)),
        )
    assert caught.value.context["unknown"] == ["alpha_task/demo_999"]
    with pytest.raises(SourceSchemaError, match="qualified episode IDs") as filter_error:
        adapter.inspect(libero_directory, selection=EpisodeSelection(filter_key="train"))
    assert filter_error.value.context == {"filter_key": "train"}

    # Corrupting beta proves that selecting alpha does not open or validate an unselected task file.
    with h5py.File(libero_directory / "beta_task_demo.hdf5", "r+") as h5_file:
        h5_file["data"].attrs["problem_info"] = "{invalid"
    alpha_only = adapter.inspect(
        libero_directory,
        selection=EpisodeSelection(episode_ids=("alpha_task/demo_2",)),
    )
    assert alpha_only.episode_ids == ("alpha_task/demo_2",)


def test_inspection_reports_fields_images_metadata_and_cross_task_drift(
    libero_directory: Path,
) -> None:
    inspection = LiberoAdapter().inspect(libero_directory)
    for selector in (
        "actions",
        "states",
        "robot_states",
        "rewards",
        "dones",
        "obs/ee_states",
        "obs/gripper_states",
        "obs/joint_states",
    ):
        assert inspection.field(selector) is not None
    for selector in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
        field = inspection.field(selector)
        assert field is not None and field.image_candidate
        assert field.dtypes == ("uint8",) and field.shapes == ((16, 16, 3),)

    states = inspection.field("states")
    assert states is not None and states.shapes == ((10,), (12,))
    assert not states.schema_consistent
    alpha_metadata = inspection.metadata["tasks"]["alpha_task"]
    assert alpha_metadata["instruction"] == "place the red bowl on the plate"
    assert alpha_metadata["env_args"]["env_kwargs"]["control_freq"] == 20
    assert alpha_metadata["macros_image_convention"] == "opengl"
    assert "fps" not in alpha_metadata and "robot_type" not in alpha_metadata
    json.dumps(inspection.to_dict())

    with h5py.File(libero_directory / "beta_task_demo.hdf5", "r+") as h5_file:
        del h5_file["data/demo_0/obs/joint_states"]
    missing = LiberoAdapter().inspect(libero_directory).field("obs/joint_states")
    assert missing is not None
    assert missing.missing_episodes == ("beta_task/demo_0",)
    assert not missing.schema_consistent


def test_lazy_iteration_preserves_values_pixels_timing_and_episode_metadata(
    libero_directory: Path,
) -> None:
    episodes = LiberoAdapter().iter_episodes(
        libero_directory,
        selection=EpisodeSelection(episode_ids=("beta_task/demo_0", "alpha_task/demo_0")),
        selectors=("actions", "obs/agentview_rgb", "obs/ee_states"),
    )
    alpha = next(episodes)
    assert alpha.episode_id == "alpha_task/demo_0" and alpha.length == 3
    assert alpha.metadata["instruction"] == "place the red bowl on the plate"
    assert alpha.metadata["task_name"] == "alpha_task"
    assert alpha.metadata["source_filename"] == "alpha_task_demo.hdf5"
    frames = list(alpha.iter_frames())
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert set(frames[0].fields) == {"actions", "obs/agentview_rgb", "obs/ee_states"}
    np.testing.assert_array_equal(frames[0].fields["actions"], np.arange(7, dtype=np.float64))
    expected_pixel = np.asarray([0, 0, 40], dtype=np.uint8)
    np.testing.assert_array_equal(frames[0].fields["obs/agentview_rgb"][0, 0], expected_pixel)
    np.testing.assert_array_equal(
        frames[0].fields["obs/agentview_rgb"][15, 7],
        np.asarray([15, 7, 40], dtype=np.uint8),
    )
    beta = next(episodes)
    assert beta.episode_id == "beta_task/demo_0" and beta.length == 4
    episodes.close()


def test_alignment_empty_episode_missing_selector_and_iterator_close_are_strict(
    libero_file: Path,
    malformed_libero_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SourceSchemaError, match="does not alter alignment") as mismatch:
        next(
            LiberoAdapter().iter_episodes(
                malformed_libero_files["field_length_mismatch"],
                selectors=("obs/ee_states",),
            )
        )
    assert mismatch.value.context == {
        "episode": "field_length_mismatch/demo_0",
        "selector": "obs/ee_states",
        "actions_length": 2,
        "field_length": 1,
    }
    with pytest.raises(SourceSchemaError, match="missing") as missing:
        next(LiberoAdapter().iter_episodes(libero_file, selectors=("obs/not_present",)))
    assert missing.value.context["selector"] == "obs/not_present"

    original_file = h5py.File
    handles: list[h5py.File] = []

    def tracking_file(*args: object, **kwargs: object) -> h5py.File:
        """Record real handles while preserving normal HDF5 behavior for lifecycle assertions."""

        handle = original_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", tracking_file)
    iterator = LiberoAdapter().iter_episodes(libero_file, selectors=("actions",))
    episode = next(iterator)
    next(iter(episode.iter_frames()))
    iterator.close()
    assert handles and all(not bool(handle.id.valid) for handle in handles)

    empty_target = tmp_path / "empty-target"
    with h5py.File(libero_file, "r+") as h5_file:
        demo = h5_file["data/demo_0"]
        del demo["actions"]
        demo.create_dataset("actions", data=np.empty((0, 7), dtype=np.float64))
        demo.attrs["num_samples"] = 0
        h5_file["data"].attrs["total"] = 4
    empty_plan = create_plan(
        libero_file,
        target_root=empty_target,
        repo_id="tests/libero-empty",
        fps=20,
        task_metadata="instruction",
        action_source="actions",
        adapter="libero",
        selection=EpisodeSelection(episode_ids=("alpha_task/demo_0",)),
        use_videos=False,
    )
    with pytest.raises(PlanValidationError, match="Empty episodes"):
        preflight(empty_plan)
    assert not empty_target.exists()


def test_registry_api_cli_and_robomimic_priority_are_integrated(
    libero_file: Path,
    robomimic_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = create_default_registry(discover_plugins=False)
    assert registry.names == ("aloha", "libero", "maniskill", "robomimic", "umi")
    assert registry.select(libero_file).name == "libero"
    assert registry.select(libero_file, name="robomimic").name == "robomimic"
    assert LiberoAdapter().probe(libero_file).confidence == 100
    assert RobomimicAdapter().probe(libero_file).confidence == 80
    assert RobomimicAdapter().probe(robomimic_file).confidence == 100
    generic = tmp_path / "generic.hdf5"
    with h5py.File(generic, "w") as h5_file:
        h5_file.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    assert RobomimicAdapter().probe(generic).confidence == 0
    assert inspect(libero_file).adapter == "libero"
    assert main(["inspect", str(libero_file), "--adapter", "libero", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episode_ids"] == [
        "alpha_task/demo_0",
        "alpha_task/demo_2",
        "alpha_task/demo_10",
    ]


def test_plan_requires_explicit_fps_and_preserves_mapping_order_and_semantics(
    libero_directory: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = create_plan(
        libero_directory,
        target_root=tmp_path / "planned",
        repo_id="tests/libero-plan",
        fps=20,
        task_metadata="instruction",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/ee_states", "obs/gripper_states"),
        state_dtype="float32",
        use_videos=False,
        adapter="libero",
    )
    assert plan.fps == 20 and plan.task.kind == "metadata" and plan.task.value == "instruction"
    assert plan.mappings["observation.state"].sources == (
        "obs/ee_states",
        "obs/gripper_states",
    )
    assert plan.mappings["action"].cast == "float32"
    assert plan.features["observation.state"].shape == (8,)
    assert plan.target.robot_type is None
    report = preflight(plan)
    assert report.episode_lengths == (3, 2, 2, 4)
    assert report.tasks == ("close the upper drawer", "place the red bowl on the plate")

    cli_plan_path = tmp_path / "cli-plan.yaml"
    assert (
        main(
            [
                "plan",
                "--source",
                str(libero_directory),
                "--output",
                str(cli_plan_path),
                "--adapter",
                "libero",
                "--target",
                str(tmp_path / "cli-target"),
                "--repo-id",
                "tests/libero-cli-plan",
                "--fps",
                "20",
                "--task-metadata",
                "instruction",
                "--action",
                "actions",
                "--action-dtype",
                "float32",
                "--state",
                "obs/ee_states",
                "--state",
                "obs/gripper_states",
                "--state-dtype",
                "float32",
                "--no-videos",
                "--json",
            ]
        )
        == 0
    )
    cli_plan = json.loads(capsys.readouterr().out)
    assert cli_plan["fps"] == 20
    assert cli_plan["task"] == {"kind": "metadata", "value": "instruction"}
    assert cli_plan_path.is_file()


def test_multitask_numeric_conversion_preserves_order_tasks_and_values(
    libero_directory: Path,
    tmp_path: Path,
) -> None:
    plan = create_plan(
        libero_directory,
        target_root=tmp_path / "numeric-target",
        repo_id="tests/libero-numeric",
        fps=20,
        task_metadata="instruction",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/ee_states", "obs/gripper_states"),
        state_dtype="float32",
        use_videos=False,
        adapter="libero",
    )
    result = convert(plan)
    assert result.validation.episode_lengths == (3, 2, 2, 4)
    assert result.validation.total_frames == 11
    assert result.validation.tasks == (
        "close the upper drawer",
        "place the red bowl on the plate",
    )
    assert validate(result.target, plan=plan).episode_lengths == (3, 2, 2, 4)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=plan.target.root)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(
        dataset[0]["observation.state"].numpy(),
        np.asarray([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(dataset[7]["action"].numpy(), np.arange(7, dtype=np.float32) + 50)


@pytest.mark.parametrize("use_videos", [False, True])
def test_two_camera_image_and_video_conversion_preserve_dimensions_and_pixels(
    libero_file: Path,
    tmp_path: Path,
    use_videos: bool,
) -> None:
    target_kind = "video" if use_videos else "image"
    plan = create_plan(
        libero_file,
        target_root=tmp_path / f"visual-{target_kind}",
        repo_id=f"tests/libero-{target_kind}",
        fps=20,
        task_metadata="instruction",
        action_source="actions",
        action_dtype="float32",
        image_sources={
            "obs/agentview_rgb": "observation.images.workspace",
            "obs/eye_in_hand_rgb": "observation.images.wrist",
        },
        use_videos=use_videos,
        adapter="libero",
        selection=EpisodeSelection(episode_ids=("alpha_task/demo_0",)),
    )
    result = convert(plan)
    assert result.validation.episode_lengths == (3,)
    assert result.validation.features["observation.images.workspace"]["dtype"] == target_kind
    assert result.validation.features["observation.images.workspace"]["shape"] == (16, 16, 3)
    assert set(result.validation.decoded_visual_features) == {
        "observation.images.workspace",
        "observation.images.wrist",
    }

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=plan.target.root)
    image = dataset[0]["observation.images.workspace"].numpy()
    pixel = image[:, 0, 0] if image.shape[0] == 3 else image[0, 0, :]
    tolerance = 0.08 if use_videos else 1 / 255
    np.testing.assert_allclose(pixel, np.asarray([0, 0, 40]) / 255, atol=tolerance)


def test_preflight_rejects_drift_or_bad_lengths_without_committing(
    libero_directory: Path,
    malformed_libero_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    drift_target = tmp_path / "drift-target"
    with pytest.raises(PlanValidationError, match="inconsistent"):
        create_plan(
            libero_directory,
            target_root=drift_target,
            repo_id="tests/libero-drift",
            fps=20,
            task_metadata="instruction",
            action_source="actions",
            state_sources=("states",),
            adapter="libero",
        )
    assert not drift_target.exists()

    length_target = tmp_path / "length-target"
    plan = create_plan(
        malformed_libero_files["field_length_mismatch"],
        target_root=length_target,
        repo_id="tests/libero-length",
        fps=20,
        task_metadata="instruction",
        action_source="actions",
        state_sources=("obs/ee_states",),
        adapter="libero",
    )
    with pytest.raises(PlanValidationError, match="different length"):
        preflight(plan)
    assert not length_target.exists()


def test_libero_dependency_isolation_keeps_core_and_other_adapters_available(
    libero_file: Path,
    robomimic_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = create_default_registry(discover_plugins=False)
    monkeypatch.setitem(sys.modules, "h5py", None)
    assert registry.get("libero").name == "libero"
    with pytest.raises(OptionalDependencyError, match="uv sync --extra libero") as caught:
        registry.get("libero").probe(libero_file)
    assert caught.value.context == {
        "adapter": "libero",
        "extra": "libero",
        "dependency": "h5py",
    }
    assert registry.get("aloha").name == "aloha"
    monkeypatch.undo()
    assert RobomimicAdapter().inspect(robomimic_file).episode_ids == (
        "demo_0",
        "demo_2",
        "demo_10",
    )


def test_libero_documentation_covers_raw_workflow_without_readme_usage() -> None:
    repository = Path(__file__).parents[1]
    guide = (repository / "docs/libero.md").read_text(encoding="utf-8")
    readme = (repository / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "*_demo.hdf5",
        "language_instruction",
        "qualified episode IDs",
        "task-invariant",
        "explicit FPS",
        "20 Hz",
        "10 Hz",
        "rotat",
        "raw",
        "atomic",
        "merge",
        "uv sync --extra libero",
    ):
        assert phrase in guide
    for selector in (
        "actions",
        "states",
        "obs/ee_states",
        "obs/gripper_states",
        "obs/agentview_rgb",
        "obs/eye_in_hand_rgb",
    ):
        assert selector in guide
    for command in ("inspect", "plan", "convert", "validate", "merge"):
        assert f"uv run leport {command}" in guide
    assert "LIBERO | HDF5 demonstrations with task metadata | ✅ Supported" in readme
    assert "docs/libero.md" in readme and "notebooks/libero.ipynb" in readme
    assert "uv run" not in readme


def test_libero_notebook_uses_real_source_and_equivalent_cli_commands() -> None:
    notebook_path = Path(__file__).parents[1] / "notebooks/libero.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert isinstance(cell["execution_count"], int)
        assert all(output["output_type"] != "error" for output in cell["outputs"])
        source = "".join(cell["source"])
        compile(source, f"{notebook_path.name}:{cell['id']}", "exec")
        comment_lines = [line for line in source.splitlines() if line.lstrip().startswith("#")]
        assert all(len(line) <= 100 for line in comment_lines)

    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    real_relative_path = "data/libero/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5"
    assert real_relative_path in notebook_source
    assert notebook_source.count("RUN_MERGE = False") == 1
    assert notebook_source.count("if RUN_MERGE:") == 1
    assert 'EpisodeSelection(episode_ids=(f"{task_name}/demo_0",))' in notebook_source
    assert 'EpisodeSelection(episode_ids=(f"{task_name}/demo_1",))' in notebook_source
    assert 'task_metadata="instruction"' in notebook_source
    assert "merge(" in notebook_source
    assert "## Equivalent CLI commands" in notebook_source
    for command in ("inspect", "plan", "convert", "validate", "merge"):
        assert f"uv run leport {command}" in notebook_source
    for selector in (
        "actions",
        "obs/ee_states",
        "obs/gripper_states",
        "obs/agentview_rgb",
        "obs/eye_in_hand_rgb",
    ):
        assert selector in notebook_source
    assert "RUN_WORKFLOW" not in notebook_source
    assert "create_dataset(" not in notebook_source
    assert "import h5py" not in notebook_source
    assert "notebook-fixture" not in notebook_source


def test_every_libero_requirement_and_scenario_has_an_automated_test_mapping() -> None:
    probe = "test_probe_requires_complete_libero_identity_and_direct_candidates"
    catalog = "test_catalog_uses_qualified_canonical_order_and_validates_conflicts"
    selection = "test_selection_is_canonical_precise_and_skips_unselected_task_catalogs"
    inspection = "test_inspection_reports_fields_images_metadata_and_cross_task_drift"
    iteration = "test_lazy_iteration_preserves_values_pixels_timing_and_episode_metadata"
    lifecycle = "test_alignment_empty_episode_missing_selector_and_iterator_close_are_strict"
    integration = "test_registry_api_cli_and_robomimic_priority_are_integrated"
    planning = "test_plan_requires_explicit_fps_and_preserves_mapping_order_and_semantics"
    numeric = "test_multitask_numeric_conversion_preserves_order_tasks_and_values"
    visual = "test_two_camera_image_and_video_conversion_preserve_dimensions_and_pixels"
    failures = "test_preflight_rejects_drift_or_bad_lengths_without_committing"
    dependencies = "test_libero_dependency_isolation_keeps_core_and_other_adapters_available"
    documentation = "test_libero_documentation_covers_raw_workflow_without_readme_usage"
    notebook = "test_libero_notebook_uses_real_source_and_equivalent_cli_commands"
    requirement_tests = {
        "Recognize official LIBERO HDF5 sources": probe,
        "Validate and order the task and episode catalog": catalog,
        "Select LIBERO episodes deterministically": selection,
        "Inspect LIBERO fields across selected tasks": inspection,
        "Enforce action-aligned LIBERO fields": lifecycle,
        "Preserve LIBERO task and source metadata": inspection,
        "Preserve raw values through lazy reads": iteration,
        "Integrate LIBERO with the existing conversion pipeline": integration,
        "Isolate LIBERO reader dependencies": dependencies,
        "Document the LIBERO raw-data workflow in English": documentation,
    }
    scenario_tests = {
        "Single task file is recognized": probe,
        "Suite directory is recognized": probe,
        "Unrelated entries are ignored": probe,
        "LIBERO identity metadata is missing": probe,
        "Directory has no task candidates": probe,
        "Task and demo lexical order differs from canonical order": catalog,
        "Episode IDs are globally unique": catalog,
        "Declared demo count conflicts with groups": catalog,
        "Declared episode length conflicts with actions": catalog,
        "Later task file is malformed": catalog,
        "Explicit cross-task subset is selected": selection,
        "Explicit ID is unknown": selection,
        "Filter key is requested": selection,
        "Standard action and state fields are inspected": inspection,
        "Workspace and wrist cameras are inspected": inspection,
        "Task-dependent field shape differs": inspection,
        "Field is absent from one task": inspection,
        "Selected fields align with actions": iteration,
        "Observation length differs": lifecycle,
        "Empty episode is selected": lifecycle,
        "Episode task is resolved from source metadata": planning,
        "File name and language differ": inspection,
        "Source reports a control frequency": inspection,
        "Image convention is present": inspection,
        "One camera is mapped": iteration,
        "Raw pixels are returned": iteration,
        "Temporal sampling is preserved": iteration,
        "Iteration advances between task files": iteration,
        "Iterator is closed early": lifecycle,
        "Automatic selection prefers LIBERO": integration,
        "Multi-task numeric conversion succeeds": numeric,
        "Two-camera conversion succeeds": visual,
        "Preflight rejects inconsistent selected state": failures,
        "LIBERO extra is installed": integration,
        "HDF5 dependency is absent": dependencies,
        "Unrelated adapter is used": dependencies,
        "Reader chooses a state mapping": documentation,
        "Reader compares hosted LIBERO data": documentation,
        "Notebook source is inspected": notebook,
        "Equivalent CLI section is inspected": notebook,
        "README is reviewed": documentation,
    }
    robomimic_requirement_tests = {"Recognize standard robomimic HDF5 structure": integration}
    robomimic_scenario_tests = {
        "Standard structure is recognized": integration,
        "Generic HDF5 file is rejected": integration,
        "Specialized LIBERO source defers during automatic probing": integration,
        "Explicit generic selection remains available": integration,
    }
    repository = Path(__file__).parents[1]
    archived_specs = (
        repository / "openspec/changes/archive/2026-07-15-add-libero-source-adapter/specs"
    )
    libero_specification = (archived_specs / "libero-source-adapter/spec.md").read_text(
        encoding="utf-8"
    )
    robomimic_specification = (archived_specs / "robomimic-source-adapter/spec.md").read_text(
        encoding="utf-8"
    )
    assert set(requirement_tests) == set(
        re.findall(r"^### Requirement: (.+)$", libero_specification, re.MULTILINE)
    )
    assert set(scenario_tests) == set(
        re.findall(r"^#### Scenario: (.+)$", libero_specification, re.MULTILINE)
    )
    assert set(robomimic_requirement_tests) == set(
        re.findall(r"^### Requirement: (.+)$", robomimic_specification, re.MULTILINE)
    )
    assert set(robomimic_scenario_tests) == set(
        re.findall(r"^#### Scenario: (.+)$", robomimic_specification, re.MULTILINE)
    )
    for test_name in (
        *requirement_tests.values(),
        *scenario_tests.values(),
        *robomimic_requirement_tests.values(),
        *robomimic_scenario_tests.values(),
    ):
        assert callable(globals().get(test_name)), test_name
