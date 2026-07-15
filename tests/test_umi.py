"""Test processed UMI Zarr recognition, lazy reads, and LeRobot conversion."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from leport.api import inspect, validate
from leport.cli import main
from leport.conversion.plan import (
    ConversionPlan,
    FeatureMapping,
    FeatureSpec,
    TargetConfig,
    TaskProvider,
    save_plan,
)
from leport.errors import OptionalDependencyError, SourceSchemaError
from leport.sources import EpisodeSelection, UmiAdapter
from leport.sources.registry import create_default_registry


def test_probe_requires_complete_umi_signature_for_zip_and_directory_stores(
    umi_zip_file: Path,
    umi_directory: Path,
    malformed_umi_sources: dict[str, Path],
) -> None:
    adapter = UmiAdapter()
    zip_result = adapter.probe(umi_zip_file)
    directory_result = adapter.probe(umi_directory)
    assert zip_result.confidence == 100 and "ZipStore" in zip_result.reason
    assert directory_result.confidence == 100 and "DirectoryStore" in directory_result.reason
    for case in ("missing_robot", "missing_camera", "misleading"):
        assert adapter.probe(malformed_umi_sources[case]).confidence == 0


def test_inspection_derives_boundaries_and_uses_only_array_metadata(
    umi_zip_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_reads: list[str] = []
    original_getitem = zarr.core.Array.__getitem__

    def tracking_getitem(array: zarr.core.Array, selection: object) -> object:
        """Track payload access while allowing the small episode-boundary index read."""

        if array.path != "meta/episode_ends":
            payload_reads.append(array.path)
        return original_getitem(array, selection)

    monkeypatch.setattr(zarr.core.Array, "__getitem__", tracking_getitem)
    inspection = UmiAdapter().inspect(umi_zip_file)
    assert payload_reads == []
    assert inspection.episode_ids == ("episode_0", "episode_1", "episode_2")
    assert inspection.episode_lengths == {"episode_0": 3, "episode_1": 4, "episode_2": 2}
    assert inspection.metadata["episode_slices"] == {
        "episode_0": [0, 3],
        "episode_1": [3, 7],
        "episode_2": [7, 9],
    }
    assert "fps" not in inspection.metadata and "task" not in inspection.metadata
    for selector in (
        "action",
        "robot1_eef_pos",
        "robot1_eef_rot_axis_angle",
        "robot1_gripper_width",
    ):
        assert inspection.field(selector) is not None
    assert inspection.field("robot0_eef_pos").shapes == ((3,),)  # type: ignore[union-attr]
    assert inspection.field("robot0_demo_start_pose").dtypes == (  # type: ignore[union-attr]
        "float64",
    )
    for selector in ("camera0_rgb", "camera1_rgb"):
        field = inspection.field(selector)
        assert field is not None and field.image_candidate
        assert field.dtypes == ("uint8",) and field.shapes == ((8, 10, 3),)
    json.dumps(inspection.to_dict())


def test_selection_is_canonical_and_rejects_unknown_ids_and_filters(umi_directory: Path) -> None:
    adapter = UmiAdapter()
    selected = EpisodeSelection(episode_ids=("episode_2", "episode_0"))
    inspection = adapter.inspect(umi_directory, selection=selected)
    assert inspection.episode_ids == ("episode_0", "episode_2")
    assert inspection.episode_lengths == {"episode_0": 3, "episode_2": 2}

    with pytest.raises(SourceSchemaError, match="unknown identifiers") as unknown:
        adapter.inspect(
            umi_directory,
            selection=EpisodeSelection(episode_ids=("episode_9",)),
        )
    assert unknown.value.context == {
        "unknown": ("episode_9",),
        "available": ("episode_0", "episode_1", "episode_2"),
    }
    with pytest.raises(SourceSchemaError, match="do not provide filter keys") as filtered:
        adapter.inspect(umi_directory, selection=EpisodeSelection(filter_key="train"))
    assert filtered.value.context == {"filter_key": "train"}


def test_iteration_reads_only_selected_fields_and_decodes_jpeg_xl(
    umi_zip_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_reads: list[str] = []
    original_getitem = zarr.core.Array.__getitem__

    def tracking_getitem(array: zarr.core.Array, selection: object) -> object:
        """Record frame payload paths while preserving Zarr decoding behavior."""

        if array.path != "meta/episode_ends":
            payload_reads.append(array.path)
        return original_getitem(array, selection)

    monkeypatch.setattr(zarr.core.Array, "__getitem__", tracking_getitem)
    iterator = UmiAdapter().iter_episodes(
        umi_zip_file,
        selection=EpisodeSelection(episode_ids=("episode_2", "episode_0")),
        selectors=(
            "camera0_rgb",
            "robot0_eef_pos",
            "robot0_eef_rot_axis_angle",
            "robot0_gripper_width",
        ),
    )
    first = next(iterator)
    assert first.episode_id == "episode_0" and first.length == 3
    first_frames = list(first.iter_frames())
    assert [frame.index for frame in first_frames] == [0, 1, 2]
    assert set(first_frames[0].fields) == {
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    }
    np.testing.assert_array_equal(
        first_frames[0].fields["camera0_rgb"][0, 0],
        np.asarray([10, 30, 50], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        first_frames[0].fields["robot0_eef_pos"],
        np.asarray([0, 1, 2], dtype=np.float32),
    )

    third = next(iterator)
    assert third.episode_id == "episode_2" and third.length == 2
    third_frames = list(third.iter_frames())
    np.testing.assert_array_equal(
        third_frames[0].fields["camera0_rgb"][0, 0],
        np.asarray([17, 30, 57], dtype=np.uint8),
    )
    iterator.close()
    assert payload_reads
    assert all(
        path
        in {
            "data/camera0_rgb",
            "data/robot0_eef_pos",
            "data/robot0_eef_rot_axis_angle",
            "data/robot0_gripper_width",
        }
        for path in payload_reads
    )

    with pytest.raises(SourceSchemaError, match="field is missing") as missing:
        next(UmiAdapter().iter_episodes(umi_zip_file, selectors=("not_present",)))
    assert missing.value.context["selectors"] == ("not_present",)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_robot", "required robot fields"),
        ("missing_camera", "at least one"),
        ("bad_boundaries", "strictly increasing"),
        ("length_mismatch", "length differs"),
        ("nested", "flat frame arrays"),
        ("misleading", "Could not open"),
    ],
)
def test_malformed_sources_report_precise_schema_failures(
    malformed_umi_sources: dict[str, Path],
    case: str,
    message: str,
) -> None:
    with pytest.raises(SourceSchemaError, match=message):
        UmiAdapter().inspect(malformed_umi_sources[case])


def test_registry_python_api_and_cli_select_umi_automatically(
    umi_zip_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = create_default_registry(discover_plugins=False)
    assert "umi" in registry.names
    assert registry.select(umi_zip_file).name == "umi"
    assert inspect(umi_zip_file).adapter == "umi"
    assert main(["inspect", str(umi_zip_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"] == "umi"
    assert payload["episode_lengths"] == {"episode_0": 3, "episode_1": 4, "episode_2": 2}


def test_cli_conversion_uses_explicit_action_concatenation_and_reload_validation(
    umi_zip_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "umi-target"
    action_sources = (
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    )
    plan = ConversionPlan(
        adapter="umi",
        source=umi_zip_file,
        selection=EpisodeSelection(episode_ids=("episode_0", "episode_2")),
        target=TargetConfig(
            repo_id="tests/umi",
            root=target,
            robot_type="umi-gripper",
            use_videos=False,
        ),
        fps=10,
        task=TaskProvider("static", "arrange the cups"),
        features={
            "observation.state": FeatureSpec("float32", (7,)),
            "observation.images.wrist": FeatureSpec("image", (8, 10, 3)),
            "action": FeatureSpec("float32", (7,)),
        },
        mappings={
            "observation.state": FeatureMapping(action_sources, operation="concat"),
            "observation.images.wrist": FeatureMapping(("camera0_rgb",)),
            "action": FeatureMapping(action_sources, operation="concat"),
        },
    )
    plan_path = tmp_path / "umi-plan.yaml"
    save_plan(plan, plan_path)
    assert main(["convert", "--config", str(plan_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_episodes"] == 2 and result["total_frames"] == 5
    assert validate(target, plan=plan).episode_lengths == (3, 2)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=target)
    np.testing.assert_array_equal(
        dataset[0]["action"].numpy(),
        np.asarray([0, 1, 2, 100, 101, 102, 0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        dataset[0]["observation.images.wrist"].numpy()[:, 0, 0],
        np.asarray([10, 30, 50], dtype=np.float32) / 255,
        atol=1 / 255,
    )


def test_umi_optional_dependencies_do_not_block_other_adapters(
    umi_zip_file: Path,
    robomimic_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = create_default_registry(discover_plugins=False)
    monkeypatch.setitem(sys.modules, "zarr", None)
    assert registry.get("umi").name == "umi"
    with pytest.raises(OptionalDependencyError, match="uv sync --extra umi") as caught:
        registry.get("umi").probe(umi_zip_file)
    assert caught.value.context == {
        "adapter": "umi",
        "extra": "umi",
        "dependency": "zarr,imagecodecs",
    }
    assert registry.select(robomimic_file).name == "robomimic"


def test_umi_documentation_notebook_and_readme_match_supported_scope() -> None:
    repository = Path(__file__).parents[1]
    guide = (repository / "docs/umi.md").read_text(encoding="utf-8")
    readme = (repository / "README.md").read_text(encoding="utf-8")
    notebook_path = repository / "notebooks/umi.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"{notebook_path.name}:{cell['id']}", "exec")
        assert cell["execution_count"] is None and cell["outputs"] == []

    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for text in (
        "data/umi/cup_in_the_wild.zarr.zip",
        "meta/episode_ends",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "RUN_MERGE = False",
        "## Equivalent CLI commands",
        "uv run leport inspect",
        "uv run leport convert",
        "uv run leport validate",
        "uv run leport merge",
    ):
        assert text in notebook_source
    assert "create_dataset(" not in notebook_source
    assert "import zarr" not in notebook_source
    assert "RUN_WORKFLOW" not in notebook_source

    for text in (
        "Zarr v2",
        "imagecodecs_jpegxl",
        "episode_ends",
        "operation: concat",
        "Raw GoPro",
    ):
        assert text in guide
    assert "docs/umi.md" in readme and "notebooks/umi.ipynb" in readme
    assert "Universal Manipulation Interface (UMI)" in readme
    assert "DROID" not in readme and "Open X-Embodiment" not in readme


def test_every_umi_requirement_and_scenario_has_an_automated_test_mapping() -> None:
    probe = "test_probe_requires_complete_umi_signature_for_zip_and_directory_stores"
    inspection = "test_inspection_derives_boundaries_and_uses_only_array_metadata"
    selection = "test_selection_is_canonical_and_rejects_unknown_ids_and_filters"
    iteration = "test_iteration_reads_only_selected_fields_and_decodes_jpeg_xl"
    malformed = "test_malformed_sources_report_precise_schema_failures"
    integration = "test_registry_python_api_and_cli_select_umi_automatically"
    conversion = "test_cli_conversion_uses_explicit_action_concatenation_and_reload_validation"
    dependencies = "test_umi_optional_dependencies_do_not_block_other_adapters"
    documentation = "test_umi_documentation_notebook_and_readme_match_supported_scope"
    requirement_tests = {
        "Recognize processed UMI Zarr replay buffers": probe,
        "Validate cumulative episode boundaries and aligned fields": malformed,
        "Select UMI episodes deterministically": selection,
        "Inspect flat UMI data arrays without reading frame payloads": inspection,
        "Lazily read selected UMI fields and decode Zarr codecs": iteration,
        "Preserve UMI values without inferring policy semantics": conversion,
        "Isolate UMI storage dependencies": dependencies,
        "Document the processed UMI workflow in English": documentation,
    }
    scenario_tests = {
        "Official ZipStore is recognized": probe,
        "Directory store is recognized": probe,
        "Generic or malformed store is rejected": probe,
        "Extension is misleading": probe,
        "Boundaries describe multiple episodes": inspection,
        "Boundaries are invalid": malformed,
        "Data length differs from the replay buffer": malformed,
        "Data member is not frame-addressable": malformed,
        "Every episode is selected": inspection,
        "Explicit subset is reordered canonically": selection,
        "Episode ID is unknown": selection,
        "Filter key is requested": selection,
        "Robot fields are inspected": inspection,
        "Camera field is inspected": inspection,
        "Several robots or cameras are present": inspection,
        "Only selected fields are converted": iteration,
        "Explicit episode subset is converted": iteration,
        "JPEG XL camera frame is read": iteration,
        "Selected field is unavailable": iteration,
        "Action is constructed explicitly by a plan": conversion,
        "Task and FPS are absent": inspection,
        "Stored action exists": inspection,
        "UMI dependencies are installed": integration,
        "UMI dependencies are missing": dependencies,
        "Another adapter is used": dependencies,
        "Reader starts with processed UMI data": documentation,
        "Reader checks project scope": documentation,
    }
    specification = (
        Path(__file__).parents[1]
        / "openspec/specs/umi-source-adapter/spec.md"
    ).read_text(encoding="utf-8")
    assert set(requirement_tests) == set(
        re.findall(r"^### Requirement: (.+)$", specification, re.MULTILINE)
    )
    assert set(scenario_tests) == set(
        re.findall(r"^#### Scenario: (.+)$", specification, re.MULTILINE)
    )
    for test_name in (*requirement_tests.values(), *scenario_tests.values()):
        assert callable(globals().get(test_name)), test_name
