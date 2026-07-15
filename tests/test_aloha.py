"""Test ALOHA discovery, inspection, lazy reads, planning, and conversion integration."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from leport.api import convert, create_plan, inspect, validate
from leport.cli import main
from leport.conversion.pipeline import preflight
from leport.errors import OptionalDependencyError, PlanValidationError, SourceSchemaError
from leport.sources.aloha import AlohaAdapter
from leport.sources.registry import create_default_registry
from leport.sources.robomimic import RobomimicAdapter
from leport.sources.types import EpisodeSelection


def test_probe_resolves_files_directories_and_structural_matches(
    aloha_directory: Path,
    robomimic_file: Path,
    tmp_path: Path,
) -> None:
    adapter = AlohaAdapter()
    assert adapter.probe(aloha_directory).confidence == 100
    assert adapter.inspect(aloha_directory).episode_ids == ("episode_0", "episode_2", "episode_10")

    single = tmp_path / "episode_7.hdf5"
    shutil.copyfile(aloha_directory / "episode_0.hdf5", single)
    single_inspection = adapter.inspect(single)
    assert single_inspection.episode_ids == ("episode_7",)
    assert single_inspection.metadata["episode_files"] == {"episode_7": "episode_7.hdf5"}

    generic = tmp_path / "episode_8.hdf5"
    with h5py.File(generic, "w") as h5_file:
        h5_file.create_dataset("unrelated", data=np.zeros(1, dtype=np.float32))
    generic_result = adapter.probe(generic)
    assert generic_result.confidence == 0
    assert "action" in generic_result.reason

    empty = tmp_path / "empty"
    empty.mkdir()
    assert adapter.probe(empty).confidence == 0
    assert "no `episode_<integer>.hdf5`" in adapter.probe(empty).reason

    registry = create_default_registry(discover_plugins=False)
    assert registry.select(aloha_directory).name == "aloha"
    assert registry.select(robomimic_file).name == "robomimic"


def test_duplicate_numeric_ids_and_selection_errors_are_precise(
    aloha_directory: Path,
    tmp_path: Path,
) -> None:
    duplicate_directory = tmp_path / "duplicate"
    duplicate_directory.mkdir()
    (duplicate_directory / "episode_1.hdf5").write_text("first", encoding="utf-8")
    (duplicate_directory / "episode_01.hdf5").write_text("second", encoding="utf-8")
    with pytest.raises(SourceSchemaError, match="duplicate numeric") as duplicate_error:
        AlohaAdapter().inspect(duplicate_directory)
    assert duplicate_error.value.context["numeric_id"] == 1

    adapter = AlohaAdapter()
    selected = adapter.inspect(
        aloha_directory,
        selection=EpisodeSelection(episode_ids=("episode_10", "episode_0")),
    )
    assert selected.episode_ids == ("episode_0", "episode_10")
    with pytest.raises(SourceSchemaError, match="unknown ALOHA IDs") as unknown_error:
        adapter.inspect(
            aloha_directory,
            selection=EpisodeSelection(episode_ids=("episode_999",)),
        )
    assert unknown_error.value.context == {
        "unknown": ["episode_999"],
        "available": ["episode_0", "episode_2", "episode_10"],
    }
    with pytest.raises(SourceSchemaError, match="do not support filter keys") as filter_error:
        adapter.inspect(aloha_directory, selection=EpisodeSelection(filter_key="train"))
    assert filter_error.value.context == {"filter_key": "train"}


def test_inspection_reports_decoded_schema_coverage_and_plain_metadata(
    aloha_directory: Path,
) -> None:
    inspection = AlohaAdapter().inspect(aloha_directory)
    assert inspection.episode_lengths == {"episode_0": 3, "episode_2": 2, "episode_10": 2}
    assert inspection.total_frames == 7
    assert inspection.field("action").dtypes == ("float64",)  # type: ignore[union-attr]
    assert inspection.field("observations/qpos").shapes == ((6,),)  # type: ignore[union-attr]
    effort = inspection.field("observations/effort")
    assert effort is not None
    assert effort.missing_episodes == ("episode_2",)
    assert not effort.schema_consistent

    for selector in ("observations/images/cam_high", "observations/images/cam_wrist"):
        camera = inspection.field(selector)
        assert camera is not None and camera.image_candidate and camera.schema_consistent
        assert camera.dtypes == ("uint8",)
        assert camera.shapes == ((16, 16, 3),)
    assert inspection.field("compress_len") is None
    assert inspection.metadata["episode_files"]["episode_10"] == "episode_10.hdf5"
    assert inspection.metadata["episode_attributes"]["episode_0"]["sim"] is True
    assert inspection.metadata["episode_attributes"]["episode_2"]["compress"] is True
    assert inspection.metadata["episode_attributes"]["episode_0"]["camera_names"] == [
        "cam_high",
        "cam_wrist",
    ]
    assert set(inspection.metadata["compression_lengths"]) == {"episode_2", "episode_10"}
    assert not {"fps", "task", "robot_type", "action_meaning"} & set(inspection.metadata)
    json.dumps(inspection.to_dict())


def test_inspection_reports_decoded_shape_drift(aloha_directory: Path) -> None:
    episode_path = aloha_directory / "episode_10.hdf5"
    with h5py.File(episode_path, "r+") as h5_file:
        del h5_file["observations/images/cam_high"]
        h5_file["observations/images"].create_dataset(
            "cam_high",
            data=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        )
    inspection = AlohaAdapter().inspect(aloha_directory)
    camera = inspection.field("observations/images/cam_high")
    assert camera is not None
    assert camera.shapes == ((8, 8, 3), (16, 16, 3))
    assert not camera.schema_consistent
    assert any("cam_high" in diagnostic for diagnostic in inspection.diagnostics)


def test_required_paths_jpeg_buffers_and_raw_layout_fail_with_context(
    malformed_aloha_directory: Path,
) -> None:
    adapter = AlohaAdapter()
    missing_action = adapter.probe(malformed_aloha_directory / "episode_20.hdf5")
    assert missing_action.confidence == 0
    assert "action" in missing_action.reason
    with pytest.raises(SourceSchemaError, match="observations/qpos") as qpos_error:
        adapter.inspect(malformed_aloha_directory / "episode_21.hdf5")
    assert qpos_error.value.context["episode"] == "episode_21"

    with pytest.raises(SourceSchemaError, match="decode ALOHA JPEG") as jpeg_error:
        adapter.inspect(malformed_aloha_directory / "episode_24.hdf5")
    assert jpeg_error.value.context["episode"] == "episode_24"
    assert jpeg_error.value.context["frame"] == 0
    assert jpeg_error.value.context["selector"] == "observations/images/cam_high"

    with pytest.raises(SourceSchemaError, match="1, 3, or 4 channels") as layout_error:
        adapter.inspect(malformed_aloha_directory / "episode_25.hdf5")
    assert layout_error.value.context["selector"] == "observations/images/cam_high"
    with pytest.raises(SourceSchemaError, match="Could not open"):
        adapter.inspect(malformed_aloha_directory / "episode_26.hdf5")


def test_lazy_iteration_preserves_values_rgb_order_metadata_and_numeric_order(
    aloha_directory: Path,
) -> None:
    episodes = AlohaAdapter().iter_episodes(
        aloha_directory,
        selection=EpisodeSelection(episode_ids=("episode_10", "episode_0")),
        selectors=("action", "observations/qpos", "observations/images/cam_high"),
    )
    first = next(episodes)
    assert first.episode_id == "episode_0"
    assert first.metadata["source_filename"] == "episode_0.hdf5"
    assert first.metadata["instruction"] == "move object in episode 0"
    first_frame = next(iter(first.iter_frames()))
    assert set(first_frame.fields) == {
        "action",
        "observations/qpos",
        "observations/images/cam_high",
    }
    np.testing.assert_array_equal(first_frame.fields["action"], np.arange(7, dtype=np.float64))
    high = first_frame.fields["observations/images/cam_high"]
    assert high.shape == (16, 16, 3) and high.dtype == np.uint8
    assert int(high[..., 0].mean()) > 200
    assert int(high[..., 2].mean()) < 40

    second = next(episodes)
    assert second.episode_id == "episode_10"
    second_frame = next(iter(second.iter_frames()))
    np.testing.assert_array_equal(
        second_frame.fields["action"],
        np.arange(7, dtype=np.float64) + 1000,
    )
    with pytest.raises(StopIteration):
        next(episodes)


@pytest.mark.parametrize("episode_id", ["episode_2", "episode_10"])
def test_fixed_and_variable_jpeg_cameras_decode_as_rgb(
    aloha_directory: Path,
    episode_id: str,
) -> None:
    episodes = AlohaAdapter().iter_episodes(
        aloha_directory,
        selection=EpisodeSelection(episode_ids=(episode_id,)),
        selectors=("observations/images/cam_high", "observations/images/cam_wrist"),
    )
    episode = next(episodes)
    frame = next(iter(episode.iter_frames()))
    high = frame.fields["observations/images/cam_high"]
    wrist = frame.fields["observations/images/cam_wrist"]
    assert high.dtype == wrist.dtype == np.uint8
    assert high.shape == wrist.shape == (16, 16, 3)
    assert int(high[..., 0].mean()) > int(high[..., 2].mean()) + 150
    assert int(wrist[..., 2].mean()) > int(wrist[..., 0].mean()) + 150
    episodes.close()


def test_episode_handle_closes_when_iteration_advances(
    aloha_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_file = h5py.File
    active_paths: list[Path] = []
    closed_paths: list[Path] = []

    class TrackingFile:
        """Delegate HDF5 access while recording context-manager lifetime transitions."""

        def __init__(self, path: str | Path, mode: str) -> None:
            self.path = Path(path)
            self.file = real_file(path, mode)
            active_paths.append(self.path)

        def __enter__(self) -> TrackingFile:
            return self

        def __exit__(self, *args: object) -> None:
            self.file.close()
            active_paths.remove(self.path)
            closed_paths.append(self.path)

        def __getattr__(self, name: str) -> object:
            return getattr(self.file, name)

        def __getitem__(self, name: str) -> object:
            return self.file[name]

    monkeypatch.setattr(h5py, "File", TrackingFile)
    episodes = AlohaAdapter().iter_episodes(
        aloha_directory,
        selectors=("action",),
    )
    first = next(episodes)
    assert active_paths == [aloha_directory / "episode_0.hdf5"]
    assert len(list(first.iter_frames())) == 3
    second = next(episodes)
    assert second.episode_id == "episode_2"
    assert closed_paths == [aloha_directory / "episode_0.hdf5"]
    assert active_paths == [aloha_directory / "episode_2.hdf5"]
    episodes.close()
    assert active_paths == []
    assert closed_paths == [
        aloha_directory / "episode_0.hdf5",
        aloha_directory / "episode_2.hdf5",
    ]


def test_only_selected_fields_are_read_and_length_mismatch_is_never_adjusted(
    aloha_directory: Path,
    malformed_aloha_directory: Path,
) -> None:
    episode_path = aloha_directory / "episode_2.hdf5"
    with h5py.File(episode_path, "r+") as h5_file:
        del h5_file["observations/images/cam_wrist"]
        h5_file["observations/images"].create_dataset(
            "cam_wrist", data=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        )
    episodes = AlohaAdapter().iter_episodes(
        episode_path,
        selectors=("action", "observations/images/cam_high"),
    )
    episode = next(episodes)
    frames = list(episode.iter_frames())
    assert len(frames) == 2
    assert all(set(frame.fields) == {"action", "observations/images/cam_high"} for frame in frames)
    episodes.close()

    mismatched = AlohaAdapter().iter_episodes(
        malformed_aloha_directory / "episode_23.hdf5",
        selectors=("action", "observations/qpos"),
    )
    with pytest.raises(SourceSchemaError, match="does not truncate or pad") as mismatch_error:
        next(mismatched)
    assert mismatch_error.value.context == {
        "episode": "episode_23",
        "selector": "observations/qpos",
        "action_length": 2,
        "field_length": 3,
    }


def test_explicit_subset_opens_only_the_selected_episode_file(aloha_directory: Path) -> None:
    with patch.object(h5py, "File", wraps=h5py.File) as tracked_file:
        episodes = AlohaAdapter().iter_episodes(
            aloha_directory,
            selection=EpisodeSelection(episode_ids=("episode_10",)),
            selectors=("action",),
        )
        episode = next(episodes)
        assert len(list(episode.iter_frames())) == 2
        episodes.close()
    assert [Path(call.args[0]).name for call in tracked_file.call_args_list] == ["episode_10.hdf5"]


def test_empty_action_is_rejected_during_preflight(
    malformed_aloha_directory: Path,
    tmp_path: Path,
) -> None:
    source = malformed_aloha_directory / "episode_22.hdf5"
    plan = create_plan(
        source,
        target_root=tmp_path / "empty-target",
        repo_id="tests/aloha-empty",
        fps=50,
        task="move object",
        action_source="action",
        state_sources=("observations/qpos",),
        use_videos=False,
        adapter="aloha",
    )
    with pytest.raises(PlanValidationError, match="Empty episodes") as error:
        preflight(plan)
    assert error.value.context == {"episode": "episode_22"}


@pytest.mark.parametrize(
    ("source_kind", "adapter_args", "episode_args", "expected_ids"),
    [
        ("single", ["--adapter", "aloha"], [], ["episode_0"]),
        ("directory", [], [], ["episode_0", "episode_2", "episode_10"]),
        (
            "directory",
            [],
            ["--episode", "episode_10,episode_0"],
            ["episode_0", "episode_10"],
        ),
    ],
)
def test_cli_inspect_supports_explicit_auto_and_subset_workflows(
    aloha_directory: Path,
    capsys: pytest.CaptureFixture[str],
    source_kind: str,
    adapter_args: list[str],
    episode_args: list[str],
    expected_ids: list[str],
) -> None:
    source = aloha_directory / "episode_0.hdf5" if source_kind == "single" else aloha_directory
    assert main(["inspect", str(source), *adapter_args, *episode_args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["episode_ids"] == expected_ids


@pytest.mark.parametrize(
    ("source_kind", "adapter_args", "episode_args", "expected_ids"),
    [
        ("single", ["--adapter", "aloha"], [], []),
        ("directory", [], [], []),
        ("directory", [], ["--episode", "episode_10,episode_0"], ["episode_10", "episode_0"]),
    ],
)
def test_cli_plan_supports_one_file_directory_and_comma_subset(
    aloha_directory: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source_kind: str,
    adapter_args: list[str],
    episode_args: list[str],
    expected_ids: list[str],
) -> None:
    source = aloha_directory / "episode_0.hdf5" if source_kind == "single" else aloha_directory
    suffix = f"{source_kind}-{len(episode_args)}-{len(adapter_args)}"
    plan_path = tmp_path / f"{suffix}.yaml"
    target = tmp_path / f"{suffix}-target"
    assert (
        main(
            [
                "plan",
                "--source",
                str(source),
                "--output",
                str(plan_path),
                *adapter_args,
                *episode_args,
                "--target",
                str(target),
                "--repo-id",
                f"tests/{suffix}",
                "--robot-type",
                "aloha-test",
                "--fps",
                "50",
                "--task",
                "move object",
                "--action",
                "action",
                "--action-dtype",
                "float32",
                "--state",
                "observations/qvel",
                "--state",
                "observations/qpos",
                "--state-dtype",
                "float32",
                "--image",
                "observations/images/cam_high=observation.images.high",
                "--no-videos",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"] == "aloha"
    assert payload["selection"]["episode_ids"] == expected_ids
    assert plan_path.is_file()


def test_create_plan_keeps_explicit_semantics_casts_state_order_and_image_names(
    aloha_directory: Path,
    tmp_path: Path,
) -> None:
    plan = create_plan(
        aloha_directory,
        target_root=tmp_path / "planned-target",
        repo_id="tests/aloha-plan",
        robot_type="aloha-test",
        fps=50,
        task="move object",
        action_source="action",
        action_dtype="float32",
        state_sources=("observations/qvel", "observations/qpos"),
        state_dtype="float32",
        image_sources={
            "observations/images/cam_high": "observation.images.high",
            "observations/images/cam_wrist": "observation.images.wrist",
        },
        use_videos=False,
        adapter="aloha",
        selection=EpisodeSelection(episode_ids=("episode_10", "episode_0")),
    )
    assert plan.adapter == "aloha"
    assert plan.fps == 50 and plan.task.value == "move object"
    assert plan.target.robot_type == "aloha-test"
    assert plan.mappings["action"].sources == ("action",)
    assert plan.mappings["action"].cast == "float32"
    assert plan.mappings["observation.state"].sources == (
        "observations/qvel",
        "observations/qpos",
    )
    assert plan.mappings["observation.state"].cast == "float32"
    assert plan.features["observation.state"].shape == (12,)
    assert plan.features["observation.images.high"].dtype == "image"
    assert plan.features["observation.images.wrist"].shape == (16, 16, 3)


def test_numeric_conversion_and_source_aware_validation(
    aloha_directory: Path,
    tmp_path: Path,
) -> None:
    plan = create_plan(
        aloha_directory,
        target_root=tmp_path / "numeric-target",
        repo_id="tests/aloha-numeric",
        robot_type="aloha-test",
        fps=50,
        task="move object",
        action_source="action",
        action_dtype="float32",
        state_sources=("observations/qpos",),
        state_dtype="float32",
        use_videos=False,
        adapter="aloha",
    )
    result = convert(plan)
    assert result.validation.total_episodes == 3
    assert result.validation.total_frames == 7
    assert result.validation.episode_lengths == (3, 2, 2)
    source_aware = validate(result.target, plan=plan)
    assert source_aware.episode_lengths == (3, 2, 2)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=plan.target.repo_id, root=plan.target.root)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(
        dataset[3]["action"].numpy(),
        np.arange(7, dtype=np.float32) + 200,
    )


@pytest.mark.parametrize(
    ("episode_id", "use_videos", "expected_frames"),
    [
        ("episode_0", False, 3),
        ("episode_0", True, 3),
        ("episode_2", False, 2),
        ("episode_2", True, 2),
    ],
)
def test_raw_and_jpeg_multi_camera_image_and_video_conversion(
    aloha_directory: Path,
    tmp_path: Path,
    episode_id: str,
    use_videos: bool,
    expected_frames: int,
) -> None:
    target_kind = "video" if use_videos else "image"
    plan = create_plan(
        aloha_directory,
        target_root=tmp_path / f"{episode_id}-{target_kind}",
        repo_id=f"tests/aloha-{episode_id}-{target_kind}",
        fps=50,
        task="move object",
        action_source="action",
        action_dtype="float32",
        image_sources={
            "observations/images/cam_high": "observation.images.high",
            "observations/images/cam_wrist": "observation.images.wrist",
        },
        use_videos=use_videos,
        adapter="aloha",
        selection=EpisodeSelection(episode_ids=(episode_id,)),
    )
    result = convert(plan)
    assert result.validation.total_episodes == 1
    assert result.validation.total_frames == expected_frames
    assert result.validation.episode_lengths == (expected_frames,)
    assert set(result.validation.decoded_visual_features) == {
        "observation.images.high",
        "observation.images.wrist",
    }
    assert result.validation.features["observation.images.high"]["dtype"] == target_kind


def test_aloha_dependencies_are_isolated_from_core_and_robomimic(
    aloha_directory: Path,
    robomimic_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = create_default_registry(discover_plugins=False)
    assert registry.names == ("aloha", "libero", "maniskill", "robomimic", "umi")

    # Adapter construction imports no HDF5 module; the actual probe provides the actionable error.
    monkeypatch.setitem(sys.modules, "h5py", None)
    assert registry.get("aloha").name == "aloha"
    with pytest.raises(OptionalDependencyError, match="uv sync --extra aloha") as hdf5_error:
        registry.get("aloha").probe(aloha_directory)
    assert hdf5_error.value.context == {
        "adapter": "aloha",
        "extra": "aloha",
        "dependency": "h5py",
    }
    monkeypatch.undo()

    # Pillow is needed only for encoded ALOHA frames. Raw ALOHA and robomimic inspection stay usable.
    monkeypatch.setitem(sys.modules, "PIL", None)
    raw = AlohaAdapter().inspect(aloha_directory / "episode_0.hdf5")
    assert raw.episode_ids == ("episode_0",)
    with pytest.raises(OptionalDependencyError, match="uv sync --extra aloha") as pillow_error:
        AlohaAdapter().inspect(aloha_directory / "episode_2.hdf5")
    assert pillow_error.value.context["dependency"] == "pillow"
    assert RobomimicAdapter().inspect(robomimic_file).episode_ids == (
        "demo_0",
        "demo_2",
        "demo_10",
    )
    assert inspect(robomimic_file, adapter="robomimic").adapter == "robomimic"


def test_documentation_and_notebook_use_supported_fixture_selectors() -> None:
    repository = Path(__file__).parents[1]
    guide = (repository / "docs/aloha.md").read_text(encoding="utf-8")
    notebook_text = (repository / "notebooks/aloha.ipynb").read_text(encoding="utf-8")
    readme = (repository / "README.md").read_text(encoding="utf-8")
    for selector in ("action", "observations/qpos", "observations/qvel", "observations/images/cam_high"):
        assert selector in guide
        assert selector in notebook_text
    assert "observations/images/cam_wrist" in guide
    assert "observations/images/cam_left_wrist" in notebook_text
    assert "observations/images/cam_right_wrist" in notebook_text
    assert "--episode episode_0,episode_10" in guide
    assert "--filter-key" in guide and "unsupported" in guide
    assert "--extra aloha" in guide
    assert "docs/aloha.md" in readme and "notebooks/aloha.ipynb" in readme
    assert "uv run" not in readme
    assert "## Equivalent CLI commands" in notebook_text
    for command in ("inspect", "plan", "convert", "validate", "merge"):
        assert f"uv run leport {command}" in notebook_text


def test_notebook_json_ids_compile_and_only_merge_is_opt_in() -> None:
    notebook_path = Path(__file__).parents[1] / "notebooks/aloha.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"{notebook_path.name}:{cell['id']}", "exec")
    assert "RUN_WORKFLOW" not in notebook_source
    assert notebook_source.count("RUN_MERGE = False") == 1
    assert notebook_source.count("if RUN_MERGE:") == 1
    assert "data/aloha/episode_0.hdf5" in notebook_source
    assert "data/aloha/episode_1.hdf5" in notebook_source


def test_every_requirement_and_scenario_has_an_automated_test_mapping() -> None:
    probe = "test_probe_resolves_files_directories_and_structural_matches"
    selection = "test_duplicate_numeric_ids_and_selection_errors_are_precise"
    inspection = "test_inspection_reports_decoded_schema_coverage_and_plain_metadata"
    alignment = "test_only_selected_fields_are_read_and_length_mismatch_is_never_adjusted"
    jpeg = "test_fixed_and_variable_jpeg_cameras_decode_as_rgb"
    lazy = "test_episode_handle_closes_when_iteration_advances"
    semantics = "test_create_plan_keeps_explicit_semantics_casts_state_order_and_image_names"
    dependencies = "test_aloha_dependencies_are_isolated_from_core_and_robomimic"
    numeric = "test_numeric_conversion_and_source_aware_validation"
    rgb_order = "test_lazy_iteration_preserves_values_rgb_order_metadata_and_numeric_order"
    requirement_tests = {
        "Recognize standard ALOHA HDF5 sources": probe,
        "Select ALOHA episodes deterministically": selection,
        "Inspect ALOHA fields across selected episodes": inspection,
        "Enforce action-aligned field lengths": alignment,
        "Decode raw and JPEG camera fields": jpeg,
        "Read ALOHA episodes and frames lazily": lazy,
        "Preserve metadata without inferring robot semantics": semantics,
        "Isolate ALOHA optional dependencies": dependencies,
    }
    scenario_tests = {
        "Single episode file is recognized": probe,
        "Episode directory is recognized": probe,
        "Unrelated directory entries are ignored": probe,
        "Generic HDF5 is rejected": probe,
        "Directory has no episode candidates": probe,
        "Numeric order differs from lexical order": rgb_order,
        "Explicit subset is selected": selection,
        "Explicit ID is unknown": selection,
        "Filter key is requested": selection,
        "Standard numeric fields are inspected": inspection,
        "Camera fields are inspected": inspection,
        "Optional field is missing from one episode": inspection,
        "Compression metadata is present": inspection,
        "Selected fields align with actions": numeric,
        "Observation length differs": alignment,
        "Empty action dataset is selected": "test_empty_action_is_rejected_during_preflight",
        "Uncompressed camera is read": rgb_order,
        "Fixed-width JPEG rows are read": jpeg,
        "Variable-length JPEG values are read": jpeg,
        "Compressed camera schema is inspected": inspection,
        "JPEG data is malformed": "test_required_paths_jpeg_buffers_and_raw_layout_fail_with_context",
        "A small explicit subset is converted": "test_raw_and_jpeg_multi_camera_image_and_video_conversion",
        "Only one camera is mapped": alignment,
        "Episode iteration advances": lazy,
        "Standard attributes are available": inspection,
        "Dataset has no task text or FPS": semantics,
        "Action values are read": numeric,
        "ALOHA dependencies are installed": probe,
        "ALOHA dependency is missing": dependencies,
        "Unrelated adapter is used": dependencies,
    }
    specification = (Path(__file__).parents[1] / "openspec/specs/aloha-source-adapter/spec.md").read_text(
        encoding="utf-8"
    )
    assert set(requirement_tests) == set(re.findall(r"^### Requirement: (.+)$", specification, re.MULTILINE))
    assert set(scenario_tests) == set(re.findall(r"^#### Scenario: (.+)$", specification, re.MULTILINE))
    for test_name in (*requirement_tests.values(), *scenario_tests.values()):
        assert callable(globals().get(test_name)), test_name


def test_repository_authored_text_is_english_only() -> None:
    repository = Path(__file__).parents[1]
    excluded_directories = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "dist"}
    text_suffixes = {".ipynb", ".json", ".lock", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    violations: list[str] = []
    for path in repository.rglob("*"):
        if not path.is_file() or excluded_directories & set(path.relative_to(repository).parts):
            continue
        if path.suffix not in text_suffixes and path.name not in {".gitignore", ".python-version"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
            violations.append(str(path.relative_to(repository)))
    assert violations == []
