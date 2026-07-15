"""Test safe merging, compatibility, and visual reloads for existing LeRobot datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from leport import merge
from leport.api import convert, create_plan
from leport.errors import MergeError
from leport.sources import EpisodeSelection
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_merge_preserves_source_order_and_remaps_tasks(
    compatible_lerobot_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Input order controls episode order while preserving frame-level task resolution."""

    first, second = compatible_lerobot_sources
    target = tmp_path / "merged"
    result = merge(
        (first, second),
        target_root=target,
        repo_id="tests/merged",
        concatenate_videos=False,
        concatenate_data=False,
    )

    assert result.sources == (first.resolve(), second.resolve())
    assert result.target == target.resolve()
    assert result.validation.episode_lengths == (3, 2, 4)
    assert result.validation.total_episodes == 3
    assert result.validation.total_frames == 9
    assert result.validation.tasks == ("lift the cube", "pick up the can")
    assert result.to_dict()["validation"]["root"] == str(target.resolve())

    dataset = LeRobotDataset(repo_id="tests/merged", root=target)
    assert dataset[0]["task"] == "lift the cube"
    assert dataset[5]["task"] == "pick up the can"

    reversed_result = merge(
        (second, first),
        target_root=tmp_path / "merged-reversed",
        repo_id="tests/merged-reversed",
        concatenate_videos=False,
        concatenate_data=False,
    )
    assert reversed_result.validation.episode_lengths == (4, 3, 2)


@pytest.mark.parametrize("source_mode", ["single", "duplicate"])
def test_merge_rejects_too_few_or_duplicate_sources(
    compatible_lerobot_sources: tuple[Path, Path],
    tmp_path: Path,
    source_mode: str,
) -> None:
    """One input or duplicate inputs are caller errors and cannot create repeated episodes."""

    first, _ = compatible_lerobot_sources
    sources = (first,) if source_mode == "single" else (first, first)
    target = tmp_path / f"invalid-{source_mode}"
    with pytest.raises(MergeError):
        merge(sources, target_root=target, repo_id="tests/invalid")
    assert not target.exists()


def test_merge_preserves_non_empty_target_and_sources(
    compatible_lerobot_sources: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Rejecting a non-empty target preserves its marker and all input metadata."""

    first, second = compatible_lerobot_sources
    first_info = (first / "meta" / "info.json").read_bytes()
    second_info = (second / "meta" / "info.json").read_bytes()
    target = tmp_path / "occupied"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(MergeError, match="not empty"):
        merge((first, second), target_root=target, repo_id="tests/occupied")

    assert marker.read_text(encoding="utf-8") == "keep"
    assert (first / "meta" / "info.json").read_bytes() == first_info
    assert (second / "meta" / "info.json").read_bytes() == second_info


def test_merge_rejects_incompatible_fps_and_cleans_staging(
    compatible_lerobot_sources: tuple[Path, Path],
    robomimic_file: Path,
    tmp_path: Path,
) -> None:
    """Official compatibility failures become merge_error without partial output or staging residue."""

    first, _ = compatible_lerobot_sources
    incompatible_target = tmp_path / "source-fps-30"
    incompatible_plan = create_plan(
        robomimic_file,
        target_root=incompatible_target,
        repo_id="tests/source-fps-30",
        fps=30,
        task="lift the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        use_videos=False,
        adapter="robomimic",
        selection=EpisodeSelection(episode_ids=("demo_2",)),
    )
    convert(incompatible_plan)

    target = tmp_path / "incompatible-merged"
    with pytest.raises(MergeError) as caught:
        merge((first, incompatible_target), target_root=target, repo_id="tests/incompatible")

    assert "Same fps is expected" in str(caught.value.context["reason"])
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.leport-merge-*"))


def test_merge_preserves_two_video_features_and_decodes_episode_boundaries(
    robomimic_file: Path,
    tmp_path: Path,
) -> None:
    """Both cameras remain decodable at merged episode boundaries when video shards stay separate."""

    image_sources = {
        "obs/agentview_image": "observation.images.agentview",
        "obs/robot0_eye_in_hand_image": "observation.images.wrist",
    }
    first_target = tmp_path / "video-source-first"
    second_target = tmp_path / "video-source-second"
    first_plan = create_plan(
        robomimic_file,
        target_root=first_target,
        repo_id="tests/video-source-first",
        fps=20,
        task="lift the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        image_sources=image_sources,
        use_videos=True,
        adapter="robomimic",
        selection=EpisodeSelection(episode_ids=("demo_0",)),
    )
    second_plan = create_plan(
        robomimic_file,
        target_root=second_target,
        repo_id="tests/video-source-second",
        fps=20,
        task="pick up the can",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        image_sources=image_sources,
        use_videos=True,
        adapter="robomimic",
        selection=EpisodeSelection(episode_ids=("demo_2",)),
    )
    convert(first_plan)
    convert(second_plan)

    target = tmp_path / "video-merged"
    result = merge(
        (first_target, second_target),
        target_root=target,
        repo_id="tests/video-merged",
        concatenate_videos=False,
        concatenate_data=False,
    )

    assert result.validation.episode_lengths == (3, 4)
    assert result.validation.decoded_visual_features == (
        "observation.images.agentview",
        "observation.images.wrist",
    )
    assert len(list((target / "videos" / "observation.images.agentview").rglob("*.mp4"))) == 2
    assert len(list((target / "videos" / "observation.images.wrist").rglob("*.mp4"))) == 2
