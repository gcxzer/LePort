"""Test robomimic HDF5 structure, selection, metadata, and lazy reading."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from leport.errors import OptionalDependencyError, SourceSchemaError
from leport.sources.robomimic import RobomimicAdapter
from leport.sources.types import EpisodeSelection


def test_probe_checks_hdf5_contents(robomimic_file: Path, tmp_path: Path) -> None:
    adapter = RobomimicAdapter()
    assert adapter.probe(robomimic_file).confidence == 100

    ordinary_hdf5 = tmp_path / "ordinary.bin"
    with h5py.File(ordinary_hdf5, "w") as h5_file:
        h5_file.create_dataset("anything", data=np.zeros(1))
    result = adapter.probe(ordinary_hdf5)
    assert result.confidence == 0
    assert "data" in result.reason


def test_missing_h5py_reports_optional_dependency(
    robomimic_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replacing the cached module with None simulates a core-only installation even when the test
    # process imported h5py earlier. monkeypatch restores the original module afterward.
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(OptionalDependencyError, match="uv sync --extra robomimic"):
        RobomimicAdapter().probe(robomimic_file)


def test_episode_sorting_and_mask_selection(robomimic_file: Path) -> None:
    adapter = RobomimicAdapter()
    inspection = adapter.inspect(robomimic_file)
    assert inspection.episode_ids == ("demo_0", "demo_2", "demo_10")

    train = adapter.inspect(robomimic_file, selection=EpisodeSelection(filter_key="train"))
    assert train.episode_ids == ("demo_0", "demo_10")

    explicit = adapter.inspect(
        robomimic_file,
        selection=EpisodeSelection(episode_ids=("demo_10", "demo_0")),
    )
    assert explicit.episode_ids == ("demo_0", "demo_10")


def test_unknown_mask_and_episode_are_rejected(robomimic_file: Path) -> None:
    adapter = RobomimicAdapter()
    with pytest.raises(SourceSchemaError, match="filter key"):
        adapter.inspect(robomimic_file, selection=EpisodeSelection(filter_key="missing"))
    with pytest.raises(SourceSchemaError, match="unknown demos"):
        adapter.inspect(
            robomimic_file,
            selection=EpisodeSelection(episode_ids=("demo_999",)),
        )

    with h5py.File(robomimic_file, "r+") as h5_file:
        h5_file["mask"].create_dataset("broken", data=np.asarray([b"demo_999"]))
    with pytest.raises(SourceSchemaError, match="unknown demos") as error:
        adapter.inspect(robomimic_file, selection=EpisodeSelection(filter_key="broken"))
    assert error.value.context == {"filter_key": "broken", "unknown": ["demo_999"]}

    with pytest.raises(ValueError, match="cannot be used together"):
        EpisodeSelection(episode_ids=("demo_0",), filter_key="train")


def test_inspection_lists_schema_and_metadata(robomimic_file: Path) -> None:
    inspection = RobomimicAdapter().inspect(robomimic_file)
    assert inspection.episode_lengths == {"demo_0": 3, "demo_2": 4, "demo_10": 2}
    assert inspection.total_frames == 9
    # Public serialization exposes only fields so the CLI JSON cannot leak the removed terminology.
    serialized = inspection.to_dict()
    assert "fields" in serialized
    assert "sequences" not in serialized
    assert inspection.field("actions").shapes == ((7,),)  # type: ignore[union-attr]
    image = inspection.field("obs/agentview_image")
    assert image is not None and image.image_candidate
    assert image.shapes == ((16, 16, 3),)
    assert inspection.metadata["data_attributes"]["env_args"]["env_name"] == "Lift"
    assert inspection.metadata["filter_keys"] == ["train", "valid"]


def test_iter_episodes_only_exposes_requested_fields(robomimic_file: Path) -> None:
    adapter = RobomimicAdapter()
    episodes = adapter.iter_episodes(
        robomimic_file,
        selection=EpisodeSelection(filter_key="valid"),
        selectors=("actions", "next_obs/robot0_eef_pos"),
    )
    episode = next(episodes)
    assert episode.episode_id == "demo_2"
    frames = iter(episode.iter_frames())
    frame = next(frames)
    assert set(frame.fields) == {"actions", "next_obs/robot0_eef_pos"}
    assert frame.fields["actions"].shape == (7,)
    assert episode.metadata["model_file"] == "model-2.xml"
    assert sum(1 for _ in frames) == episode.length - 1
    with pytest.raises(StopIteration):
        next(episodes)


def test_num_samples_mismatch_blocks_inspection(robomimic_file: Path) -> None:
    with h5py.File(robomimic_file, "r+") as h5_file:
        h5_file["data/demo_0"].attrs["num_samples"] = 99
    with pytest.raises(SourceSchemaError, match="num_samples") as error:
        RobomimicAdapter().inspect(robomimic_file)
    assert error.value.context["episode"] == "demo_0"


def test_selected_field_length_mismatch_is_not_truncated(robomimic_file: Path) -> None:
    with h5py.File(robomimic_file, "r+") as h5_file:
        del h5_file["data/demo_0/obs/robot0_eef_pos"]
        h5_file["data/demo_0/obs"].create_dataset(
            "robot0_eef_pos",
            data=np.zeros((4, 3), dtype=np.float64),
        )
    iterator = RobomimicAdapter().iter_episodes(
        robomimic_file,
        selection=EpisodeSelection(episode_ids=("demo_0",)),
        selectors=("actions", "obs/robot0_eef_pos"),
    )
    with pytest.raises(SourceSchemaError, match="does not truncate or shift") as error:
        next(iterator)
    assert error.value.context == {
        "episode": "demo_0",
        "selector": "obs/robot0_eef_pos",
        "actions_length": 3,
        "field_length": 4,
    }


def test_partial_field_is_reported_across_episodes(robomimic_file: Path) -> None:
    with h5py.File(robomimic_file, "r+") as h5_file:
        del h5_file["data/demo_2/obs/robot0_gripper_qpos"]
    inspection = RobomimicAdapter().inspect(robomimic_file)
    field = inspection.field("obs/robot0_gripper_qpos")
    assert field is not None
    assert field.missing_episodes == ("demo_2",)
    assert not field.schema_consistent
