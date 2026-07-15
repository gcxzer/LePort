"""Reusable source datasets and ConversionPlan fixtures."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from imagecodecs.numcodecs import Jpegxl, register_codecs
from leport.api import convert, create_plan
from leport.conversion.plan import ConversionPlan
from leport.sources import EpisodeSelection
from PIL import Image


def _jpeg_bytes(frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> bytes:
    """Encode deterministic RGB pixels with settings that preserve strong channel identity."""

    output = BytesIO()
    Image.fromarray(frame, mode="RGB").save(output, format="JPEG", quality=100, subsampling=0)
    return output.getvalue()


@pytest.fixture
def robomimic_file(tmp_path: Path) -> Path:
    """Create a small HDF5 fixture with standard fields, two cameras, metadata, and filter keys."""

    source = tmp_path / "robomimic.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Lift", "type": 1})
        for episode_index, length in ((0, 3), (2, 4), (10, 2)):
            episode = data.create_group(f"demo_{episode_index}")
            episode.attrs["num_samples"] = length
            episode.attrs["model_file"] = f"model-{episode_index}.xml"
            offset = float(episode_index)
            episode.create_dataset(
                "actions",
                data=np.arange(length * 7, dtype=np.float64).reshape(length, 7) + offset,
            )
            episode.create_dataset(
                "states",
                data=np.arange(length * 10, dtype=np.float64).reshape(length, 10) + offset,
            )
            episode.create_dataset("rewards", data=np.arange(length, dtype=np.float32))
            episode.create_dataset("dones", data=np.zeros(length, dtype=np.bool_))
            obs = episode.create_group("obs")
            obs.create_dataset(
                "robot0_eef_pos",
                data=np.arange(length * 3, dtype=np.float64).reshape(length, 3) + offset,
            )
            obs.create_dataset(
                "robot0_gripper_qpos",
                data=np.arange(length * 2, dtype=np.float64).reshape(length, 2) + offset,
            )
            # Pixel values encode episode and frame identity so end-to-end tests can detect misalignment.
            agentview = np.zeros((length, 16, 16, 3), dtype=np.uint8)
            wrist = np.zeros((length, 16, 16, 3), dtype=np.uint8)
            for frame_index in range(length):
                agentview[frame_index, :, :, :] = episode_index + frame_index + 10
                wrist[frame_index, :, :, :] = episode_index + frame_index + 30
            obs.create_dataset("agentview_image", data=agentview)
            obs.create_dataset("robot0_eye_in_hand_image", data=wrist)

            next_obs = episode.create_group("next_obs")
            next_obs.create_dataset(
                "robot0_eef_pos",
                data=np.arange(length * 3, dtype=np.float64).reshape(length, 3) + offset + 1,
            )

        mask = h5_file.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_0", b"demo_10"]))
        mask.create_dataset("valid", data=np.asarray([b"demo_2"]))
    return source


def _write_libero_task_file(
    path: Path,
    *,
    instruction: str,
    demo_lengths: tuple[tuple[int, int], ...],
    state_width: int,
    offset: int,
) -> None:
    """Write a compact task file matching the official LIBERO robomimic-derived structure."""

    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"problem_name": path.stem, "language_instruction": instruction}
        )
        data.attrs["bddl_file_name"] = f"{path.stem}.bddl"
        data.attrs["bddl_file_content"] = f"(define (problem {path.stem}))"
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "LIBERO",
                "env_kwargs": {"control_freq": 20, "camera_heights": 16, "camera_widths": 16},
            }
        )
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["num_demos"] = len(demo_lengths)
        data.attrs["total"] = sum(length for _, length in demo_lengths)
        for demo_index, length in demo_lengths:
            episode = data.create_group(f"demo_{demo_index}")
            episode.attrs["num_samples"] = length
            episode.attrs["init_state"] = np.arange(5, dtype=np.float64) + offset + demo_index
            episode.attrs["model_file"] = "<mujoco model='fixture'/>"
            numeric_offset = float(offset + demo_index * 100)
            episode.create_dataset(
                "actions",
                data=np.arange(length * 7, dtype=np.float64).reshape(length, 7) + numeric_offset,
            )
            episode.create_dataset(
                "states",
                data=(
                    np.arange(length * state_width, dtype=np.float64).reshape(length, state_width)
                    + numeric_offset
                ),
            )
            episode.create_dataset(
                "robot_states",
                data=np.arange(length * 9, dtype=np.float64).reshape(length, 9) + numeric_offset,
            )
            episode.create_dataset("rewards", data=np.arange(length, dtype=np.float32))
            episode.create_dataset("dones", data=np.zeros(length, dtype=np.bool_))
            observations = episode.create_group("obs")
            observations.create_dataset(
                "ee_states",
                data=np.arange(length * 6, dtype=np.float64).reshape(length, 6) + numeric_offset,
            )
            observations.create_dataset(
                "gripper_states",
                data=np.arange(length * 2, dtype=np.float64).reshape(length, 2) + numeric_offset,
            )
            observations.create_dataset(
                "joint_states",
                data=np.arange(length * 7, dtype=np.float64).reshape(length, 7) + numeric_offset,
            )

            # Row, column, and channel gradients expose flips, rotations, transposes, and channel swaps.
            rows = np.arange(16, dtype=np.uint8)[:, None]
            columns = np.arange(16, dtype=np.uint8)[None, :]
            agentview = np.empty((length, 16, 16, 3), dtype=np.uint8)
            wrist = np.empty((length, 16, 16, 3), dtype=np.uint8)
            for frame_index in range(length):
                agentview[frame_index, ..., 0] = rows + frame_index
                agentview[frame_index, ..., 1] = columns + offset
                agentview[frame_index, ..., 2] = 40 + demo_index
                wrist[frame_index, ..., 0] = 120 + demo_index
                wrist[frame_index, ..., 1] = rows + frame_index
                wrist[frame_index, ..., 2] = columns + offset
            observations.create_dataset("agentview_rgb", data=agentview)
            observations.create_dataset("eye_in_hand_rgb", data=wrist)


@pytest.fixture
def libero_directory(tmp_path: Path) -> Path:
    """Create a flat suite with lexical tasks, numeric demos, metadata, and schema drift."""

    source = tmp_path / "libero_suite"
    source.mkdir()
    _write_libero_task_file(
        source / "alpha_task_demo.hdf5",
        instruction="place the red bowl on the plate",
        demo_lengths=((0, 3), (2, 2), (10, 2)),
        state_width=10,
        offset=0,
    )
    _write_libero_task_file(
        source / "beta_task_demo.hdf5",
        instruction="close the upper drawer",
        demo_lengths=((0, 4),),
        state_width=12,
        offset=50,
    )
    (source / "README.txt").write_text("ignored non-task entry\n", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    _write_libero_task_file(
        nested / "ignored_task_demo.hdf5",
        instruction="this nested file must not be discovered",
        demo_lengths=((0, 1),),
        state_width=10,
        offset=90,
    )
    return source


@pytest.fixture
def libero_file(libero_directory: Path) -> Path:
    """Return one official-structure task file for single-file behavior tests."""

    return libero_directory / "alpha_task_demo.hdf5"


@pytest.fixture
def malformed_libero_files(tmp_path: Path) -> dict[str, Path]:
    """Create independent failures for metadata, catalog counts, and field alignment."""

    source = tmp_path / "malformed_libero"
    source.mkdir()
    cases: dict[str, Path] = {}
    for case in (
        "missing_metadata",
        "invalid_problem_json",
        "missing_bddl",
        "noncanonical_demo",
        "num_demos_mismatch",
        "num_samples_mismatch",
        "field_length_mismatch",
        "missing_actions",
    ):
        path = source / f"{case}_demo.hdf5"
        _write_libero_task_file(
            path,
            instruction="valid instruction before mutation",
            demo_lengths=((0, 2),),
            state_width=10,
            offset=0,
        )
        with h5py.File(path, "a") as h5_file:
            data = h5_file["data"]
            episode = data["demo_0"]
            if case == "missing_metadata":
                del data.attrs["problem_info"]
            elif case == "invalid_problem_json":
                data.attrs["problem_info"] = "{invalid"
            elif case == "missing_bddl":
                del data.attrs["bddl_file_name"]
                del data.attrs["bddl_file_content"]
            elif case == "noncanonical_demo":
                data.move("demo_0", "demo_00")
            elif case == "num_demos_mismatch":
                data.attrs["num_demos"] = 2
            elif case == "num_samples_mismatch":
                episode.attrs["num_samples"] = 3
            elif case == "field_length_mismatch":
                del episode["obs/ee_states"]
                episode["obs"].create_dataset("ee_states", data=np.zeros((1, 6), dtype=np.float64))
            elif case == "missing_actions":
                del episode["actions"]
        cases[case] = path
    return cases


@pytest.fixture
def aloha_directory(tmp_path: Path) -> Path:
    """Create numeric ALOHA episodes spanning raw, padded JPEG, and variable JPEG storage."""

    source = tmp_path / "aloha"
    source.mkdir()
    for episode_index, length, storage in (
        (0, 3, "raw"),
        (2, 2, "fixed-jpeg"),
        (10, 2, "variable-jpeg"),
    ):
        episode_path = source / f"episode_{episode_index}.hdf5"
        with h5py.File(episode_path, "w") as h5_file:
            h5_file.attrs["sim"] = np.bool_(episode_index == 0)
            h5_file.attrs["compress"] = np.bool_(storage != "raw")
            h5_file.attrs["instruction"] = f"move object in episode {episode_index}"
            h5_file.attrs["camera_names"] = np.asarray([b"cam_high", b"cam_wrist"])
            offset = float(episode_index * 100)
            h5_file.create_dataset(
                "action",
                data=np.arange(length * 7, dtype=np.float64).reshape(length, 7) + offset,
            )
            observations = h5_file.create_group("observations")
            observations.create_dataset(
                "qpos",
                data=np.arange(length * 6, dtype=np.float64).reshape(length, 6) + offset,
            )
            observations.create_dataset(
                "qvel",
                data=np.arange(length * 6, dtype=np.float32).reshape(length, 6) + offset + 0.5,
            )
            # Leaving effort out of one episode exercises optional-field coverage without weakening
            # the required qpos contract.
            if episode_index != 2:
                observations.create_dataset(
                    "effort",
                    data=np.arange(length * 6, dtype=np.float32).reshape(length, 6) + offset + 1.5,
                )
            images = observations.create_group("images")
            high_frames = np.empty((length, 16, 16, 3), dtype=np.uint8)
            wrist_frames = np.empty((length, 16, 16, 3), dtype=np.uint8)
            for frame_index in range(length):
                high_frames[frame_index, ...] = (240 - frame_index * 8, 24 + frame_index, 8)
                wrist_frames[frame_index, ...] = (8, 32 + frame_index, 240 - frame_index * 8)

            if storage == "raw":
                images.create_dataset("cam_high", data=high_frames)
                images.create_dataset("cam_wrist", data=wrist_frames)
            else:
                encoded_by_camera = [
                    [_jpeg_bytes(frame) for frame in high_frames],
                    [_jpeg_bytes(frame) for frame in wrist_frames],
                ]
                if storage == "fixed-jpeg":
                    for camera_name, encoded_frames in zip(
                        ("cam_high", "cam_wrist"), encoded_by_camera, strict=True
                    ):
                        encoded_width = max(len(payload) for payload in encoded_frames) + 17
                        padded = np.zeros((length, encoded_width), dtype=np.uint8)
                        for frame_index, payload in enumerate(encoded_frames):
                            padded[frame_index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
                        images.create_dataset(camera_name, data=padded)
                else:
                    variable_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                    for camera_name, encoded_frames in zip(
                        ("cam_high", "cam_wrist"), encoded_by_camera, strict=True
                    ):
                        dataset = images.create_dataset(camera_name, shape=(length,), dtype=variable_dtype)
                        for frame_index, payload in enumerate(encoded_frames):
                            dataset[frame_index] = np.frombuffer(payload, dtype=np.uint8)
                h5_file.create_dataset(
                    "compress_len",
                    data=np.asarray(
                        [[len(payload) for payload in camera] for camera in encoded_by_camera],
                        dtype=np.int32,
                    ),
                )

    # These entries prove directory discovery is flat, filename-driven, and tolerant of unrelated
    # files while structural recognition remains HDF5-based.
    (source / "notes.txt").write_text("fixture notes", encoding="utf-8")
    with h5py.File(source / "recording.hdf5", "w") as h5_file:
        h5_file.create_dataset("unrelated", data=np.zeros(1, dtype=np.float32))
    nested = source / "logs"
    nested.mkdir()
    (nested / "episode_99.hdf5").write_text("not scanned recursively", encoding="utf-8")
    return source


@pytest.fixture
def malformed_aloha_directory(tmp_path: Path) -> Path:
    """Create isolated structural, length, image-layout, and JPEG failures for negative tests."""

    source = tmp_path / "malformed-aloha"
    source.mkdir()
    with h5py.File(source / "episode_20.hdf5", "w") as h5_file:
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((1, 6), dtype=np.float32))
    with h5py.File(source / "episode_21.hdf5", "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((1, 7), dtype=np.float32))
        h5_file.create_group("observations")
    with h5py.File(source / "episode_22.hdf5", "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((0, 7), dtype=np.float32))
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((0, 6), dtype=np.float32))
    with h5py.File(source / "episode_23.hdf5", "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((2, 7), dtype=np.float32))
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((3, 6), dtype=np.float32))
    with h5py.File(source / "episode_24.hdf5", "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((1, 7), dtype=np.float32))
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((1, 6), dtype=np.float32))
        images = observations.create_group("images")
        images.create_dataset("cam_high", data=np.asarray([[1, 2, 3, 4]], dtype=np.uint8))
    with h5py.File(source / "episode_25.hdf5", "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((1, 7), dtype=np.float32))
        observations = h5_file.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((1, 6), dtype=np.float32))
        images = observations.create_group("images")
        images.create_dataset("cam_high", data=np.zeros((1, 8, 8, 2), dtype=np.uint8))
    (source / "episode_26.hdf5").write_text("not an HDF5 file", encoding="utf-8")
    return source


@pytest.fixture
def numeric_plan(robomimic_file: Path, tmp_path: Path) -> ConversionPlan:
    """Build a fast end-to-end plan without visual features."""

    return create_plan(
        robomimic_file,
        target_root=tmp_path / "lerobot-numeric",
        repo_id="tests/robomimic-numeric",
        fps=20,
        task="lift the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        use_videos=False,
        adapter="robomimic",
    )


@pytest.fixture
def compatible_lerobot_sources(robomimic_file: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Create two minimal LeRobot inputs with matching schemas but different lengths and tasks."""

    first_target = tmp_path / "merge-source-first"
    second_target = tmp_path / "merge-source-second"
    first_plan = create_plan(
        robomimic_file,
        target_root=first_target,
        repo_id="tests/merge-source-first",
        fps=20,
        task="lift the cube",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        use_videos=False,
        adapter="robomimic",
        selection=EpisodeSelection(episode_ids=("demo_0", "demo_10")),
    )
    second_plan = create_plan(
        robomimic_file,
        target_root=second_target,
        repo_id="tests/merge-source-second",
        fps=20,
        task="pick up the can",
        action_source="actions",
        action_dtype="float32",
        state_sources=("obs/robot0_eef_pos", "obs/robot0_gripper_qpos"),
        state_dtype="float32",
        use_videos=False,
        adapter="robomimic",
        selection=EpisodeSelection(episode_ids=("demo_2",)),
    )
    convert(first_plan)
    convert(second_plan)
    return first_target, second_target


def _maniskill_metadata(episodes: tuple[tuple[int, int], ...], *, obs_mode: str) -> dict[str, object]:
    """Build plain companion metadata shared by valid and malformed ManiSkill fixtures."""

    return {
        "env_info": {
            "env_id": "PickCube-v1",
            "max_episode_steps": 100,
            "env_kwargs": {"obs_mode": obs_mode, "sim_backend": "physx_cpu"},
        },
        "episodes": [
            {
                "episode_id": episode_id,
                "reset_kwargs": {"seed": episode_id + 100},
                "control_mode": "pd_joint_delta_pos",
                "elapsed_steps": length,
                "info": {"success": True},
                "instruction": "pick up the cube",
            }
            for episode_id, length in episodes
        ],
        "source_type": "motionplanning",
        "source_desc": "Deterministic ManiSkill test trajectories",
    }


@pytest.fixture
def maniskill_file(tmp_path: Path) -> Path:
    """Create paired replayed trajectories with nested T+1 observations and exact endpoint markers."""

    source = tmp_path / "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    episode_specs = ((10, 2), (0, 3), (2, 4))
    with h5py.File(source, "w") as h5_file:
        for episode_id, length in episode_specs:
            trajectory = h5_file.create_group(f"traj_{episode_id}")
            offset = float(episode_id * 100)
            trajectory.create_dataset(
                "actions",
                data=np.arange(length * 7, dtype=np.float32).reshape(length, 7) + offset,
            )
            trajectory.create_dataset(
                "terminated",
                data=np.asarray([False] * (length - 1) + [True], dtype=np.bool_),
            )
            trajectory.create_dataset("truncated", data=np.zeros(length, dtype=np.bool_))
            trajectory.create_dataset(
                "rewards",
                data=np.arange(length, dtype=np.float32) + offset,
            )
            trajectory.create_dataset(
                "success",
                data=np.asarray([False] * (length - 1) + [True], dtype=np.bool_),
            )
            trajectory.create_dataset("fail", data=np.zeros(length, dtype=np.bool_))

            observations = trajectory.create_group("obs")
            agent = observations.create_group("agent")
            state_offset = float(episode_id * 1000)
            agent.create_dataset(
                "qpos",
                data=np.arange((length + 1) * 6, dtype=np.float32).reshape(length + 1, 6) + state_offset,
            )
            agent.create_dataset(
                "qvel",
                data=np.arange((length + 1) * 6, dtype=np.float64).reshape(length + 1, 6)
                + state_offset
                + 0.5,
            )
            sensor_data = observations.create_group("sensor_data")
            base_camera = sensor_data.create_group("base_camera")
            wrist_camera = sensor_data.create_group("wrist_camera")
            base_rgb = np.zeros((length + 1, 16, 20, 3), dtype=np.uint8)
            wrist_rgb = np.zeros((length + 1, 16, 20, 3), dtype=np.uint8)
            depth = np.zeros((length + 1, 16, 20, 1), dtype=np.uint16)
            for observation_index in range(length + 1):
                # Channel values encode episode and endpoint identity so alignment and RGB-order
                # assertions detect an off-by-one read or accidental channel conversion.
                base_rgb[observation_index, ...] = (
                    10 + episode_id + observation_index,
                    30 + episode_id,
                    50 + observation_index,
                )
                wrist_rgb[observation_index, ...] = (
                    100 + episode_id + observation_index,
                    120 + episode_id,
                    140 + observation_index,
                )
                depth[observation_index, ...] = episode_id * 100 + observation_index
            base_camera.create_dataset("rgb", data=base_rgb)
            base_camera.create_dataset("depth", data=depth)
            wrist_camera.create_dataset("rgb", data=wrist_rgb)

            environment_states = trajectory.create_group("env_states")
            actors = environment_states.create_group("actors")
            articulations = environment_states.create_group("articulations")
            actors.create_dataset(
                "cube",
                data=np.arange((length + 1) * 3, dtype=np.float32).reshape(length + 1, 3) + state_offset,
            )
            articulations.create_dataset(
                "robot",
                data=np.arange((length + 1) * 9, dtype=np.float32).reshape(length + 1, 9) + state_offset,
            )

    source.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(episode_specs, obs_mode="rgb+state"), indent=2),
        encoding="utf-8",
    )
    return source


@pytest.fixture
def raw_maniskill_file(tmp_path: Path) -> Path:
    """Create a compressed-style raw trajectory with environment states but no observations."""

    source = tmp_path / "trajectory.none.pd_joint_pos.physx_cpu.h5"
    with h5py.File(source, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.arange(14, dtype=np.float32).reshape(2, 7))
        trajectory.create_dataset("terminated", data=np.asarray([False, True], dtype=np.bool_))
        trajectory.create_dataset("truncated", data=np.zeros(2, dtype=np.bool_))
        actors = trajectory.create_group("env_states").create_group("actors")
        actors.create_dataset("cube", data=np.arange(9, dtype=np.float32).reshape(3, 3))
    source.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="none"), indent=2),
        encoding="utf-8",
    )
    return source


@pytest.fixture
def malformed_maniskill_pairs(tmp_path: Path) -> dict[str, Path]:
    """Create isolated pair failures so diagnostics cannot mask one another."""

    sources: dict[str, Path] = {}

    missing_json = tmp_path / "missing-json.h5"
    with h5py.File(missing_json, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    sources["missing_json"] = missing_json

    invalid_json = tmp_path / "invalid-json.h5"
    with h5py.File(invalid_json, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    invalid_json.with_suffix(".json").write_text("{not valid json", encoding="utf-8")
    sources["invalid_json"] = invalid_json

    duplicate_json = tmp_path / "duplicate-json-id.h5"
    with h5py.File(duplicate_json, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    duplicate_metadata = _maniskill_metadata(((0, 2), (0, 2)), obs_mode="none")
    duplicate_json.with_suffix(".json").write_text(json.dumps(duplicate_metadata), encoding="utf-8")
    sources["duplicate_json_id"] = duplicate_json

    json_without_hdf5 = tmp_path / "json-without-hdf5.h5"
    with h5py.File(json_without_hdf5, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    json_without_hdf5.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2), (2, 2)), obs_mode="none")),
        encoding="utf-8",
    )
    sources["json_without_hdf5"] = json_without_hdf5

    hdf5_without_json = tmp_path / "hdf5-without-json-episode.h5"
    with h5py.File(hdf5_without_json, "w") as h5_file:
        for episode_id in (0, 2):
            h5_file.create_group(f"traj_{episode_id}").create_dataset(
                "actions", data=np.zeros((2, 7), dtype=np.float32)
            )
    hdf5_without_json.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="none")),
        encoding="utf-8",
    )
    sources["hdf5_without_json_episode"] = hdf5_without_json

    declared_length = tmp_path / "declared-length.h5"
    with h5py.File(declared_length, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
    declared_length.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 3),), obs_mode="none")),
        encoding="utf-8",
    )
    sources["declared_length"] = declared_length

    missing_actions = tmp_path / "missing-actions.h5"
    with h5py.File(missing_actions, "w") as h5_file:
        h5_file.create_group("traj_0").create_dataset("terminated", data=np.zeros(2, dtype=np.bool_))
    missing_actions.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="none")),
        encoding="utf-8",
    )
    sources["missing_actions"] = missing_actions

    schema_drift = tmp_path / "schema-drift.h5"
    with h5py.File(schema_drift, "w") as h5_file:
        for episode_id, width in ((0, 6), (2, 7)):
            trajectory = h5_file.create_group(f"traj_{episode_id}")
            trajectory.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
            trajectory.create_group("obs").create_dataset(
                "state", data=np.zeros((3, width), dtype=np.float32)
            )
    schema_drift.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2), (2, 2)), obs_mode="state")),
        encoding="utf-8",
    )
    sources["schema_drift"] = schema_drift

    bad_transition = tmp_path / "bad-transition-length.h5"
    with h5py.File(bad_transition, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        trajectory.create_dataset("rewards", data=np.zeros(3, dtype=np.float32))
    bad_transition.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="none")),
        encoding="utf-8",
    )
    sources["bad_transition_length"] = bad_transition

    bad_observation = tmp_path / "bad-observation-length.h5"
    with h5py.File(bad_observation, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        trajectory.create_group("obs").create_dataset("state", data=np.zeros((2, 6), dtype=np.float32))
    bad_observation.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="state")),
        encoding="utf-8",
    )
    sources["bad_observation_length"] = bad_observation

    bad_environment_state = tmp_path / "bad-environment-state-length.h5"
    with h5py.File(bad_environment_state, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        trajectory.create_group("env_states").create_dataset("state", data=np.zeros((2, 6), dtype=np.float32))
    bad_environment_state.with_suffix(".json").write_text(
        json.dumps(_maniskill_metadata(((0, 2),), obs_mode="none")),
        encoding="utf-8",
    )
    sources["bad_environment_state_length"] = bad_environment_state
    return sources


def _write_umi_store(store: object) -> None:
    """Write a compact processed UMI replay buffer with official field and codec conventions."""

    register_codecs()
    root = zarr.group(store=store, overwrite=True)
    metadata = root.create_group("meta")
    metadata.array(
        "episode_ends",
        np.asarray([3, 7, 9], dtype=np.int64),
        chunks=(3,),
        compressor=None,
    )
    data = root.create_group("data")
    total_frames = 9
    data.array(
        "robot0_eef_pos",
        np.arange(total_frames * 3, dtype=np.float32).reshape(total_frames, 3),
        chunks=(4, 3),
    )
    data.array(
        "robot0_eef_rot_axis_angle",
        np.arange(total_frames * 3, dtype=np.float32).reshape(total_frames, 3) + 100,
        chunks=(4, 3),
    )
    data.array(
        "robot0_gripper_width",
        np.arange(total_frames, dtype=np.float32).reshape(total_frames, 1) / 100,
        chunks=(4, 1),
    )
    data.array(
        "robot0_demo_start_pose",
        np.zeros((total_frames, 6), dtype=np.float64),
        chunks=(4, 6),
    )
    data.array(
        "robot0_demo_end_pose",
        np.ones((total_frames, 6), dtype=np.float64),
        chunks=(4, 6),
    )
    data.array(
        "robot1_eef_pos",
        np.arange(total_frames * 3, dtype=np.float32).reshape(total_frames, 3) + 200,
        chunks=(4, 3),
    )
    data.array(
        "robot1_eef_rot_axis_angle",
        np.arange(total_frames * 3, dtype=np.float32).reshape(total_frames, 3) + 300,
        chunks=(4, 3),
    )
    data.array(
        "robot1_gripper_width",
        np.arange(total_frames, dtype=np.float32).reshape(total_frames, 1) / 50,
        chunks=(4, 1),
    )
    # An explicit action remains an ordinary selectable field; conversion tests intentionally map
    # the robot0 components instead to prove the adapter does not prefer or synthesize semantics.
    data.array(
        "action",
        np.full((total_frames, 7), -5, dtype=np.float32),
        chunks=(4, 7),
    )

    camera0 = np.empty((total_frames, 8, 10, 3), dtype=np.uint8)
    camera1 = np.empty((total_frames, 8, 10, 3), dtype=np.uint8)
    for frame_index in range(total_frames):
        camera0[frame_index, ...] = (10 + frame_index, 30, 50 + frame_index)
        camera1[frame_index, ...] = (100 + frame_index, 120, 140 + frame_index)
    data.array(
        "camera0_rgb",
        camera0,
        chunks=(1, 8, 10, 3),
        compressor=Jpegxl(level=99, numthreads=1),
    )
    data.array(
        "camera1_rgb",
        camera1,
        chunks=(1, 8, 10, 3),
        compressor=None,
    )


@pytest.fixture
def umi_directory(tmp_path: Path) -> Path:
    """Create a processed UMI directory store with three cumulative episode slices."""

    source = tmp_path / "dataset.zarr"
    store = zarr.DirectoryStore(str(source))
    try:
        _write_umi_store(store)
    finally:
        store.close()
    return source


@pytest.fixture
def umi_zip_file(tmp_path: Path) -> Path:
    """Create the ZipStore representation emitted by the official UMI processing pipeline."""

    source = tmp_path / "dataset.zarr.zip"
    store = zarr.ZipStore(str(source), mode="w")
    try:
        _write_umi_store(store)
    finally:
        store.close()
    return source


@pytest.fixture
def malformed_umi_sources(umi_directory: Path, tmp_path: Path) -> dict[str, Path]:
    """Create isolated UMI failures for signature, boundary, layout, and alignment checks."""

    import shutil

    sources: dict[str, Path] = {}
    for case in ("missing_robot", "missing_camera", "bad_boundaries", "length_mismatch", "nested"):
        path = tmp_path / f"umi-{case}.zarr"
        shutil.copytree(umi_directory, path)
        root = zarr.open_group(str(path), mode="a")
        if case == "missing_robot":
            del root["data/robot0_eef_pos"]
        elif case == "missing_camera":
            del root["data/camera0_rgb"]
            del root["data/camera1_rgb"]
        elif case == "bad_boundaries":
            root["meta/episode_ends"][:] = np.asarray([3, 3, 9], dtype=np.int64)
        elif case == "length_mismatch":
            root["data/robot0_demo_end_pose"].resize((8, 6))
        else:
            root["data"].create_group("nested")
        sources[case] = path

    misleading = tmp_path / "not-really.zarr.zip"
    misleading.write_text("not a zip store", encoding="utf-8")
    sources["misleading"] = misleading
    return sources
