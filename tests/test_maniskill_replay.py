"""Test optional ManiSkill replay orchestration without loading the simulator in the test process."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from leport import ManiSkillReplayOptions, ManiSkillReplayResult, replay_maniskill
from leport.cli import main
from leport.errors import OptionalDependencyError, ReplayError
from leport.sources.maniskill import ManiSkillAdapter


def _write_valid_replay_pair(path: Path) -> None:
    """Create one small materialized pair that the real ManiSkill adapter can validate."""

    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.arange(14, dtype=np.float32).reshape(2, 7))
        trajectory.create_dataset("terminated", data=np.asarray([False, True], dtype=np.bool_))
        trajectory.create_dataset("truncated", data=np.zeros(2, dtype=np.bool_))
        observations = trajectory.create_group("obs")
        observations.create_group("agent").create_dataset(
            "qpos", data=np.arange(18, dtype=np.float32).reshape(3, 6)
        )
        rgb = np.zeros((3, 8, 10, 3), dtype=np.uint8)
        rgb[0, ...] = (10, 30, 50)
        rgb[1, ...] = (11, 30, 51)
        rgb[2, ...] = (12, 30, 52)
        observations.create_group("sensor_data").create_group("base_camera").create_dataset("rgb", data=rgb)
        trajectory.create_group("env_states").create_group("actors").create_dataset(
            "cube", data=np.arange(9, dtype=np.float32).reshape(3, 3)
        )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "max_episode_steps": 100,
                    "env_kwargs": {
                        "obs_mode": "rgb",
                        "control_mode": "pd_joint_delta_pos",
                        "sim_backend": "physx_cpu",
                    },
                },
                "episodes": [
                    {
                        "episode_id": 0,
                        "episode_seed": 0,
                        "reset_kwargs": {"seed": 0},
                        "control_mode": "pd_joint_delta_pos",
                        "elapsed_steps": 2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"obs_mode": ""}, "non-empty"),
        ({"count": 0}, "count must be positive"),
        ({"count": -1}, "count must be positive"),
        ({"num_envs": 0}, "num_envs must be positive"),
    ],
)
def test_replay_options_reject_invalid_values_before_source_or_runtime_access(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ReplayError, match=message):
        replay_maniskill(tmp_path / "missing.h5", **kwargs)


def test_replay_rejects_missing_or_malformed_input_pairs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5"
    with pytest.raises(ReplayError, match=r"regular \.h5"):
        replay_maniskill(missing)

    no_metadata = tmp_path / "trajectory.h5"
    no_metadata.write_bytes(b"not opened before metadata validation")
    with pytest.raises(ReplayError, match="same-basename JSON"):
        replay_maniskill(no_metadata)

    no_metadata.with_suffix(".json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(ReplayError, match="parse ManiSkill replay metadata"):
        replay_maniskill(no_metadata)


def test_missing_replay_runtime_is_precise_and_materialized_adapter_remains_available(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("leport.maniskill_replay.importlib.util.find_spec", lambda name: None)
    inspection = ManiSkillAdapter().inspect(raw_maniskill_file)
    assert inspection.episode_ids == ("traj_0",)
    with pytest.raises(OptionalDependencyError, match="uv sync --extra maniskill-replay") as caught:
        replay_maniskill(raw_maniskill_file)
    assert caught.value.context == {
        "feature": "maniskill-replay",
        "extra": "maniskill-replay",
        "dependency": "mani_skill",
    }


def test_successful_replay_forwards_exact_options_preserves_input_and_validates_output(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = raw_maniskill_file.parent / "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    original_bytes = raw_maniskill_file.read_bytes()
    recorded: dict[str, Any] = {}
    monkeypatch.setattr("leport.maniskill_replay.importlib.util.find_spec", lambda name: object())

    def run_replay(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        _write_valid_replay_pair(output)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "initial log\nReplayed 1 episodes, 1/1=100.00% demos saved\n"
                "Destroyed VkDevice after replay.\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("leport.maniskill_replay.subprocess.run", run_replay)
    result = replay_maniskill(
        raw_maniskill_file,
        obs_mode="rgb",
        use_env_states=True,
        target_control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        count=1,
        num_envs=2,
        record_rewards=True,
        reward_mode="dense",
        allow_failure=True,
    )

    assert recorded["command"] == [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(raw_maniskill_file.resolve()),
        "--save-traj",
        "--obs-mode",
        "rgb",
        "--num-envs",
        "2",
        "--use-env-states",
        "--target-control-mode",
        "pd_joint_delta_pos",
        "--sim-backend",
        "physx_cpu",
        "--count",
        "1",
        "--record-rewards",
        "--reward-mode",
        "dense",
        "--allow-failure",
    ]
    assert recorded["kwargs"] == {"check": False, "capture_output": True, "text": True}
    assert raw_maniskill_file.read_bytes() == original_bytes
    assert result.output_hdf5 == output.resolve()
    assert result.output_json == output.with_suffix(".json").resolve()
    assert result.runtime_summary == "Replayed 1 episodes, 1/1=100.00% demos saved"
    assert result.options.to_dict() == {
        "obs_mode": "rgb",
        "use_env_states": True,
        "target_control_mode": "pd_joint_delta_pos",
        "sim_backend": "physx_cpu",
        "count": 1,
        "num_envs": 2,
        "record_rewards": True,
        "reward_mode": "dense",
        "allow_failure": True,
    }
    assert (
        ManiSkillAdapter()
        .inspect(result.output_hdf5)
        .field("obs/sensor_data/base_camera/rgb")
        .image_candidate
    )


def test_replay_refuses_a_predictable_existing_output_before_runtime_start(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = raw_maniskill_file.parent / "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    output.write_bytes(b"existing")
    output.with_suffix(".json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "leport.maniskill_replay.subprocess.run",
        lambda *args, **kwargs: pytest.fail("replay process must not start"),
    )
    with pytest.raises(ReplayError, match="output already exists") as caught:
        replay_maniskill(raw_maniskill_file)
    assert caught.value.context["output_hdf5"] == str(output)


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_replay_requires_exactly_one_new_output_pair(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_count: int,
) -> None:
    monkeypatch.setattr("leport.maniskill_replay.importlib.util.find_spec", lambda name: object())

    def run_replay(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        for index in range(candidate_count):
            _write_valid_replay_pair(raw_maniskill_file.parent / f"unexpected-{index}.h5")
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    monkeypatch.setattr("leport.maniskill_replay.subprocess.run", run_replay)
    with pytest.raises(ReplayError, match="exactly one new trajectory pair") as caught:
        replay_maniskill(raw_maniskill_file)
    assert len(caught.value.context["candidates"]) == candidate_count


def test_replay_reports_process_exit_and_start_failures(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("leport.maniskill_replay.importlib.util.find_spec", lambda name: object())
    long_diagnostics = "failure:" + "x" * 2_500
    monkeypatch.setattr(
        "leport.maniskill_replay.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7, stdout="", stderr=long_diagnostics),
    )
    with pytest.raises(ReplayError, match="process failed") as exited:
        replay_maniskill(raw_maniskill_file)
    assert exited.value.context["returncode"] == 7
    assert len(exited.value.context["diagnostics"]) == 2_000

    monkeypatch.setattr(
        "leport.maniskill_replay.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn denied")),
    )
    with pytest.raises(ReplayError, match="Could not start") as not_started:
        replay_maniskill(raw_maniskill_file)
    assert not_started.value.context["reason"] == "spawn denied"


def test_replay_rejects_an_invalid_generated_pair(
    raw_maniskill_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = raw_maniskill_file.parent / "trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5"
    monkeypatch.setattr("leport.maniskill_replay.importlib.util.find_spec", lambda name: object())

    def run_replay(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        with h5py.File(output, "w") as h5_file:
            h5_file.create_group("not-a-trajectory")
        output.with_suffix(".json").write_text(json.dumps({"env_info": {}, "episodes": []}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    monkeypatch.setattr("leport.maniskill_replay.subprocess.run", run_replay)
    with pytest.raises(ReplayError, match="not a valid materialized trajectory pair"):
        replay_maniskill(raw_maniskill_file)


def test_replay_cli_forwards_options_and_emits_one_json_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "trajectory.h5"
    output = tmp_path / "trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    captured: dict[str, Any] = {}

    def replay(source_value: Path, **kwargs: Any) -> ManiSkillReplayResult:
        captured["source"] = source_value
        captured["kwargs"] = kwargs
        return ManiSkillReplayResult(
            source=source.resolve(),
            output_hdf5=output.resolve(),
            output_json=output.with_suffix(".json").resolve(),
            options=ManiSkillReplayOptions(
                obs_mode="rgb",
                use_env_states=True,
                target_control_mode="pd_joint_delta_pos",
                sim_backend="physx_cpu",
                count=2,
                num_envs=3,
                record_rewards=True,
                reward_mode="dense",
                allow_failure=True,
            ),
            runtime_summary="Replayed 2 episodes",
        )

    monkeypatch.setattr("leport.cli.replay_maniskill", replay)
    assert (
        main(
            [
                "replay-maniskill",
                str(source),
                "--obs-mode",
                "rgb",
                "--use-env-states",
                "--target-control-mode",
                "pd_joint_delta_pos",
                "--sim-backend",
                "physx_cpu",
                "--count",
                "2",
                "--num-envs",
                "3",
                "--record-rewards",
                "--reward-mode",
                "dense",
                "--allow-failure",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert captured == {
        "source": source,
        "kwargs": {
            "obs_mode": "rgb",
            "use_env_states": True,
            "target_control_mode": "pd_joint_delta_pos",
            "sim_backend": "physx_cpu",
            "count": 2,
            "num_envs": 3,
            "record_rewards": True,
            "reward_mode": "dense",
            "allow_failure": True,
        },
    }
    assert payload["output_hdf5"] == str(output.resolve())
    assert payload["runtime_summary"] == "Replayed 2 episodes"


def test_replay_cli_maps_validation_errors_to_exit_code_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["replay-maniskill", str(tmp_path / "missing.h5"), "--count", "0"]) == 2
    assert "replay_error" in capsys.readouterr().err


def test_every_replay_requirement_and_scenario_has_automated_evidence() -> None:
    success = "test_successful_replay_forwards_exact_options_preserves_input_and_validates_output"
    isolation = "test_missing_replay_runtime_is_precise_and_materialized_adapter_remains_available"
    inputs = "test_replay_rejects_missing_or_malformed_input_pairs"
    overwrite = "test_replay_refuses_a_predictable_existing_output_before_runtime_start"
    discovery = "test_replay_requires_exactly_one_new_output_pair"
    failures = "test_replay_reports_process_exit_and_start_failures"
    options = "test_replay_options_reject_invalid_values_before_source_or_runtime_access"
    cli = "test_replay_cli_forwards_options_and_emits_one_json_document"
    requirement_tests = {
        "Replay a paired ManiSkill trajectory explicitly": success,
        "Keep the replay runtime optional and isolated": isolation,
        "Preserve replay inputs and identify one generated output pair": discovery,
        "Expose explicit replay options and structured results": cli,
    }
    scenario_tests = {
        "Raw trajectory is replayed with RGB observations": success,
        "Ordinary adapter workflow is used": isolation,
        "Input pair is incomplete": inputs,
        "Replay runtime is installed": success,
        "Replay runtime is missing": isolation,
        "Another source adapter is used": isolation,
        "Replay creates one new pair": success,
        "Predictable output already exists": overwrite,
        "Runtime produces no unique pair": discovery,
        "Runtime process fails": failures,
        "Optional arguments are provided": success,
        "Count is invalid": options,
        "CLI emits JSON": cli,
    }
    specification = (
        Path(__file__).parents[1] / "openspec/specs/maniskill-trajectory-replay/spec.md"
    ).read_text(encoding="utf-8")
    assert set(requirement_tests) == set(re.findall(r"^### Requirement: (.+)$", specification, re.MULTILINE))
    assert set(scenario_tests) == set(re.findall(r"^#### Scenario: (.+)$", specification, re.MULTILINE))
    for test_name in (*requirement_tests.values(), *scenario_tests.values()):
        assert callable(globals().get(test_name)), test_name
