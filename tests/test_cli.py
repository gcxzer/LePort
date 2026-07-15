"""Test successful CLI workflows, read-only behavior, and stable failure exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from leport.cli import main
from leport.errors import AdapterAmbiguousError, OptionalDependencyError


def test_inspect_is_read_only_and_json_serializable(
    robomimic_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "must-not-exist"
    assert main(["inspect", str(robomimic_file), "--adapter", "robomimic", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episode_ids"] == ["demo_0", "demo_2", "demo_10"]
    assert not target.exists()


def test_plan_check_and_numeric_cli_conversion(
    robomimic_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.yaml"
    target = tmp_path / "cli-target"
    plan_args = [
        "plan",
        "--source",
        str(robomimic_file),
        "--output",
        str(plan_path),
        "--episode",
        "demo_0,demo_2,demo_10",
        "--target",
        str(target),
        "--repo-id",
        "tests/cli",
        "--fps",
        "20",
        "--task",
        "lift the cube",
        "--action",
        "actions",
        "--action-dtype",
        "float32",
        "--state",
        "obs/robot0_eef_pos",
        "--state",
        "obs/robot0_gripper_qpos",
        "--state-dtype",
        "float32",
        "--no-videos",
        "--json",
    ]
    assert main(plan_args) == 0
    assert plan_path.is_file()
    generated_plan = json.loads(capsys.readouterr().out)
    assert generated_plan["selection"]["episode_ids"] == ["demo_0", "demo_2", "demo_10"]
    assert main(["plan", "--check", str(plan_path), "--json"]) == 0
    json.loads(capsys.readouterr().out)

    assert main(["convert", "--config", str(plan_path), "--json"]) == 0
    conversion = json.loads(capsys.readouterr().out)
    assert conversion["total_episodes"] == 3
    assert conversion["total_frames"] == 9
    assert target.is_dir()

    assert main(["validate", str(target), "--config", str(plan_path), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["episode_lengths"] == [3, 4, 2]


@pytest.mark.parametrize(
    ("episode_args", "expected_reason"),
    [
        (["--episode", "demo_0", "--episode", "demo_2"], "may appear only once"),
        (["--episode", "demo_0,,demo_2"], "cannot contain an empty episode ID"),
    ],
)
def test_cli_rejects_noncanonical_episode_lists(
    robomimic_file: Path,
    capsys: pytest.CaptureFixture[str],
    episode_args: list[str],
    expected_reason: str,
) -> None:
    """The CLI accepts one comma-separated list and rejects ambiguous repeats or empty items."""

    assert main(["inspect", str(robomimic_file), *episode_args]) == 2
    assert expected_reason in capsys.readouterr().err


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (AdapterAmbiguousError("ambiguous"), "adapter_ambiguous"),
        (OptionalDependencyError("missing"), "optional_dependency_missing"),
    ],
)
def test_cli_maps_expected_failures_to_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    code: str,
) -> None:
    monkeypatch.setattr("leport.cli.inspect", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    assert main(["inspect", str(tmp_path / "source")]) == 2
    assert code in capsys.readouterr().err


def test_cli_reports_no_matching_adapter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "unknown.txt"
    source.write_text("not hdf5", encoding="utf-8")
    assert main(["inspect", str(source)]) == 2
    assert "adapter_not_found" in capsys.readouterr().err


def test_merge_cli_emits_structured_json(
    compatible_lerobot_sources: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI reports positional source order and complete validation as machine-readable JSON."""

    first, second = compatible_lerobot_sources
    target = tmp_path / "cli-merged"
    assert (
        main(
            [
                "merge",
                str(first),
                str(second),
                "--target",
                str(target),
                "--repo-id",
                "tests/cli-merged",
                "--no-concatenate-videos",
                "--no-concatenate-data",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"] == [str(first.resolve()), str(second.resolve())]
    assert payload["target"] == str(target.resolve())
    assert payload["total_episodes"] == 3
    assert payload["total_frames"] == 9
    assert payload["validation"]["episode_lengths"] == [3, 2, 4]


def test_merge_cli_maps_too_few_inputs_to_exit_code_two(
    compatible_lerobot_sources: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After argparse accepts one source, the public API returns a stable merge_error."""

    first, _ = compatible_lerobot_sources
    assert (
        main(
            [
                "merge",
                str(first),
                "--target",
                str(tmp_path / "invalid-cli-merged"),
                "--repo-id",
                "tests/invalid-cli-merged",
            ]
        )
        == 2
    )
    assert "merge_error" in capsys.readouterr().err
