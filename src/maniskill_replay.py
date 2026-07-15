"""Optional ManiSkill trajectory replay orchestration.

The materialized ManiSkill source adapter intentionally does not import the simulator runtime. This
module preserves that boundary by validating lightweight inputs locally and launching ManiSkill's
documented replay module in a child process only when replay is explicitly requested.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LePortError, OptionalDependencyError, ReplayError
from .sources.maniskill import ManiSkillAdapter

__all__ = ["ManiSkillReplayOptions", "ManiSkillReplayResult", "run_maniskill_replay"]

_REPLAY_MODULE = "mani_skill.trajectory.replay_trajectory"
_DIAGNOSTIC_LIMIT = 2_000


@dataclass(frozen=True)
class ManiSkillReplayOptions:
    """Explicit options passed to ManiSkill's supported trajectory replay command."""

    obs_mode: str = "rgb"
    use_env_states: bool = False
    target_control_mode: str | None = None
    sim_backend: str | None = None
    count: int | None = None
    num_envs: int = 1
    record_rewards: bool = False
    reward_mode: str | None = None
    allow_failure: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous strings and non-positive process or episode counts."""

        for name in ("obs_mode", "target_control_mode", "sim_backend", "reward_mode"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ReplayError(
                    "ManiSkill replay string options must be non-empty",
                    context={"option": name, "value": value},
                )
        if self.count is not None and self.count <= 0:
            raise ReplayError(
                "ManiSkill replay count must be positive",
                context={"count": self.count},
            )
        if self.num_envs <= 0:
            raise ReplayError(
                "ManiSkill replay num_envs must be positive",
                context={"num_envs": self.num_envs},
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible effective replay options."""

        return {
            "obs_mode": self.obs_mode,
            "use_env_states": self.use_env_states,
            "target_control_mode": self.target_control_mode,
            "sim_backend": self.sim_backend,
            "count": self.count,
            "num_envs": self.num_envs,
            "record_rewards": self.record_rewards,
            "reward_mode": self.reward_mode,
            "allow_failure": self.allow_failure,
        }


@dataclass(frozen=True)
class ManiSkillReplayResult:
    """Validated output pair produced by one successful ManiSkill replay process."""

    source: Path
    output_hdf5: Path
    output_json: Path
    options: ManiSkillReplayOptions
    runtime_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON/YAML output for public API and CLI callers."""

        return {
            "source": str(self.source),
            "output_hdf5": str(self.output_hdf5),
            "output_json": str(self.output_json),
            "options": self.options.to_dict(),
            "runtime_summary": self.runtime_summary,
        }


def run_maniskill_replay(
    source: str | Path,
    *,
    options: ManiSkillReplayOptions,
) -> ManiSkillReplayResult:
    """Run the optional ManiSkill replay module and identify one new validated output pair."""

    source_path = Path(source).expanduser().resolve()
    metadata_path = source_path.with_suffix(".json")
    if source_path.suffix.lower() != ".h5" or not source_path.is_file():
        raise ReplayError(
            "ManiSkill replay source must be a regular .h5 file",
            context={"source": str(source_path)},
        )
    if not metadata_path.is_file():
        raise ReplayError(
            "ManiSkill replay source is missing its same-basename JSON metadata file",
            context={"source": str(source_path), "metadata": str(metadata_path)},
        )
    try:
        metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(
            "Could not parse ManiSkill replay metadata",
            context={"metadata": str(metadata_path), "reason": str(exc)},
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("env_info"), dict):
        raise ReplayError(
            "ManiSkill replay metadata requires an env_info object",
            context={"metadata": str(metadata_path)},
        )

    env_info = metadata["env_info"]
    env_kwargs = env_info.get("env_kwargs") if isinstance(env_info.get("env_kwargs"), dict) else {}
    episodes = metadata.get("episodes") if isinstance(metadata.get("episodes"), list) else []
    first_episode = episodes[0] if episodes and isinstance(episodes[0], dict) else {}
    effective_control_mode = (
        options.target_control_mode or env_kwargs.get("control_mode") or first_episode.get("control_mode")
    )
    effective_backend = options.sim_backend or env_kwargs.get("sim_backend") or "physx_cpu"
    if effective_backend == "auto":
        effective_backend = "physx_cpu"
    base_name = source_path.stem.split(".", 1)[0]
    predictable_output: Path | None = None
    if isinstance(effective_control_mode, str) and isinstance(effective_backend, str):
        predictable_output = source_path.parent / (
            f"{base_name}.{options.obs_mode}.{effective_control_mode}.{effective_backend}.h5"
        )
        predictable_json = predictable_output.with_suffix(".json")
        if predictable_output.exists() or predictable_json.exists():
            raise ReplayError(
                "ManiSkill replay output already exists",
                context={
                    "output_hdf5": str(predictable_output),
                    "output_json": str(predictable_json),
                },
            )

    try:
        runtime_available = importlib.util.find_spec("mani_skill") is not None
    except (ImportError, ValueError):
        runtime_available = False
    if not runtime_available:
        raise OptionalDependencyError(
            "ManiSkill replay requires the simulator runtime; run `uv sync --extra maniskill-replay`",
            context={
                "feature": "maniskill-replay",
                "extra": "maniskill-replay",
                "dependency": "mani_skill",
            },
        )

    paired_before = _paired_hdf5_files(source_path.parent)
    command = [
        sys.executable,
        "-m",
        _REPLAY_MODULE,
        "--traj-path",
        str(source_path),
        "--save-traj",
        "--obs-mode",
        options.obs_mode,
        "--num-envs",
        str(options.num_envs),
    ]
    if options.use_env_states:
        command.append("--use-env-states")
    if options.target_control_mode is not None:
        command.extend(("--target-control-mode", options.target_control_mode))
    if options.sim_backend is not None:
        command.extend(("--sim-backend", options.sim_backend))
    if options.count is not None:
        command.extend(("--count", str(options.count)))
    if options.record_rewards:
        command.append("--record-rewards")
    if options.reward_mode is not None:
        command.extend(("--reward-mode", options.reward_mode))
    if options.allow_failure:
        command.append("--allow-failure")

    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise ReplayError(
            "Could not start the ManiSkill replay process",
            context={"reason": str(exc), "module": _REPLAY_MODULE},
        ) from exc
    if completed.returncode != 0:
        diagnostics = (completed.stderr.strip() or completed.stdout.strip())[-_DIAGNOSTIC_LIMIT:]
        raise ReplayError(
            "ManiSkill replay process failed",
            context={"returncode": completed.returncode, "diagnostics": diagnostics},
        )

    paired_after = _paired_hdf5_files(source_path.parent)
    generated = sorted(paired_after - paired_before)
    if predictable_output is not None and predictable_output.resolve() in generated:
        generated = [predictable_output.resolve()]
    if len(generated) != 1 or generated[0] == source_path:
        raise ReplayError(
            "ManiSkill replay did not create exactly one new trajectory pair",
            context={"candidates": [str(path) for path in generated]},
        )

    output_hdf5 = generated[0]
    try:
        ManiSkillAdapter().inspect(output_hdf5)
    except LePortError as exc:
        raise ReplayError(
            "ManiSkill replay output is not a valid materialized trajectory pair",
            context={"output": str(output_hdf5), "reason": str(exc)},
        ) from exc

    summary_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    replay_summary_lines = [
        line[line.find("Replayed ") :]
        for line in summary_lines
        if "Replayed " in line and "demos saved" in line
    ]
    return ManiSkillReplayResult(
        source=source_path,
        output_hdf5=output_hdf5,
        output_json=output_hdf5.with_suffix(".json"),
        options=options,
        runtime_summary=(
            replay_summary_lines[-1] if replay_summary_lines else (summary_lines[-1] if summary_lines else "")
        ),
    )


def _paired_hdf5_files(directory: Path) -> set[Path]:
    """Return resolved HDF5 files that currently have a same-basename JSON companion."""

    return {
        path.resolve()
        for path in directory.glob("*.h5")
        if path.is_file() and path.with_suffix(".json").is_file()
    }
