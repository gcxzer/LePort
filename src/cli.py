"""Argparse CLI that delegates all domain logic to the public Python API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .api import convert, create_plan, inspect, merge, replay_maniskill, validate
from .conversion.plan import load_plan, save_plan
from .errors import LePortError, PlanValidationError
from .sources.types import EpisodeSelection

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert stable LePort errors into exit code 2."""

    parser = argparse.ArgumentParser(prog="leport", description="Convert robot data to LeRobot Dataset v3")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect source structure without writing")
    inspect_parser.add_argument("source", type=Path)
    inspect_parser.add_argument("--adapter")
    inspect_parser.add_argument(
        "--episode",
        action="append",
        metavar="ID[,ID...]",
        help="Comma-separated episode IDs, for example demo_0,demo_1,demo_2",
    )
    inspect_parser.add_argument("--filter-key")
    inspect_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="Create or validate ConversionPlan YAML")
    plan_mode = plan_parser.add_mutually_exclusive_group(required=True)
    plan_mode.add_argument("--source", type=Path)
    plan_mode.add_argument("--check", type=Path)
    plan_parser.add_argument("--output", type=Path)
    plan_parser.add_argument("--adapter")
    plan_parser.add_argument(
        "--episode",
        action="append",
        metavar="ID[,ID...]",
        help="Comma-separated episode IDs, for example demo_0,demo_1,demo_2",
    )
    plan_parser.add_argument("--filter-key")
    plan_parser.add_argument("--target", type=Path)
    plan_parser.add_argument("--repo-id")
    plan_parser.add_argument("--robot-type")
    plan_parser.add_argument("--fps", type=int)
    task_group = plan_parser.add_mutually_exclusive_group()
    task_group.add_argument("--task")
    task_group.add_argument("--task-metadata")
    plan_parser.add_argument("--action")
    plan_parser.add_argument("--action-dtype")
    plan_parser.add_argument("--state", action="append")
    plan_parser.add_argument("--state-dtype")
    plan_parser.add_argument(
        "--image",
        action="append",
        metavar="SOURCE=TARGET",
        help="For example obs/agentview=observation.images.agentview",
    )
    plan_parser.add_argument("--no-videos", action="store_true")
    plan_parser.add_argument("--json", action="store_true")

    convert_parser = subparsers.add_parser("convert", help="Execute a validated conversion plan")
    convert_parser.add_argument("--config", type=Path, required=True)
    convert_parser.add_argument("--json", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay-maniskill",
        help="Materialize a new ManiSkill trajectory pair with the optional simulator runtime",
    )
    replay_parser.add_argument("source", type=Path)
    replay_parser.add_argument("--obs-mode", default="rgb")
    replay_parser.add_argument("--use-env-states", action="store_true")
    replay_parser.add_argument("--target-control-mode")
    replay_parser.add_argument("--sim-backend")
    replay_parser.add_argument("--count", type=int)
    replay_parser.add_argument("--num-envs", type=int, default=1)
    replay_parser.add_argument("--record-rewards", action="store_true")
    replay_parser.add_argument("--reward-mode")
    replay_parser.add_argument("--allow-failure", action="store_true")
    replay_parser.add_argument("--json", action="store_true")

    merge_parser = subparsers.add_parser("merge", help="Merge existing LeRobot datasets")
    merge_parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        metavar="SOURCE",
        help="Two or more LeRobot dataset directories in the desired episode order",
    )
    merge_parser.add_argument("--target", type=Path, required=True)
    merge_parser.add_argument("--repo-id", required=True)
    merge_parser.add_argument(
        "--no-concatenate-videos",
        action="store_true",
        help="Preserve video shards instead of concatenating compatible inputs into larger MP4 files",
    )
    merge_parser.add_argument(
        "--no-concatenate-data",
        action="store_true",
        help="Preserve Parquet shards instead of concatenating compatible inputs into larger files",
    )
    merge_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Reload and validate a LeRobot dataset")
    validate_parser.add_argument("target", type=Path)
    validate_parser.add_argument("--repo-id")
    validate_parser.add_argument("--config", type=Path)
    validate_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspection_result = inspect(
                args.source,
                adapter=args.adapter,
                selection=_selection_from_args(args),
            )
            _emit(inspection_result.to_dict(), as_json=args.json)
            return 0

        if args.command == "plan":
            if args.check is not None:
                checked = load_plan(args.check)
                _emit(checked.to_dict(), as_json=args.json)
                return 0
            required_values = {
                "--output": args.output,
                "--target": args.target,
                "--repo-id": args.repo_id,
                "--fps": args.fps,
                "--action": args.action,
            }
            missing = [name for name, value in required_values.items() if value is None]
            if missing:
                raise PlanValidationError("Required plan arguments are missing", context={"missing": missing})
            images: dict[str, str] = {}
            for value in args.image or ():
                if "=" not in value:
                    raise PlanValidationError(
                        "--image must use SOURCE=TARGET format", context={"value": value}
                    )
                source_selector, target_name = value.split("=", 1)
                if not source_selector or not target_name or source_selector in images:
                    raise PlanValidationError(
                        "--image mapping is empty or duplicated", context={"value": value}
                    )
                images[source_selector] = target_name
            generated = create_plan(
                args.source,
                target_root=args.target,
                repo_id=args.repo_id,
                fps=args.fps,
                action_source=args.action,
                task=args.task,
                task_metadata=args.task_metadata,
                state_sources=tuple(args.state or ()),
                image_sources=images,
                action_dtype=args.action_dtype,
                state_dtype=args.state_dtype,
                robot_type=args.robot_type,
                use_videos=not args.no_videos,
                adapter=args.adapter,
                selection=_selection_from_args(args),
            )
            save_plan(generated, args.output)
            _emit(generated.to_dict(), as_json=args.json)
            return 0

        if args.command == "convert":
            conversion_result = convert(args.config)
            _emit(
                {
                    "target": str(conversion_result.target),
                    "total_episodes": conversion_result.validation.total_episodes,
                    "total_frames": conversion_result.validation.total_frames,
                    "validation": conversion_result.validation.to_dict(),
                },
                as_json=args.json,
            )
            return 0

        if args.command == "replay-maniskill":
            replay_result = replay_maniskill(
                args.source,
                obs_mode=args.obs_mode,
                use_env_states=args.use_env_states,
                target_control_mode=args.target_control_mode,
                sim_backend=args.sim_backend,
                count=args.count,
                num_envs=args.num_envs,
                record_rewards=args.record_rewards,
                reward_mode=args.reward_mode,
                allow_failure=args.allow_failure,
            )
            _emit(replay_result.to_dict(), as_json=args.json)
            return 0

        if args.command == "merge":
            merge_result = merge(
                args.sources,
                target_root=args.target,
                repo_id=args.repo_id,
                concatenate_videos=not args.no_concatenate_videos,
                concatenate_data=not args.no_concatenate_data,
            )
            _emit(merge_result.to_dict(), as_json=args.json)
            return 0

        if args.command == "validate":
            validation_result = validate(args.target, repo_id=args.repo_id, plan=args.config)
            _emit(validation_result.to_dict(), as_json=args.json)
            return 0
    except LePortError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


def _selection_from_args(args: argparse.Namespace) -> EpisodeSelection:
    """Convert shared CLI episode arguments into the strict selection model."""

    try:
        # ``append`` detects repeated --episode options so the CLI can reject ambiguity instead of
        # silently retaining only the last value.
        episode_arguments = tuple(args.episode or ())
        if len(episode_arguments) > 1:
            raise ValueError("--episode may appear only once; separate multiple episode IDs with commas")

        episode_ids: tuple[str, ...] = ()
        if episode_arguments:
            # Whitespace after commas is accepted, but empty entries such as demo_0,,demo_2 are
            # rejected before the source file is opened.
            episode_ids = tuple(item.strip() for item in episode_arguments[0].split(","))
            if any(not episode_id for episode_id in episode_ids):
                raise ValueError("--episode comma-separated values cannot contain an empty episode ID")
        return EpisodeSelection(episode_ids, args.filter_key)
    except ValueError as exc:
        raise PlanValidationError(
            "Episode selection arguments are invalid", context={"reason": str(exc)}
        ) from exc


def _emit(value: Any, *, as_json: bool) -> None:
    """Emit machine-readable JSON or human-readable YAML consistently."""

    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip())
