# Convert ManiSkill trajectories to LeRobot

LePort reads materialized ManiSkill trajectories through the `maniskill` source adapter. Inspection and
conversion do not import the ManiSkill runtime, replay simulations, render missing observations, convert
control modes, or infer robot semantics. A separate opt-in replay command delegates preprocessing to the
official ManiSkill runtime when observations must be materialized.

## Install the adapter dependency

```bash
uv sync --extra maniskill --group dev
```

The extra contains `h5py`. Reading an existing trajectory pair does not require the `mani_skill` package.

Install replay separately only when simulation or rendering is required:

```bash
uv sync --extra maniskill --extra maniskill-replay --group dev
```

The replay extra contains `mani-skill>=3.0.1,<4.0` and its platform-specific simulation dependencies.
Rendering may additionally require the host configuration documented by ManiSkill.

## Source contract

Pass the HDF5 file as `source`. A JSON file with the same basename must be beside it. The downloaded raw
sample is:

```text
trajectory.h5
trajectory.json
```

Replay creates a separate pair such as:

```text
trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5
trajectory.rgb.pd_joint_delta_pos.physx_cpu.json
```

The adapter requires:

- JSON `env_info` and `episodes` structures;
- one JSON episode object for every numeric HDF5 `traj_<episode_id>` group and no unmatched groups;
- a frame-addressable `actions` dataset in every trajectory;
- `episodes[*].elapsed_steps == actions.shape[0]`;
- unique integer episode IDs.

Public IDs retain the `traj_<episode_id>` form and are emitted in numeric order. Select all episodes or
an explicit list. `filter_key` is unsupported in this release because historical ManiSkill success
metadata does not define one stable filter table.

## Raw and replayed variants

Official compressed demonstrations commonly contain actions and `env_states` but no `obs`. LePort can
inspect and convert those files by mapping environment-state leaves explicitly. Use the optional replay
command before conversion only when RGB or another policy observation mode is needed:

```bash
uv run leport replay-maniskill "<raw-trajectory.h5>" \
  --obs-mode rgb \
  --use-env-states \
  --json
```

LePort invokes `python -m mani_skill.trajectory.replay_trajectory --save-traj` in an isolated process,
captures its logs, and returns the new same-basename HDF5 and JSON paths. ManiSkill names the output
`<base>.<obs_mode>.<control_mode>.<sim_backend>.h5`; for the downloaded sample and default CPU backend,
RGB output is `trajectory.rgb.pd_joint_pos.physx_cpu.h5`. LePort refuses a predictable existing output
instead of overwriting it.

`--use-env-states` is useful when replaying actions alone would diverge from the recorded simulation.
Additional explicit options include `--target-control-mode`, `--sim-backend`, `--count`, `--num-envs`,
`--record-rewards`, `--reward-mode`, and `--allow-failure`. Simulation, rendering, control conversion,
and reward computation remain ManiSkill responsibilities.

## Transition alignment

ManiSkill records `T` actions but `T+1` observations and environment states. For action index `i`, LePort
provides two explicit views:

| HDF5 leaf | Public selector | Source index | Public length |
|---|---|---:|---:|
| `actions` | `actions` | `i` | `T` |
| `terminated` | `terminated` | `i` | `T` |
| `obs/agent/qpos` | `obs/agent/qpos` | `i` | `T` |
| `obs/agent/qpos` | `next_obs/agent/qpos` | `i + 1` | `T` |
| `env_states/actors/cube` | `env_states/actors/cube` | `i` | `T` |
| `env_states/actors/cube` | `next_env_states/actors/cube` | `i + 1` | `T` |

The same projection applies recursively to every `obs` and `env_states` leaf. If either root is a dataset
instead of a group, its selectors are `obs`/`next_obs` or `env_states`/`next_env_states`. Direct
transition fields such as `rewards`, `success`, and `fail` must have length `T`. Observation and
environment-state leaves must have length `T+1`; LePort rejects other lengths rather than truncating,
padding, shifting, or resampling them.

An HWC `uint8` projected leaf is reported as an image candidate. Depth or segmentation arrays with other
dtypes remain numeric fields. LePort never resizes images or changes channel order in the adapter.

## Workflow

### 1. Inspect the pair

```bash
uv run leport inspect "<trajectory.h5>" \
  --adapter maniskill \
  --json
```

Review:

- `episode_ids`, `episode_lengths`, and `total_frames`;
- `fields[].selector`, dtype, shape, coverage, and `image_candidate`;
- `diagnostics` for fields missing from an episode or schema differences;
- `metadata.env_info`, provenance, filenames, and `episode_metadata`.

Metadata is evidence, not an instruction. LePort does not convert `env_info.env_id` into task text and
does not infer FPS, robot type, action meaning, units, or coordinate frames.

### 2. Select episodes

Omit `--episode` to use all episodes. To select a subset:

```bash
uv run leport inspect data/maniskill/PickCube-v1-teleop/trajectory.h5 \
  --adapter maniskill \
  --episode traj_0,traj_9 \
  --json
```

The result stays in numeric source order even when the argument order differs.

### 3. Create an explicit plan

```bash
uv run leport plan \
  --source "<trajectory.h5>" \
  --output "<plan.yaml>" \
  --adapter maniskill \
  --target "<target-directory>" \
  --repo-id "<namespace>/<dataset-name>" \
  --robot-type "<robot-type>" \
  --fps "<fps>" \
  --task "<natural-language-task>" \
  --action actions \
  --action-dtype float32 \
  --state env_states/articulations/panda \
  --state env_states/actors/cube \
  --state-dtype float32 \
  --no-videos
```

`--state` order defines the flattened concatenation order. Remove `--no-videos` to encode visual
features as videos. After replay, use the replayed HDF5 as `--source` and add
`--image "obs/sensor_data/base_camera/rgb=observation.images.base"`. Use complete selectors from
inspection; camera and agent names are not hard-coded.

Task, FPS, robot type, action meaning, controller convention, and state interpretation are explicit user
decisions. `control_mode` and `env_info` are preserved for review but never silently adopted.

### 4. Check, convert, and validate

```bash
uv run leport plan --check "<plan.yaml>" --json
uv run leport convert --config "<plan.yaml>" --json
uv run leport validate "<target-directory>" --config "<plan.yaml>" --json
```

Conversion runs source preflight before writing, stages output beside the target, reloads the completed
dataset through LeRobot, and commits atomically. Missing fields, schema differences, invalid alignment,
or a non-empty target leave no partial dataset.

### 5. Merge compatible converted datasets (optional)

Merge accepts completed LeRobot dataset directories, not ManiSkill HDF5 files. The notebook converts
`traj_0` as its primary dataset and can convert `traj_1` with the same schema before merging them:

```bash
uv run leport merge \
  outputs/maniskill-pickcube-demo \
  outputs/maniskill-pickcube-demo-b \
  --target outputs/maniskill-pickcube-demo-merged \
  --repo-id local/maniskill-pickcube-demo-merged \
  --json
```

Both inputs must have matching FPS, robot type, and feature schemas. Merge creates a new target and does
not modify either input.

## Python API

```python
from pathlib import Path

from leport import convert, create_plan, inspect, replay_maniskill, validate
from leport.sources import EpisodeSelection

source = Path("data/maniskill/PickCube-v1-teleop/trajectory.h5").resolve()
selection = EpisodeSelection(episode_ids=("traj_0", "traj_9"))

inspection = inspect(source, adapter="maniskill", selection=selection)
print(inspection.to_dict())

plan = create_plan(
    source,
    target_root="<target-directory>",
    repo_id="<namespace>/<dataset-name>",
    robot_type="<robot-type>",
    fps=30,
    task="<natural-language-task>",
    action_source="actions",
    action_dtype="float32",
    state_sources=("env_states/articulations/panda", "env_states/actors/cube"),
    state_dtype="float32",
    image_sources={},
    use_videos=False,
    adapter="maniskill",
    selection=selection,
)

result = convert(plan)
report = validate(result.target, plan=plan)
print(report.to_dict())
```

To materialize RGB first in Python:

```python
replay = replay_maniskill(source, obs_mode="rgb", use_env_states=True)
inspection = inspect(replay.output_hdf5, adapter="maniskill")
print(inspection.to_dict())
```

The runnable [ManiSkill notebook](../notebooks/maniskill.ipynb) uses the downloaded PickCube-v1
teleoperation pair, converts `traj_0` by default, exposes replay through `RUN_REPLAY`, and optionally
converts `traj_1` and merges both completed datasets through `RUN_MERGE`.
