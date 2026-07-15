# Convert robomimic to LeRobot

## Inputs

- `source`: robomimic HDF5 file.
- `target`: new LeRobot dataset directory. It must not exist or must be empty.
- `repo_id`: LeRobot repository identifier.
- `fps`: positive integer.
- `task`: static task text, or the name of an episode attribute containing task text.
- `action_source`: source selector for `action`.
- `action_dtype`: target action dtype (optional; otherwise preserve the source dtype).
- `state_sources`: ordered source selectors to concatenate into `observation.state` (optional).
- `state_dtype`: target state dtype (required when selected state fields have different dtypes).
- `image_sources`: source-to-target image mappings (optional).
- `robot_type`: LeRobot robot metadata (optional).
- `use_videos`: store visual observations as videos or individual images.
- `selection`: all episodes, explicit episode ids, or one filter key.

If FPS, task, or action semantics are unknown, inspect the source first and request the missing
information before creating a plan.

## Workflow

### 1. Inspect the source

```bash
uv run leport inspect "<source.hdf5>" --adapter robomimic --json
```

Read these fields from the result:

- `episode_ids`, `episode_lengths`, `total_frames`
- `fields[].selector`, `dtypes`, `shapes`
- `fields[].missing_episodes`, `schema_consistent`, `image_candidate`
- `metadata.data_attributes.env_args`
- `metadata.filter_keys`
- `diagnostics`

Do not create a plan with a selector that is absent, missing from any selected episode, or schema
inconsistent. Use `env_args.env_kwargs.control_freq` only as evidence for an FPS choice; do not adopt
it silently.

### 2. Choose episodes

Omit both options to select every episode.

Select explicit episodes:

```bash
uv run leport inspect "<source.hdf5>" \
  --adapter robomimic \
  --episode demo_0,demo_1 \
  --json
```

Select a mask stored at `mask/<filter-key>`:

```bash
uv run leport inspect "<source.hdf5>" \
  --adapter robomimic \
  --filter-key "<filter-key>" \
  --json
```

Pass `--episode` once and join multiple IDs with commas. Never combine `--episode` and
`--filter-key`. Explicit ids are emitted in numeric demo order, not in
argument order.

### 3. Create a plan

Use the following pattern for low-dimensional data:

```bash
uv run leport plan \
  --source "<source.hdf5>" \
  --output "<plan.yaml>" \
  --adapter robomimic \
  --target "<target-directory>" \
  --repo-id "<namespace>/<dataset-name>" \
  --robot-type "<robot-type>" \
  --fps "<fps>" \
  --task "<task>" \
  --action "<action-selector>" \
  --action-dtype float32 \
  --state "<state-selector-1>" \
  --state "<state-selector-2>" \
  --state-dtype float32 \
  --no-videos
```

`--state` order defines the concatenation order. The resulting shape is the sum of the flattened
source shapes.

Use an episode attribute instead of static task text when every selected episode contains a non-empty
string attribute:

```bash
--task-metadata "<episode-attribute>"
```

Map inspected `uint8` three-dimensional image fields with:

```bash
--image "<source-image-selector>=observation.images.<camera-name>"
```

Repeat `--image` for multiple cameras. Omit `--no-videos` to encode visual observations as videos;
include it to store image frames.

Apply episode selection to `plan` with the same `--episode` or `--filter-key` option used during
inspection.

### 4. Check the plan

```bash
uv run leport plan --check "<plan.yaml>" --json
```

This checks YAML structure only. Conversion performs a full preflight over all selected episodes and
maps the first frame of each episode before writing.

### 5. Convert

```bash
uv run leport convert --config "<plan.yaml>" --json
```

LePort writes to a temporary sibling directory, finalizes the dataset, reloads it with LeRobot, and
then atomically moves it to `target`. A failed conversion does not commit a partial target.

### 6. Validate against the source

```bash
uv run leport validate "<target-directory>" \
  --config "<plan.yaml>" \
  --json
```

Confirm:

- episode count and per-episode lengths match the selected source episodes;
- total frame count matches inspection;
- `action` and every planned observation feature exist;
- task values match the plan;
- every visual feature appears in `decoded_visual_features`.

### 7. Merge converted LeRobot datasets (optional)

`merge` accepts two or more existing LeRobot dataset directories in one call; it does not accept raw
HDF5 files. There is no two-input upper limit. Source order is preserved: all episodes from the first
directory precede all episodes from the second directory, followed by the third directory, and so on.

```bash
uv run leport merge \
  "<first-lerobot-directory>" \
  "<second-lerobot-directory>" \
  "<third-lerobot-directory>" \
  --target "<new-merged-directory>" \
  --repo-id "<namespace>/<merged-dataset-name>" \
  --json
```

Every input must use the same FPS, robot type, and complete feature schema. Task strings may differ;
LeRobot rebuilds the merged task table and task indices. LePort never drops, pads, renames, casts, or
resamples incompatible features during merge. The same directory cannot appear twice, but LePort does
not deduplicate equal content stored in different directories. Merging two independently converted
copies of the same source therefore creates two copies of those episodes in the output.

By default, the official LeRobot merger concatenates compatible video and Parquet shards. For a small
dataset this commonly produces one Parquet file and one MP4 per visual feature, not one MP4 per episode.
This is expected: episode-to-video timestamp ranges remain in the episode metadata. Larger datasets may
still be split according to LeRobot shard limits. Preserve separate source shards while rebuilding valid
metadata with:

```bash
--no-concatenate-videos --no-concatenate-data
```

The target is always a new dataset. A non-empty target is rejected, and a staged merge is committed
only after episode lengths, tasks, features, and representative visual boundary frames pass reload
validation. Inputs are never modified. Validate an already merged result independently with:

```bash
uv run leport validate "<new-merged-directory>" \
  --repo-id "<namespace>/<merged-dataset-name>" \
  --json
```

For a correct result, `total_episodes` and `total_frames` equal the sums from all inputs, the reported
episode lengths follow the declared source order, and every video feature appears in
`decoded_visual_features`. A single MP4 per camera is not an error when video concatenation is enabled.

## Python API

```python
from pathlib import Path

from leport import convert, create_plan, inspect, merge, validate
from leport.sources import EpisodeSelection

# Replace these values with the requested source and target configuration.
source = Path("<source.hdf5>").expanduser().resolve()
target = Path("<target-directory>").expanduser().resolve()
selection = EpisodeSelection()

# Inspect before selecting fields for the conversion plan.
inspection = inspect(source, adapter="robomimic", selection=selection)
print(inspection.to_dict())

# Preserve the state selector order because it defines observation.state layout.
plan = create_plan(
    source,
    target_root=target,
    repo_id="<namespace>/<dataset-name>",
    robot_type="<robot-type>",
    fps=20,
    task="<task>",
    action_source="actions",
    action_dtype="float32",
    state_sources=(
        "<state-selector-1>",
        "<state-selector-2>",
    ),
    state_dtype="float32",
    use_videos=False,
    adapter="robomimic",
    selection=selection,
)

# Conversion includes preflight, writing, finalization, reload validation, and atomic commit.
result = convert(plan)

# Supplying the plan enables source-to-target episode and schema comparison.
report = validate(result.target, plan=plan)
print(report.to_dict())

# Merge two or more converted LeRobot datasets; raw HDF5 files are not valid inputs.
merge_result = merge(
    (
        "<first-lerobot-directory>",
        "<second-lerobot-directory>",
        "<third-lerobot-directory>",
    ),
    target_root="<new-merged-directory>",
    repo_id="<namespace>/<merged-dataset-name>",
)
print(merge_result.to_dict())
```

Selection variants:

```python
from leport.sources import EpisodeSelection

# Select one robomimic mask.
train_selection = EpisodeSelection(filter_key="train")

# Select explicit episodes. The adapter restores numeric demo order.
sample_selection = EpisodeSelection(episode_ids=("demo_0", "demo_1"))
```

Catch structured failures without parsing error messages:

```python
from leport.errors import LePortError

try:
    result = convert(plan)
except LePortError as error:
    # Use code for branching and context for selector, episode, frame, or dependency details.
    print(error.code)
    print(error.context)
    raise
```

## Source contract

```text
/
├── data/
│   ├── demo_0/
│   │   ├── actions              required Dataset
│   │   ├── states               optional Dataset
│   │   ├── rewards              optional Dataset
│   │   ├── dones                optional Dataset
│   │   ├── obs/                 optional Group of Datasets
│   │   └── next_obs/            optional Group of Datasets
│   └── demo_1/ ...
└── mask/                        optional Group of demo-id Datasets
```

- Accept episode names matching `demo_<integer>` and process them in numeric order.
- Use `actions.shape[0]` as the episode length.
- Require every mapped field to have the same first dimension as `actions`.
- Expose top-level Datasets and Datasets nested below `obs` or `next_obs`.
- Address fields relative to the episode, for example `actions`, `states`, or
  `obs/robot0_eef_pos`.
- Treat `num_samples`, when present, as a consistency check rather than a replacement for the actual
  Dataset length.

## Mapping behavior

- `direct`: map one source selector to one target feature.
- `concat`: flatten source values and concatenate them in declared order.
- `cast`: cast numeric values only; the cast dtype must equal the target dtype.
- image/video: accept a three-dimensional `uint8` frame in HWC or CHW layout.
- task: use static text or a direct episode metadata string.

LePort does not transform action semantics, coordinate frames, quaternion conventions, units, joint
order, normalization, or temporal alignment. It does not truncate, pad, shift, or merge episodes.

## Common robosuite selectors

Use this table only after confirming the selectors with `inspect`.

| Selector | Typical meaning |
|---|---|
| `actions` | Controller input; absolute/delta semantics depend on collection configuration |
| `states` | Simulator state; usually not identical to policy observation |
| `obs/robot0_eef_pos` | End-effector position |
| `obs/robot0_eef_quat_site` | End-effector site quaternion |
| `obs/robot0_gripper_qpos` | Gripper joint position |
| `obs/robot0_joint_pos` | Arm joint position |
| `obs/robot0_joint_vel` | Arm joint velocity |
| `obs/object` | Task-specific object state vector |
| `next_obs/...` | Next-step observation |
| `rewards` | Environment reward; not mapped automatically |
| `dones` | Termination flag; not mapped automatically |

## Troubleshooting

| Failure | Action |
|---|---|
| No adapter matches | Verify the file path and HDF5 root structure |
| `h5py` is missing | Install the `leport[robomimic]` extra with the active environment manager |
| Selector is absent | Use the exact selector returned by `inspect` |
| Schema differs across episodes | Report the differing episodes; do not choose one shape silently |
| Field length differs from `actions` | Stop; do not truncate or pad |
| Task metadata is missing or not a string | Use a valid episode attribute or static task text |
| Target is non-empty | Choose a new target; do not delete existing data without authorization |
| Merge inputs have different schemas | Align FPS, robot type, feature names, dtypes, and shapes before merging |
| Merge output contains duplicated episodes | Check whether different input directories were converted from the same source; merge preserves data and does not deduplicate content |
| One merged MP4 exists for each camera | This is expected with default video concatenation; use `--no-concatenate-videos` to preserve separate video shards |
| LeRobot reload fails | Report the error code and context; verify dependency version and target schema |

## Guardrails

- Do not infer action semantics from field name or shape.
- Do not infer image availability from `raw`, `low_dim`, or `image` in a filename.
- Do not mix `obs/...` with `next_obs/...` unless the requested mapping explicitly requires it.
- Do not map a field with missing episodes or inconsistent dtype/shape.
- Do not modify the source HDF5.
- Do not overwrite or append to a non-empty target.
- Do not treat merge as data deduplication; verify that every input represents the intended episodes.
- Do not report success until source-to-target validation passes.
