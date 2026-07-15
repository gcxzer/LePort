# Convert ALOHA HDF5 recordings to LeRobot

## Inputs

- `source`: one `episode_<integer>.hdf5` file or a directory containing immediate matching files.
- `target`: a new LeRobot dataset directory. It must not exist or must be empty.
- `repo_id`: LeRobot repository identifier.
- `fps`: a positive integer confirmed from the recording setup.
- `task`: static task text, or the name of a string root attribute available in every selected episode.
- `action_source`: normally `action`, after confirming it with inspection.
- `state_sources`: ordered selectors such as `observations/qpos` and `observations/qvel`.
- `image_sources`: explicit source-to-target mappings for inspected camera selectors.
- `robot_type`: explicit LeRobot robot metadata when known.
- `selection`: every episode or an explicit list of episode IDs.

LePort does not infer FPS, task text, robot type, action meaning, joint order, units, or target feature
names from ALOHA conventions. Confirm these semantics before creating a plan.

## Accepted source structure

```text
<source-directory>/
├── episode_0.hdf5
├── episode_2.hdf5
├── episode_10.hdf5
└── unrelated entries are ignored

episode_<integer>.hdf5
├── action                              required frame-addressable Dataset
├── observations/                       required Group
│   ├── qpos                            required frame-addressable Dataset
│   ├── qvel                            optional frame-addressable Dataset
│   ├── effort                          optional frame-addressable Dataset
│   └── images/
│       ├── cam_high                    optional raw or per-frame JPEG Dataset
│       └── cam_wrist                   optional raw or per-frame JPEG Dataset
└── compress_len                        optional compression metadata, not a field selector
```

A matching filename is not enough. The adapter opens a candidate read-only and requires a root
`action` Dataset plus a frame-addressable `observations/qpos` Dataset. Directory discovery is
non-recursive. Files are ordered by the integer suffix, so `episode_2` precedes `episode_10`.
Different filenames that resolve to the same numeric ID, such as `episode_1.hdf5` and
`episode_01.hdf5`, are rejected.

`action.shape[0]` is the episode length. Every mapped field must have exactly that length. LePort does
not truncate, pad, shift, resample, or otherwise align values.

## Selectors and image storage

Inspection exposes frame-addressable root Datasets and Datasets nested below `observations` as
slash-separated selectors without a leading slash. Common selectors are:

| Selector | Source value |
|---|---|
| `action` | Numeric action array; semantics and units remain explicit caller inputs |
| `observations/qpos` | Joint-position values in source order |
| `observations/qvel` | Optional joint-velocity values in source order |
| `observations/effort` | Optional effort values in source order |
| `observations/images/cam_high` | Raw HWC `uint8` frames or per-frame JPEG buffers |
| `observations/images/cam_wrist` | Raw HWC `uint8` frames or per-frame JPEG buffers |

Raw cameras are returned without color conversion. JPEG cameras may use fixed-width padded `uint8`
rows or variable-length `uint8` values. The adapter decodes only the current requested frame and
returns HWC RGB `uint8`. Inspection decodes one representative JPEG per camera and episode so its
reported dtype and shape describe decoded pixels rather than encoded byte width. `compress_len` is
preserved as metadata and cannot be mapped as a frame field.

## CLI workflow

### 1. Inspect the source

```bash
uv run leport inspect "<aloha-directory>" --adapter aloha --json
```

The adapter can also be selected automatically when ALOHA is the unique structural match:

```bash
uv run leport inspect "<aloha-directory>" --json
```

Inspect one file or a canonical explicit subset:

```bash
uv run leport inspect "<aloha-directory>/episode_0.hdf5" --adapter aloha --json

uv run leport inspect "<aloha-directory>" \
  --adapter aloha \
  --episode episode_0,episode_10 \
  --json
```

ALOHA has no mask table, so `--filter-key` is unsupported. Explicit IDs may be supplied in any order,
but inspection and conversion always restore numeric source order.

Review `episode_ids`, `episode_lengths`, every selected field's `dtypes`, `shapes`,
`missing_episodes`, and `schema_consistent`, plus `metadata.episode_attributes`. Do not plan a field
that is missing or inconsistent across the selection.

### 2. Create a plan

This example maps two state arrays and two cameras to image features:

```bash
uv run leport plan \
  --source "<aloha-directory>" \
  --output "<plan.yaml>" \
  --adapter aloha \
  --episode episode_0,episode_10 \
  --target "<target-directory>" \
  --repo-id "<namespace>/<dataset-name>" \
  --robot-type "<confirmed-robot-type>" \
  --fps 50 \
  --task "<confirmed-task-text>" \
  --action action \
  --action-dtype float32 \
  --state observations/qpos \
  --state observations/qvel \
  --state-dtype float32 \
  --image observations/images/cam_high=observation.images.high \
  --image observations/images/cam_wrist=observation.images.wrist \
  --no-videos \
  --json
```

`--state` order defines the concatenation order. `--no-videos` stores individual image frames; omit
it to create video features. Use `--task-metadata <root-attribute>` instead of `--task` only when
inspection confirms that every selected episode supplies a non-empty string attribute.

Check the saved plan without accessing the source:

```bash
uv run leport plan --check "<plan.yaml>" --json
```

### 3. Convert and validate

```bash
uv run leport convert --config "<plan.yaml>" --json

uv run leport validate "<target-directory>" \
  --config "<plan.yaml>" \
  --json
```

Source-aware validation confirms selected episode counts and lengths, planned features, task values,
and decoded visual features. Conversion writes to a sibling staging directory and commits only after
LeRobot reload validation succeeds.

## Python workflow

```python
from pathlib import Path

from leport import convert, create_plan, inspect, validate
from leport.sources import EpisodeSelection

source = Path("<aloha-directory>").expanduser().resolve()
target = Path("<target-directory>").expanduser().resolve()
selection = EpisodeSelection(episode_ids=("episode_0", "episode_10"))

inspection = inspect(source, adapter="aloha", selection=selection)
print(inspection.to_dict())

plan = create_plan(
    source,
    target_root=target,
    repo_id="<namespace>/<dataset-name>",
    robot_type="<confirmed-robot-type>",
    fps=50,
    task="<confirmed-task-text>",
    action_source="action",
    action_dtype="float32",
    state_sources=("observations/qpos", "observations/qvel"),
    state_dtype="float32",
    image_sources={
        "observations/images/cam_high": "observation.images.high",
        "observations/images/cam_wrist": "observation.images.wrist",
    },
    use_videos=False,
    adapter="aloha",
    selection=selection,
)

result = convert(plan)
report = validate(result.target, plan=plan)
print(report.to_dict())
```

## Troubleshooting

- `optional_dependency_missing`: run `uv sync --extra aloha`; HDF5 access uses `h5py` and JPEG
  decoding uses Pillow.
- `adapter_not_found`: confirm the filename matches `episode_<integer>.hdf5` and the file contains
  frame-addressable `action` and `observations/qpos` Datasets.
- Unknown episode ID: copy exact IDs from inspection. Do not use a path or numeric suffix alone.
- Filter-key failure: select explicit IDs because standard ALOHA directories have no mask table.
- Field-length failure: repair or exclude the source episode. LePort never truncates or pads it.
- JPEG decode failure: the diagnostic identifies the episode, frame, and camera selector. Confirm the
  dataset contains one valid JPEG byte buffer per timestep.
- Inconsistent camera shape: select a uniform subset or repair the source; LePort does not resize.
- Existing target or plan: choose a new path or move the existing output explicitly. LePort does not
  overwrite reviewed plans or append to existing datasets.
