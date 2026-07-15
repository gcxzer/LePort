# Convert processed UMI replay buffers to LeRobot

LePort reads training-ready Universal Manipulation Interface replay buffers through the `umi` source
adapter. The supported input is an existing Zarr v2 ZipStore or directory store. Raw GoPro videos, IMU
streams, camera calibration, gripper tracking, and the UMI SLAM pipeline are outside this adapter.

## Install the reader

```bash
uv sync --extra umi --group dev
```

The optional extra provides Zarr v2 and `imagecodecs`. The codec registration is required because
official processed RGB arrays can use the `imagecodecs_jpegxl` compressor.

## Processed source layout

Pass either `dataset.zarr.zip` or the root of an equivalent Zarr directory store:

```text
dataset.zarr.zip
├── data/
│   ├── camera0_rgb
│   ├── robot0_eef_pos
│   ├── robot0_eef_rot_axis_angle
│   ├── robot0_gripper_width
│   ├── robot0_demo_start_pose
│   └── robot0_demo_end_pose
└── meta/
    └── episode_ends
```

The adapter requires the three `robot0_*` motion fields, at least one `camera<integer>_rgb` field, and
valid cumulative episode boundaries. Every array directly below `data` must have the same leading time
dimension. Nested data groups, scalar data arrays, misaligned fields, and Zarr v3 stores are rejected.

For boundaries `[120, 270, 400]`, LePort exposes:

| Public episode | Global source slice | Length |
|---|---:|---:|
| `episode_0` | `[0:120]` | 120 |
| `episode_1` | `[120:270]` | 150 |
| `episode_2` | `[270:400]` | 130 |

Explicit episode subsets remain in this numeric source order. UMI replay buffers do not contain a named
mask table, so `filter_key` is unsupported.

## Inspect before planning

```bash
uv run leport inspect data/umi/cup_in_the_wild.zarr.zip \
  --adapter umi \
  --json
```

Inspection reads array metadata and the compact `episode_ends` index, not image payloads. It reports
every flat replay-buffer key with dtype, single-frame shape, episode coverage, and image candidacy.
Common selectors are:

| Selector | Stored value | Mapping note |
|---|---|---|
| `camera0_rgb` | HWC `uint8` RGB | Map to a LeRobot image or video feature |
| `robot0_eef_pos` | End-effector XYZ | Confirm coordinate frame and units externally |
| `robot0_eef_rot_axis_angle` | Axis-angle rotation | Preserve the stored representation |
| `robot0_gripper_width` | Gripper width | Confirm units and sign convention externally |
| `robot0_demo_start_pose` | Repeated demonstration start pose | Optional observation or provenance field |
| `robot0_demo_end_pose` | Repeated demonstration end pose | Optional observation or provenance field |

Additional robots and cameras are exposed through their recorded keys. LePort does not assume a fixed
number of either.

## Declare action semantics explicitly

The official UMI policy sampler constructs an action by concatenating end-effector position,
axis-angle rotation, and gripper width when no stored `action` array exists. The adapter deliberately
does not reproduce that policy behavior. Record the exact order in a ConversionPlan instead:

```yaml
schema_version: 1
adapter: umi
source: data/umi/cup_in_the_wild.zarr.zip
selection:
  episode_ids: []
  filter_key: null
target:
  repo_id: local/umi-cup-demo
  root: outputs/umi-cup-demo
  robot_type: umi-gripper
  use_videos: false
fps: 10
task:
  kind: static
  value: arrange the cups
features:
  observation.state:
    dtype: float32
    shape: [7]
  observation.images.wrist:
    dtype: image
    shape: [224, 224, 3]
  action:
    dtype: float32
    shape: [7]
mappings:
  observation.state:
    operation: concat
    sources:
      - robot0_eef_pos
      - robot0_eef_rot_axis_angle
      - robot0_gripper_width
  observation.images.wrist:
    operation: direct
    sources: [camera0_rgb]
  action:
    operation: concat
    sources:
      - robot0_eef_pos
      - robot0_eef_rot_axis_angle
      - robot0_gripper_width
```

Treat the example FPS, task, robot type, action interpretation, coordinate frames, units, and image
shape as values to confirm for the specific processed store. Inspection provides recorded structure,
not those semantics. Use the inspected camera shape rather than copying `224 x 224` when it differs.

## Convert and validate

Save the reviewed YAML as `outputs/umi-cup-demo.yaml`, then run:

```bash
uv run leport plan --check outputs/umi-cup-demo.yaml --json
uv run leport convert --config outputs/umi-cup-demo.yaml --json
uv run leport validate outputs/umi-cup-demo \
  --config outputs/umi-cup-demo.yaml --json
```

Conversion preflights the selected schemas and slices, reads only mapped frame arrays, lets Zarr decode
the current compressed chunk, and commits the target only after LeRobot reload validation succeeds. It
does not normalize, shift, resample, resize, apply horizons, compensate latency, or convert poses.

## Merge converted episodes (optional)

Convert disjoint UMI episode selections with identical FPS, robot type, task handling, and feature
schemas, then merge their completed LeRobot targets:

```bash
uv run leport merge \
  outputs/umi-cup-demo-a \
  outputs/umi-cup-demo-b \
  --target outputs/umi-cup-demo-merged \
  --repo-id local/umi-cup-demo-merged \
  --json
```

Merge accepts converted LeRobot datasets, never raw or processed UMI stores, and does not modify either
input.

The runnable [UMI notebook](../notebooks/umi.ipynb) uses an existing
`data/umi/cup_in_the_wild.zarr.zip`, creates an explicit concatenating plan, converts `episode_0`,
validates the result, and optionally converts `episode_1` before merging both targets.
