# Convert LIBERO demonstrations to LeRobot

LePort reads official LIBERO task HDF5 files through the `libero` source adapter. The reader preserves
recorded transitions, numeric arrays, and camera pixels. It does not import LIBERO, robosuite, MuJoCo,
or a simulator runtime, and it does not infer policy semantics.

## Install the reader

```bash
uv sync --extra libero --group dev
```

The lightweight extra contains only `h5py` beyond the project core.

## Official source layout

Pass either one task file or a flat suite directory. Only direct `*_demo.hdf5` children participate;
nested directories and unrelated files are ignored.

```text
data/libero/libero_90/
└── KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5
```

Each file must contain a root `data` group, canonical `demo_<integer>` groups, frame-addressable
`actions`, JSON `problem_info` with a non-empty `language_instruction`, and BDDL identity metadata.
For a file named `close_the_drawer_demo.hdf5`, public IDs are `close_the_drawer/demo_0`,
`close_the_drawer/demo_1`, and so on. Files are ordered lexically and demos numerically.

The adapter rejects conflicting `num_demos`, `total`, or per-demo `num_samples` declarations. Standard
LIBERO suites have no cross-task mask table, so `filter_key` is unsupported; use qualified episode IDs.

## Inspect before planning

```bash
uv run leport inspect \
  data/libero/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5 \
  --adapter libero \
  --episode KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet/demo_0 \
  --json
```

Inspection reports source dtype, single-frame shape, episode coverage, and image candidacy for root
datasets and nested `obs` leaves. Common selectors include:

| Selector | Typical role | Mapping note |
|---|---|---|
| `actions` | Recorded control | Interpret the controller and units from the source setup |
| `states` | Full MuJoCo state | Shape can vary between tasks; avoid it for a shared suite schema |
| `robot_states` | Robot state vector | Inspect shape and ordering before use |
| `obs/ee_states` | End-effector observation | A common task-invariant state component |
| `obs/gripper_states` | Gripper observation | Append explicitly after `ee_states` when desired |
| `obs/joint_states` | Joint observation | Confirm joint order from LIBERO metadata or code |
| `obs/agentview_rgb` | Workspace RGB | HWC `uint8`, preserved exactly by the reader |
| `obs/eye_in_hand_rgb` | Wrist RGB | HWC `uint8`, preserved exactly by the reader |

For multi-task conversion, prefer fields whose shapes are identical across selected tasks. The ordered
state list `obs/ee_states`, then `obs/gripper_states`, produces one concatenated target state with the
same order. Mapping `states` across a suite fails preflight when task-dependent object state changes its
shape.

Inspection also exposes `problem_info`, BDDL identity, `env_args`, `macros_image_convention`, source
filename, and task name. A reported `control_freq` remains source evidence: plan creation still requires
an explicit FPS. LePort never infers FPS, robot type, action meaning, units, coordinate frames, or suite
curriculum order.

## Plan, convert, and validate

The official raw demonstrations used here report a 20 Hz control frequency, so this example supplies
`--fps 20` explicitly. Confirm that value for every source you convert.

```bash
uv run leport plan \
  --source data/libero/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5 \
  --output outputs/libero-close-drawer-demo.yaml \
  --adapter libero \
  --episode KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet/demo_0 \
  --target outputs/libero-close-drawer-demo \
  --repo-id local/libero-close-drawer-demo \
  --fps 20 \
  --task-metadata instruction \
  --action actions --action-dtype float32 \
  --state obs/ee_states \
  --state obs/gripper_states \
  --state-dtype float32 \
  --image obs/agentview_rgb=observation.images.workspace \
  --image obs/eye_in_hand_rgb=observation.images.wrist \
  --no-videos

uv run leport plan --check outputs/libero-close-drawer-demo.yaml --json
uv run leport convert --config outputs/libero-close-drawer-demo.yaml --json
uv run leport validate outputs/libero-close-drawer-demo \
  --config outputs/libero-close-drawer-demo.yaml --json
```

`--task-metadata instruction` assigns every frame the natural-language instruction parsed from its task
file. State selector order defines target concatenation order. Action and state casts are explicit;
camera dimensions, channel order, orientation, and temporal sampling are unchanged. Omit `--no-videos`
to encode the two visual features as videos instead of images.

Conversion preflights all selected schemas and lengths, writes to a staging directory, reloads the
completed LeRobot Dataset v3 target, and commits atomically. It never truncates, pads, shifts, resamples,
rotates, flips, or resizes source data.

## Suite conversion and merge boundary

A flat directory can be converted as one dataset when all mapped feature schemas are compatible. Its
metadata-derived instructions remain distinct tasks. Alternatively, convert disjoint qualified demo
subsets into separate compatible LeRobot targets and merge those completed targets:

```bash
uv run leport merge \
  outputs/libero-close-drawer-demo \
  outputs/libero-close-drawer-demo-b \
  --target outputs/libero-close-drawer-demo-merged \
  --repo-id local/libero-close-drawer-demo-merged \
  --json
```

Merge accepts LeRobot dataset directories, never raw LIBERO HDF5 files. Inputs must agree on FPS, robot
type, and feature schemas, and the new target does not modify either source.

## Raw fidelity and separately hosted datasets

Some separately published LeRobot LIBERO datasets use 10 Hz data or rotate images during their own
preprocessing. Those artifacts are useful but are not evidence that raw official HDF5 transitions have
the same cadence or pixel orientation. This adapter preserves every raw action-aligned transition and
returns camera arrays exactly as stored. Any downsampling or image transform must be an explicit
external preprocessing decision.

The runnable [LIBERO notebook](../notebooks/libero.ipynb) uses the downloaded official task file,
converts `demo_0`, validates the result, and optionally converts disjoint `demo_1` before merging the two
completed targets.
