# LePort

LePort converts robot datasets to LeRobot Dataset v3.

| Dataset or ecosystem | Primary source representation | Conversion status |
|---|---|:---:|
| robomimic | HDF5 | ✅ Supported |
| ALOHA / Mobile ALOHA | Per-episode HDF5 | ✅ Supported |
| ManiSkill | Paired HDF5 and JSON | ✅ Supported |
| LIBERO | HDF5 demonstrations with task metadata | ✅ Supported |
| Universal Manipulation Interface (UMI) | Zarr v2 ZipStore or directory store | ✅ Supported |

## Documentation

- Runnable examples: [robomimic notebook](notebooks/robomimic.ipynb),
  [ALOHA notebook](notebooks/aloha.ipynb), [ManiSkill notebook](notebooks/maniskill.ipynb),
  [LIBERO notebook](notebooks/libero.ipynb), and [UMI notebook](notebooks/umi.ipynb)
- Guides for agents: [robomimic documentation](docs/robomimic.md),
  [ALOHA documentation](docs/aloha.md), [ManiSkill documentation](docs/maniskill.md),
  [LIBERO documentation](docs/libero.md), and [UMI documentation](docs/umi.md)

Installation, CLI usage, and complete workflows are documented in the format guides and notebooks.
