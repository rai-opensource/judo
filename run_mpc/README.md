# `run_mpc`

Run batched MPC rollouts from a JSON config and save successful trajectories to an HDF5 dataset.

## Prerequisite

This must be run in the `mjwarp` pixi environment (requires GPU):

```bash
pixi shell -e mjwarp
```

Run commands below from the repository root (`judo/`).

## Basic usage

```bash
run_mpc --config-path run_mpc/configs/cylinder_push.json
```

By default, this writes output to `run_mpc/configs/trajectories.h5`.

## Common examples

Run 50 trajectories with 8 in parallel:

```bash
run_mpc \
  --config-path run_mpc/configs/cylinder_push.json \
  --num-trajectories 50 \
  --num-parallel 8
```

Choose a custom output file:

```bash
run_mpc \
  --config-path run_mpc/configs/spot_navigate.json \
  --dataset-output-path outputs/mpc/spot_navigate.h5
```

Enable visualization (single trajectory only):

```bash
run_mpc \
  --config-path run_mpc/configs/spot_navigate.json \
  --num-parallel 1 \
  --visualize
```

## Skip success filtering

By default, only trajectories where `task.success()` returns `True` are saved.
For tasks without a `success()` implementation, the CLI raises an error upfront.
Use `--no-require-success` to collect all trajectories regardless:

```bash
run_mpc \
  --config-path run_mpc/configs/cartpole.json \
  --no-require-success
```

## Debug NaN issues

Enable `--check-nan` to log warnings when mujoco_warp rollouts produce NaN
(common with low-friction contacts):

```bash
run_mpc \
  --config-path run_mpc/configs/cylinder_push.json \
  --check-nan
```

## Visualize trajectories

After generating trajectories, visualize them with:

```bash
python run_mpc/visualize_trajectories.py --dataset-path run_mpc/configs/trajectories.h5
```

## Without entering shell

```bash
pixi run -e mjwarp run_mpc -- --config-path run_mpc/configs/cylinder_push.json
```

## Running tests

The MPC tests (`tests/test_run_mpc.py`) require a CUDA GPU and the `dev-mjwarp` pixi
environment. They are automatically skipped on machines without a GPU.

There is no GPU runner on CI for this public repo yet, so these tests must be run manually:

```bash
pixi run -e dev-mjwarp pytest tests/test_run_mpc.py -v
```

This runs end-to-end tests for all tasks (cartpole, cylinder_push, spot_navigate,
spot_box_push, spot_tire_roll, spot_tire_upright), verifying MPC execution, trajectory
shapes, reward values, and HDF5 output.
