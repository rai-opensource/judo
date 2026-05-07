# Quickstart
This section walks you through the installation and usage of `judo`. For more details on the design of `judo` and creating custom tasks and optimizers, see the [Interface](../interface/intro#the-judo-interface) section.

## 1. Installation
### Using `pip`
We recommend installing `judo` using `pip` as follows:
```bash
pip install judo-rai  # if you want dev dependencies, use judo-rai[dev]
```

### Developers
#### Conda
For developers, run the following commands from this folder after cloning:
```bash
conda create -n judo python=3.13
conda activate judo
pip install -e .[dev]  # includes docs dependencies
pre-commit install
pybind11-stubgen mujoco -o typings/  # stops type checkers from complaining
```

#### Pixi
You can also use [`pixi`](https://pixi.sh/dev/) instead of `conda`, which has the added benefit of having an associated lock file that ensures complete reproducibility.

To install `pixi`, run the following:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
To create our environment (and activate it each time later), run the following in the repo root:
```bash
# every time you want to activate
pixi shell -e dev  # includes docs dependencies

# first time only
pre-commit install
pybind11-stubgen mujoco -o typings/
```

## 2. Run the `judo` app!
To start the simulator, you can simply run:
```bash
judo
```
We package `judo` with a few starter tasks and optimizers. If you want to start the simulator with one of these, you can run:
```bash
judo task=<task_name> optimizer=<optimizer_name>
```
where `task_name` is one of the following:
```
cylinder_push
cartpole
fr3_pick
leap_cube
leap_cube_down
caltech_leap_cube
```
and `optimizer_name` is one of the following:
```
cem
mppi
ps
```
This is not necessary, though, because you can use the dropdown menus to switch between tasks and optimizers after launching the app.
<p align="center">
  <img src="/judo/_static/images/task_dropdown.png" alt="task_dropdown" width="300">
  <img src="/judo/_static/images/optimizer_dropdown.png" alt="optimizer_dropdown" width="300">
</p>

You can also run the app programmatically from some other script or program.
```python
from judo.cli import app

if __name__ == "__main__":
    # do whatever you want here, like registering tasks/optimizers/overrides, etc.
    app()  # this runs the app from your own script
```

## 3. Running `judo` as a Dependency
You can easily install `judo` as a dependency in your own project. A few comments:
* You can still use the `judo` CLI command from anywhere, so long as you are working in an environment where `judo` is installed.
* If you do this, you should use the `hydra` configuration system to do things like registering custom tasks and optimizers, modifying the `dora` nodes in the sim stack, etc. See the [Configuration with hydra](../interface/config_with_hydra#configuration-with-hydra) and [Config Registration](../interface/config_registration#config-registration) sections for more details.
* You can also run the app programmatically, as shown above.

## 4. Benchmarking
To benchmark all registered tasks and optimizers, simply run
```bash
benchmark
```
This will loop through all task/optimizer pairs and check the planning time over 100 samples. The end result will be printed to the console, showing useful statistics on your system.

Note that the benchmarking program runs the default task and optimizer parameters (subject to default task-specific overrides). If you want to benchmark with different settings, please read the information below, which explains how to change defaults.
