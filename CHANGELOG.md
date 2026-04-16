# v0.1.0

## Added
* Added support for parallelization using MJWarp
  * Parallel simulation backend using MJWarp
  * GPU accelerated data collection in run_mpc

## Changed
  * Added an optional GPU test since GPU runners aren't currently available on repo

## Dev
  * Updated mujoco to `>=3.5.0` to support MJWarp

# v0.0.7

## Added
* Spot quadruped robot support with locomotion policy integration (@dta-bdai, #110)
  * `SpotBase` task base class with locomotion policy command interface
  * `SpotTireUpright` task: flip a tire from flat to upright using Spot's arm
  * `mujoco_extensions` C++ pybind11 module for threaded policy rollout with ONNX inference
  * `PolicyMJSimulation` and `PolicyMJRolloutBackend` for simulation with locomotion policy
  * Spot MuJoCo XML primitives (body, legs, arm, actuators, contacts, sensors)
  * Tire object XML with collision meshes
  * Automatic mesh downloading from GitHub releases
* Added `robot_descriptions` dependency for loading mujoco_menagerie Spot meshes

## Changed
* **Switched fully to `pixi` for environment management; removed `conda` instructions from README.** `pixi` provides reproducible environments via lock file with per-Python-version environments (`py310`–`py313`)
* **Overhauled PyPI wheel building pipeline.** Wheels now include the compiled `mujoco_extensions` C++ extension with ONNX Runtime baked in via `auditwheel repair` (`scripts/repair_wheel.sh`). Per-Python-version wheels (cp310–cp313) are built, repaired, published to TestPyPI for validation, then promoted to PyPI
* **Rewrote all CI workflows to use `pixi` instead of `pip install`:**
  * `test.yml` — builds `mujoco_extensions` with build caching, downloads meshes, pre-warms `robot_descriptions` cache to avoid pytest-xdist race conditions
  * `precommit.yml` — simplified to single runner (was matrix of OS x Python)
  * `pypi_publish.yml` — full build matrix (2 OS x 4 Python), TestPyPI staging with dev version suffix, install verification, then PyPI publish

## Fixed
* Fixed viser trace visualization bug (#107)

## Dev
* Pinned mujoco to `>=3.5.0,<3.6` for API compatibility
* Bump dependency group updates (#109)
* Added `auditwheel`, `build`, `cmake`, `pybind11`, `onnxruntime` to dev dependencies
* Added mesh upload script (`scripts/upload_meshes.sh`)

# v0.0.6

## Added
* New copyrights

## Fixed
* Fixed bug that caused Viser and Mujoco to crash on visualization due to different API
* Added pycparser deps
* Fixed a new mujoco API compatibility

# v0.0.5

## Added
* Separation between middleware implementation and core functionality
* Support for different middleware implementations
* Sliders with enforced bounds to errors caused by unintended values
* Interface to create controllers from registered tasks and optimizers
* Abstract interface to implement different simulation backends
* Removed duplication of objects inside a node to ensure that all references to data point to the same object

## Fixed
* Fixed bug that would allow certain sliders to be set to arbitrary values that could crash the app
* Fixed bug that would allow NaN as a valid entry into the slider text box
* Fixed bug that caused the ControllerConfig to not be recreated when the `TaskConfig` was reset.

## Documentation
* Added specification between different slider types.
* Added guidelines on how to use the `Controller`. `Visualizer`, and `Simulation` objects

## Dev
* Modified release process so that any CODEOWNER can publish to pypi

# v0.0.4

## Added
* Added support for arm-based Macs (@johnzhang3, #87)
* Included functions for simpler model indexing for sensors and joints (@bhung-bdai, @alberthli, #76)

## Fixed
* Fixed bug after spec changes where unnamed geoms were causing `judo` to crash (@alberthli, @pculbertson, #72)

## Dev
* Bump version to 0.0.4 in pyproject.toml (@alberthli, #73)
* Cache meshes in CI to prevent CI failures when request limit is hit (@alberthli, #79)

# v0.0.3

## Added
* Action normalization (@yunhaif, #54)
    * Added three types of action normalizer: `none` (default, doing nothing), `min_max` (normalizing with actuator control range), and `running` (tracking running mean and std to normalize actions)
    * Added tests for the normalizers and their integration with the controller.
    * Users can change the action normalizer under the controller tab in the GUI.

## Documentation
* Updated the README with the arxiv link for the paper (@alberthli, #56)
* Created `docs` dependency group for even easier one-command installation of docs deps for both conda and pixi (@alberthli, #57)

## Dev
* Made `pixi-version` use `latest` instead of pinned version in all workflows (@alberthli, #59)
* Bumped Viser to 1.0.0 (@brentyi, #66)

# v0.0.2
This release contains bugfixes prior to RSS 2025.

## Added
* New `judo`-specific branding! (@slecleach, #42)

## Fixed
* Brandon's last name misspelling in citation in README.md (@pculbertson, #15)
* Fixed `max_opt_iters` not correctly being applied (@lujieyang and @tzhao-bdai, #14)
    * Added a test to check for this case (@alberthli, #16)
* Fix bug where if no `exclude_geom_substring` is passed to the model visualizer, all geoms are accidentally excluded (@alberthli, #17)
* Fix bug where when `max_opt_iters>1`, the shape of `nominal_knots` is not correct. **NEW BEHAVIOR:** in the controller, `self.nominal_knots` now has shape `(num_knots, nu)` instead of `(1, num_knots, nu)` (@yunhaif, #29).
* Update model loading so that textures appear correctly in the visualizer (@pculbertson, #32)
* Fix bug where leap cube task encountered division by 0 in axis normalization (@alberthli, #34)
* Fixed bug where changing tasks added accumulating grey lines to the GUI (@slecleach, #44)
* Fixed bug in FR3 task where the MJC distance sensors were flakily not reporting when cube was being lifted (@alberthli, #49).

## Documentation
* Changelog file to track changes in the repository (@alberthli, #43)
* Added contributor guidelines to the README (@alberthli, #43)
* Added information about the tasks in a task README (@alberthli, #43)
* Updated author order in the README citation (@pculbertson, #50)

## Dev
* Bump prefix-dev/setup-pixi from 0.8.8 to 0.8.10 (#18)
* Create workflow for manually publishing releases to PyPi (@alberthli, #33)
* Update a bunch of versions for pixi (@alberthli, #40)

# v0.0.1
Initial release!
