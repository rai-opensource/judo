## Task-Specific Config Overrides
In the GUI, we may want to use different default sets of parameters in the controller/optimizer depending on the task. `judo` provides an interface to specify these overrides.

For example, we want to specify a horizon of 1.0 seconds and a zero-order spline for the cylinder push task. We also have some opinions on default values for the predictive sampling optimizer. To register these choices with the GUI, we can do the following any time before running the app:
```python
from judo.config import set_config_overrides

# for the controller
set_config_overrides(
    "cylinder_push",  # task name in the GUI
    ControllerConfig,  # class to supply overrides for
    {
        "horizon": 1.0,
        "spline_order": "zero",
    },  # override keys/values
)

# for the predictive sampling optimizer
set_config_overrides(
    "caltech_leap_cube",
    PredictiveSamplingConfig,
    {
        "num_nodes": 4,
        "num_rollouts": 32,
        "use_noise_ramp": True,
        "noise_ramp": 4.0,
        "sigma": 0.2,
    },
)
```

This override mechanism is also available non-programmatically via `hydra`. To achieve the same overrides as above, you can define your own `hydra` config file like so:
```yaml
defaults:
  - judo

task: "cylinder_push"

# example of how to use task-specific controller config overrides
controller_config_overrides:
  cylinder_push:
    horizon: 1.0
    spline_order: "zero"

optimizer_config_overrides:
  caltech_leap_cube:
    ps:
      num_nodes: 4
      num_rollouts: 32
      use_noise_ramp: true
      noise_ramp: 4.0
      sigma: 0.2
```
