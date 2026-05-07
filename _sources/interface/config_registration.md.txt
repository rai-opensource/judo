## Config Registration
After defining a new task and/or optimizer, you need to register it with `judo` so that it can be used in the GUI and show up in the dropdown menus. This can be done by running the following in a script:
```python
from judo.cli import app
from judo.optimizers import register_optimizer
from judo.tasks import register_task

if __name__ == "__main__":
    register_optimizer("my_optimizer_name", MyOptimizer, MyOptimizerConfig)
    register_task("my_task_name", MyTask, MyTaskConfig)
    app()
```

If you instead want to use the `judo` CLI command to start the app, you can register the task and optimizer using a `hydra` config. We provide a convenient interface to do so. Consider this example:
```yaml
defaults:
  - judo  # you must have this default!

task: "my_cylinder_push"
optimizer: "my_cem"

# these custom tasks/optimizers are defined in the judo/examples folder
# they are nearly identical to the regular cylinder push task and cem optimizer,
# but we add an extra parameter to each of their configs for demonstration
custom_tasks:
  my_cylinder_push:
    task: judo.examples.example_task.MyCylinderPush
    config: judo.examples.example_task.MyCylinderPushConfig
  # we can keep passing in more custom tasks here...

custom_optimizers:
  my_cem:
    optimizer: judo.examples.example_optimizer.MyCrossEntropyMethod
    config: judo.examples.example_optimizer.MyCrossEntropyMethodConfig
  # we can keep passing in more custom optimizers here...
```

Similarly, you can run the benchmarking program on custom registered tasks/configs in a similar way, replacing `app` with `benchmark` either programmatically or in the `hydra` YAML file.
