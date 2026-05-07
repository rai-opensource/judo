# The `judo` Interface

The design of `judo` is built around two blocks: a "system" block (here, the simulator, but could be your hardware setup), and a "controller" block, which contains a task and an optimizer.

The `SimulationNode` class is a `dora` node that steps forward a MuJoCo system as specified by a provided XML. In particular, we currently only support Lagrangian dynamics with rigid bodies and stateless actuators (technically, we support anything MuJoCo supports, but we have not rigorously tested more exotic features).

It may be possible to extend this to other simulation backends, but we do not have urgent plans to do so. If this is something you are interested in, please open an issue or a pull request.

The `Controller` class manages the relationship between the task and optimizer interally, and has an interface that looks like this:
```python
class Controller:
    def __init__(
        self,
        controller_config: ControllerConfig,
        task: Task,
        task_config: TaskConfig,
        optimizer: Optimizer,
        optimizer_config: OptimizerConfig,
    ) -> None:
```
Specifically, the `Controller` assumes an algorithm that roughly looks like this:
```python
def update_action(curr_state: np.ndarray, curr_time: float):
    """Pseudo-code for the controller action optimization loop."""
    interpolate_nominal_knots(curr_time)  # time shifts the spline knots
    for i in range(max_num_iters):
        # sample new control sequences
        candidate_knots = optimizer.sample_control_knots(nominal_knots)
        candidate_splines = make_spline(candidate_knots)
        rollout_controls = candidate_splines(query_times)

        # roll out and evaluate the candidate controls
        states, sensors = rollout(curr_state, rollout_controls)
        rewards = task.reward(states, sensors, rollout_controls, task_config, metadata)

    # update the nominal knots based on the rewards
    nominal_knots = optimizer.update_nominal_knots(candidate_knots, rewards)
```
The functions associated with the `task` and `optimizer` objects above are what must be implemented in the `judo` API.
