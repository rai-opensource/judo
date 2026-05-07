## Tasks
The `Task` class allows you to define custom environments and rewards. Creating a new task is easy.
```python
from dataclasses import dataclass
from judo.tasks import Task, TaskConfig

@dataclass
class MyTaskConfig(TaskConfig):
    my_param1: float = 1.0
    my_param2: int = 2

class MyTask(Task[MyTaskConfig]):
    def __init__(self, model_path: Path | str, sim_model_path: Path | str | None = None) -> None:
        super().__init__(model_path, sim_model_path=sim_model_path)
        # rest of __init__...

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: MyTaskConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Abstract reward function for task.

        Args:
            states: The rolled out states (after the initial condition).
                Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensors readings.
                Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config (passed in from the top-level controller).
            system_metadata: Any additional metadata from the system that is useful for
                computing the reward. For example, in the cube rotation task, the system
                could pass in new goal cube orientations to the controller here.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """
```
If the system is our `SimulationNode` object, then there are two copies of the `Task` in the system and the controller respectively. The `SimulationNode` is responsible for stepping the `mujoco` simulation, while the `Controller` is responsible for rolling out the task. We expose functions for modifying the task before and after each of these steps. Additionally, we also allow a task-specific optimizer warm start, which is useful for tasks that require some initial setup before the optimization loop starts. The interface for these functions is as follows:
```python
class MyTask(Task[MyTaskConfig]):
    def pre_rollout(self, curr_state: np.ndarray, config: MyTaskConfig) -> None:
        """Pre-rollout behavior for task (does nothing by default).

        Args:
            curr_state: Current state of the task. Shape=(nq + nv,).
        """

    def post_rollout(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: MyTaskConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post-rollout behavior for task (does nothing by default).

        Same inputs as in reward function.
        """

    def pre_sim_step(self) -> None:
        """Pre-simulation step behavior for task."""

    def post_sim_step(self) -> None:
        """Post-simulation step behavior for task."""

    def optimizer_warm_start(self) -> np.ndarray:
        """Returns a warm start for the optimizer."""
        return np.zeros(self.nu)  # default is zeros
```
Lastly, we also provide the `get_sim_metadata` function, which allows the task instance in the `SimulationNode` to pass metadata to the task instance in the `Controller`.
```python
class MyTask(Task[MyTaskConfig]):
    def get_sim_metadata(self) -> dict[str, Any]:
        """Get metadata from the simulation node to pass to the controller."""
        return {"my_metadata": self.my_value}
```
