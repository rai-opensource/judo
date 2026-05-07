## Optimizers
The `Optimizer` class controls how to update the nominal control spline, which determines what is executed on the system. The interface is as follows:
```python
from dataclasses import dataclass
from judo.optimizers import Optimizer, OptimizerConfig

@dataclass
class MyOptimizerConfig(OptimizerConfig):
    my_param1: float = 1.0
    my_param2: int = 2

class MyOptimizer(Optimizer[MyOptimizerConfig]):
    def __init__(self, config: MyOptimizerConfig, nu: int) -> None:
        super().__init__(config, nu)
        # rest of __init__...

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots given a nominal control input.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on the sampled controls and rewards.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
```
To summarize, the `Optimizer` determines two things: (1) given a nominal control spline, how to sample from it, and (2) given a set of sampled controls and their corresponding rewards, how to update the nominal control spline. The top-level `Controller` class handles everything else.

The `Optimizer` can also run multiple iterations, and in general, its parameters may depend on the current state of the system. Thus, our interface also includes the following:
```python
class MyOptimizer(Optimizer[MyOptimizerConfig]):
    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """An entrypoint to the optimizer before optimization.

        This is used to update optimizer parameters with new information.

        Args:
            old_times: The old times for spline interpolation right before sampling. Shape=(num_nodes,).
            new_times: The new times for spline interpolation right before sampling. Shape=(num_nodes,).
        """

    def stop_cond(self) -> bool:
        """Check if the optimization should stop aside from reaching max iters (by default, never).

        Returns:
            True if the optimization should stop, False otherwise.
        """
```
