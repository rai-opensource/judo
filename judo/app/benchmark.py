# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
import signal
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table

from judo.app.structs import GuiState
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


class BenchmarkerNode:
    """A node that performs benchmarking of task/optimizer pairs."""

    def __init__(self, num_samples: int = 100) -> None:
        """Initialize the benchmarker node."""
        self.num_samples = num_samples
        self.active_pair: tuple[str, str] | None = None
        self.results: dict[tuple[str, str], dict[str, list[float]]] = {}
        self.done = False

    def setup(self) -> None:
        """Enumerate all task/optimizer pairs (runs once the node's process has started)."""
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()
        self.key_pairs = [(task, opt) for task in self.available_tasks for opt in self.available_optimizers]
        print("Starting benchmarking!")

    def cycle_task_optimizer_pair(self) -> GuiState | None:
        """Moves to the next task/optimizer pair, returning the GUI state that switches the sim and controller to it."""
        if not self.key_pairs:
            self.print_results()
            self.done = True
            # Ask the runner (our parent process) for its orderly tiered shutdown, as Ctrl-C would. Stopping this node or
            # exiting here instead would unlink the shared memory of the gui_state queue while sim/controller are
            # still reading them.
            os.kill(os.getppid(), signal.SIGTERM)
            return None

        task_name, optimizer_name = self.key_pairs.pop(0)
        print(f"    Pair: {(task_name, optimizer_name)}")
        self.active_pair = (task_name, optimizer_name)
        self.results[self.active_pair] = {"plan_times": []}
        return GuiState(task=task_name, optimizer=optimizer_name)

    def print_results(self) -> None:
        """Prints the results of the benchmarking using separate tables per task, with clean rules."""
        console = Console()
        print()

        # group results by task name
        grouped = defaultdict(list)
        for (task_name, optimizer_name), data in self.results.items():
            grouped[task_name].append((optimizer_name, data))

        # print results for each task
        for task_name, optimizers in grouped.items():
            table = Table(title=f"Results for Task: [bold]{task_name}[/bold]")
            table.add_column("Optimizer")
            table.add_column("Mean ± Std", justify="center")
            table.add_column("Median (IQR)", justify="center")
            table.add_column("Min / Max", justify="center")

            for optimizer_name, data in optimizers:
                # compute statistics for plan times
                plan_times = np.array(data["plan_times"])
                mean_time = np.mean(plan_times)
                std_time = np.std(plan_times)
                median_time = np.median(plan_times)
                iqr_25 = np.percentile(plan_times, 25)
                iqr_75 = np.percentile(plan_times, 75)
                min_time = np.min(plan_times)
                max_time = np.max(plan_times)

                table.add_row(
                    optimizer_name,
                    f"{mean_time:.4f} ± {std_time:.4f}",
                    f"{median_time:.4f} ({iqr_25:.4f}, {iqr_75:.4f})",
                    f"{min_time:.4f} / {max_time:.4f}",
                )

            console.print(table)
        print("Benchmarking complete! Shutting down the graph.")

    def step(self, plan_time: float | None) -> GuiState | None:
        """Record a plan time and, once enough samples are collected, switch to the next pair.

        Returns:
            The GUI state switching to the next pair, or None (publishes nothing) while sampling the current one.
        """
        if self.done:
            return None
        if self.active_pair is None:
            return self.cycle_task_optimizer_pair()
        if plan_time is None:
            return None

        plan_times = self.results[self.active_pair]["plan_times"]
        plan_times.append(plan_time)
        if len(plan_times) == self.num_samples:
            return self.cycle_task_optimizer_pair()
        return None
