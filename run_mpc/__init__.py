# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# Importing the judo application layer registers judo's built-in tasks (and their backends /
# locomotion policies) in the task registry, which run_mpc relies on when loading configs.
import judo.app  # noqa: F401
