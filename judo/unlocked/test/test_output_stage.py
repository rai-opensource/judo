# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import hashlib
import unittest
from unittest import mock

from judo.unlocked import Node


class TestFunctionAnotations(unittest.TestCase):
    """Test function annotations."""

    def test_annotation(self) -> None:
        def no_return() -> None:
            pass

        def return_int() -> int:
            return 0

        class My:
            """Empty test class."""

        def return_My_str() -> tuple[My, str]:
            return My(), "test"

        stage = Node("test_None", no_return).output_stage
        stage.open()
        self.assertEqual(len(stage), 0)
        with self.assertRaises(IndexError):
            stage[0]
        stage.close()

        stage = Node("test_int", return_int).output_stage
        stage.open()
        self.assertEqual(len(stage), 1)
        with self.assertRaises(IndexError):
            stage[1]
        stage.close()

        stage = Node("test_tuple", return_My_str).output_stage
        stage.open()
        self.assertEqual(len(stage), 2)
        with self.assertRaises(IndexError):
            stage[2]
        stage.close()

    def test_naming(self) -> None:
        """Queue names use a deterministic 20-char SHA-1 hex prefix + _{queue_index}.

        The hash is derived from the node name and the creation timestamp.  We
        pin the timestamp via mock to make the assertion deterministic.
        """

        def return_int() -> int:
            return 0

        fixed_time = 1_000_000_000.0
        with mock.patch("judo.unlocked.output_stage.time") as mock_time:
            mock_time.time.return_value = fixed_time
            stage = Node("test_hash_node", return_int).output_stage
            stage.open()
            try:
                expected_hash = hashlib.sha1(f"test_hash_node_{fixed_time}".encode("utf-8")).hexdigest()[:20]
                self.assertEqual(stage._output_queues[0]._name, f"{expected_hash}_0")
            finally:
                stage.close()


if __name__ == "__main__":
    unittest.main()
