# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import unittest
from dataclasses import dataclass
from typing import Any

from judo.unlocked import Const as const


class TestConst(unittest.TestCase):
    """Test const class."""

    def test_simple_const(self) -> None:
        @dataclass
        class Simple:
            value: int

        const_simple = const(Simple(value=5))

        with self.assertRaises(RuntimeError):
            const_simple.value = -5
        self.assertEqual(const_simple.value, 5)

        simple = Simple(value=6)
        simple.value = -6
        self.assertEqual(simple.value, -6)

    def test_composed_const(self) -> None:
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            first: Inner
            second: Inner

        const_composed = const(Outer(first=Inner(value=4), second=Inner(value=2)))

        self.assertEqual(const_composed.first.value, 4)
        with self.assertRaises(RuntimeError):
            const_composed.second.value = -2
        self.assertEqual(const_composed.second.value, 2)

        composed = Outer(first=Inner(value=6), second=Inner(value=6))

        self.assertEqual(composed.first.value, 6)
        composed.second.value = -6
        self.assertEqual(composed.second.value, -6)

    def test_mutable_const(self) -> None:
        class Mutable:
            def __init__(self, value: Any):
                self.value = value

            def add_one(self) -> None:
                self.value += 1

        const_mutable = const(Mutable(value=3))

        with self.assertRaises(RuntimeError):
            const_mutable.add_one()
        self.assertEqual(const_mutable.value, 3)

        mutable = Mutable(value=7)

        mutable.add_one()
        self.assertEqual(mutable.value, 8)

    def test_comparison(self) -> None:
        const_5 = const(5)
        self.assertEqual(const_5, 5)
        self.assertNotEqual(const_5, 6)
        self.assertGreater(const_5, 4)
        self.assertLess(const_5, 6)
        self.assertGreaterEqual(const_5, 5)
        self.assertLessEqual(const_5, 5)


if __name__ == "__main__":
    unittest.main()
