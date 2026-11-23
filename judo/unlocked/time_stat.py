# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from contextlib import nullcontext
from ctypes import c_double, c_uint64
from math import floor as _floor
from math import log2 as _log2
from math import log10 as _log10
from math import sqrt
from multiprocessing.sharedctypes import RawValue
from time import time
from typing import Any

from judo.unlocked.ipc import Mutex

_prefix = ["q", "r", "y", "z", "a", "f", "p", "n", "μ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]
_normalizer = [10 ** (-3 * i) for i in range(-10, 11)]


def format_prefix(value: float, unit: str = "") -> str:
    sign = -1 if value < 0 else 1
    value = abs(value)
    index = (0 if value == 0 else int(_floor(_log10(value) / 3.0))) + 10
    prefix = _prefix[index]
    value = sign * 0.01 * _floor(100 * value * _normalizer[index])
    return f"{(value):7.2f} {prefix + unit:2s}"


_binary_prefix = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi", "Ri", "Qi"]
_binary_normalizer = [1024**i for i in range(11)]


def format_binary_prefix(value: int, unit: str = "") -> str:
    sign = -1 if value < 0 else 1
    value = abs(value)
    index = 0 if value == 0 else int(_floor(_log2(value) * 0.1))
    prefix = _binary_prefix[index]
    if value % _binary_normalizer[index] == 0:
        value = sign * value // _binary_normalizer[index]
        return f"{(value):7d} {prefix + unit:3s}"
    value = sign * 0.1 * _floor(10 * value / _binary_normalizer[index])
    return f"{(value):7.1f} {prefix + unit:3s}"


class SharedTimeStat:
    """Accumulate time statistics across multiple processes."""

    def __init__(self, name: str, *, lock: bool = True):
        self._name = name
        self._mutex: Mutex | nullcontext = Mutex(name + "_mutex") if lock else nullcontext()
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)
        self._tic = RawValue(c_uint64, 0)

    def __enter__(self) -> "SharedTimeStat":
        self.tic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.toc()

    def __str__(self) -> str:
        with self._mutex:
            n = self._n.value
            if n == 0:
                return "================================= No Data ================================="
            return (
                f"n = {n}, "
                f"min = {format_prefix(self._min.value * 1.0e-9, 's')}, "
                f"max = {format_prefix(self._max.value * 1.0e-9, 's')}, "
                f"avg = {format_prefix(self._mu.value * 1.0e-9, 's')}, "
                f"std = {format_prefix(sqrt(self._s2.value / n) * 1.0e-9, 's')}"
            )

    def clear(self) -> None:
        """Clear accumulated statistics."""
        with self._mutex:
            self._n.value = 0
            self._min.value = -1
            self._max.value = 0
            self._mu.value = 0
            self._s2.value = 0
            self._tic.value = 0

    def tic(self, now: int | None = None) -> int:
        """Start time tracking."""
        now = now or int(time() * 1.0e9)
        with self._mutex:
            self._tic.value = now
        return now

    def toc(self, now: int | None = None) -> int:
        """Stop time tracking and accumulated staticstics."""
        now = now or int(time() * 1.0e9)
        with self._mutex:
            self._n.value += 1
            x = now - self._tic.value
            self._min.value = min(self._min.value, x)
            self._max.value = max(self._max.value, x)
            delta = x - self._mu.value
            self._mu.value = self._mu.value + delta / self._n.value
            delta2 = x - self._mu.value
            self._s2.value = self._s2.value + delta * delta2
        return x

    @property
    def n(self) -> int:
        """Number of records."""
        with self._mutex:
            return self._n.value

    @property
    def mu(self) -> float:
        """Avarage time in seconds."""
        with self._mutex:
            return self._mu.value * 1.0e-9

    @property
    def sigma(self) -> float:
        """Time standard deviation in seconds."""
        with self._mutex:
            n = self._n.value
            if n == 0:
                return float("nan")
            return sqrt(self._s2.value / self._n.value) * 1.0e-9

    @property
    def min(self) -> float:
        """Minimum time in seconds."""
        with self._mutex:
            return self._min.value * 1.0e-9

    @property
    def max(self) -> float:
        """Maximum time in seconds."""
        with self._mutex:
            return self._max.value * 1.0e-9
