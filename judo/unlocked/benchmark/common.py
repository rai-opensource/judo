# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import csv
import os

SIZES = [
    64,
    512,
    4096,
    32768,
    262144,
    921600,  # 480p
    2097152,
    6220800,  # 1080p @ 24 bit
    16777216,
    24883200,  # 4K @ 24 bit
    48600000,  # 4K @ 30 bit
    134217728,
]

N_TRIALS = 100

csv_file = os.getenv("CSV_TIME_FILE")
LOG_HEADER = [
    "name",
    "size",
    "latency",
]


def record(name: str, size: int, latency: int) -> None:
    """Record latency statitics to CSV file."""
    if csv_file is None:
        return
    log_row = [name, size, 1.0e3 * latency]  # recording microseconds
    if os.path.isfile(csv_file):
        with open(csv_file, "a", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(log_row)
    else:
        with open(csv_file, "w+", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(LOG_HEADER)
            w.writerow(log_row)
