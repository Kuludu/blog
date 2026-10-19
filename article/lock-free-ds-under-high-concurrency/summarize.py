#!/usr/bin/env python3
"""Summarize benchmark CSV using only Python's standard library."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.csv")
    groups = defaultdict(list)
    keys = set()
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            threads = int(row["threads"])
            implementation = row["implementation"]
            key = (int(row["round"]), threads, implementation)
            if key in keys or implementation not in {"mutex", "treiber"}:
                raise ValueError(f"duplicate or invalid row: {key}")
            keys.add(key)
            groups[threads, implementation].append(row)
    if not groups:
        raise ValueError("empty results")
    rounds = {key[0] for key in keys}
    if rounds != set(range(1, max(rounds) + 1)):
        raise ValueError("non-contiguous rounds")
    if len({int(row["pairs"]) for rows in groups.values() for row in rows}) != 1:
        raise ValueError("inconsistent workload")

    def median(threads, impl, column):
        rows = groups[threads, impl]
        if len(rows) != len(rounds):
            raise ValueError(f"incomplete samples: {threads}, {impl}")
        return statistics.median(float(row[column]) for row in rows)

    counts = sorted({key[0] for key in groups})
    print(f"Samples per configuration: {len(rounds)}; values below are medians.\n")
    print("| 线程数 | Mutex 吞吐量 | CAS 吞吐量 | Mutex / CAS | CAS 失败次数 / 操作 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for n in counts:
        mutex = median(n, "mutex", "million_ops_per_second")
        cas = median(n, "treiber", "million_ops_per_second")
        failures = median(n, "treiber", "failures_per_operation")
        print(f"| {n} | {mutex:.2f} | {cas:.2f} | {mutex / cas:.2f} | {failures:.2f} |")
    print("\n| 线程数 | Mutex 耗时（ms） | CAS 耗时（ms） | Mutex CPU（s） | CAS CPU（s） |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for n in counts:
        mw = median(n, "mutex", "wall_seconds") * 1000
        cw = median(n, "treiber", "wall_seconds") * 1000
        mc = median(n, "mutex", "cpu_seconds")
        cc = median(n, "treiber", "cpu_seconds")
        print(f"| {n} | {mw:.2f} | {cw:.2f} | {mc:.3f} | {cc:.3f} |")
    print("\nThroughput range (min–max, Mops/s):")
    for n in counts:
        ranges = []
        for impl in ("mutex", "treiber"):
            values = [float(row["million_ops_per_second"]) for row in groups[n, impl]]
            ranges.append(f"{impl}={min(values):.2f}–{max(values):.2f}")
        print(f"{n}: {', '.join(ranges)}")


if __name__ == "__main__":
    main()
