#!/usr/bin/env python3
"""Summarize the single-thread follow-up CSV, without third-party packages."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("single-thread-ordering.csv")
    groups = defaultdict(list)
    seen = set()
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = tuple(row[c] for c in ("experiment", "qos", "pairs", "passes", "block", "position"))
            if key in seen:
                raise ValueError(f"duplicate sample: {key}")
            seen.add(key)
            groups[tuple(row[c] for c in ("experiment", "qos", "pairs", "passes", "implementation"))].append(row)
    if not groups:
        raise ValueError("empty data")
    print("Values are medians of per-sample metrics; throughput unit: Mops/s.")
    print("| Experiment / QoS / pairs / passes / implementation | N | Mops/s | Range | Cycles/op | Instructions/op | P instructions % | CAS failures total |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for key, rows in groups.items():
        speeds = [float(row["million_ops_per_second"]) for row in rows]
        extra = ["—", "—", "—"]
        counted = [row for row in rows if row.get("counters_collected") == "1"]
        if counted:
            if len(counted) != len(rows):
                raise ValueError("mixed counter collection")
            operations = [2 * int(r["pairs"]) * int(r["passes"]) for r in counted]
            cycles = [int(r["cycles"]) / n for r, n in zip(counted, operations)]
            instructions = [int(r["instructions"]) / n for r, n in zip(counted, operations)]
            p_fractions = [100 * int(r["p_instructions"]) / int(r["instructions"]) for r in counted]
            extra = [f"{statistics.median(v):.2f}" for v in (cycles, instructions, p_fractions)]
        print(f"| {' / '.join(key)} | {len(rows)} | {statistics.median(speeds):.2f} | "
              f"{min(speeds):.2f}–{max(speeds):.2f} | {' | '.join(extra)} | "
              f"{sum(int(r['cas_failures']) for r in rows)} |")
    wake = [float(row["wall_seconds"]) for key, rows in groups.items()
            if key[0] == "wake_only" for row in rows]
    if wake:
        print(f"\nEmpty-work wake/finish median: {statistics.median(wake) * 1e6:.2f} us")


if __name__ == "__main__":
    main()
