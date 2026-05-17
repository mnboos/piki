#!/usr/bin/env python3
"""Parse a piki profile log and print a per-stage timing summary.

Usage:
    uv run python benchmarks/parse_perf_log.py logs/profile_<ts>.log

Looks for lines of the form:
    PERF stage=<name> ms=<value>

and prints a table: stage | count | p50_ms | p95_ms | p99_ms | avg_ms
"""

import re
import sys
from collections import defaultdict


def parse(path: str) -> dict[str, list[float]]:
    pattern = re.compile(r"PERF stage=(\S+) ms=([\d.]+)")
    buckets: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                buckets[m.group(1)].append(float(m.group(2)))
    return dict(buckets)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_perf_log.py <logfile>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    buckets = parse(path)

    if not buckets:
        print(f"No PERF lines found in {path}")
        sys.exit(0)

    # Column widths
    stage_w = max(len(s) for s in buckets) + 2
    col_w = 10

    header = (
        f"{'stage':<{stage_w}}"
        f"{'count':>{col_w}}"
        f"{'p50_ms':>{col_w}}"
        f"{'p95_ms':>{col_w}}"
        f"{'p99_ms':>{col_w}}"
        f"{'avg_ms':>{col_w}}"
    )
    sep = "-" * len(header)
    print(f"\nProfile report: {path}")
    print(sep)
    print(header)
    print(sep)

    # Sort by p95 descending so the hottest stages appear first
    for stage, vals in sorted(buckets.items(), key=lambda kv: percentile(kv[1], 95), reverse=True):
        p50 = percentile(vals, 50)
        p95 = percentile(vals, 95)
        p99 = percentile(vals, 99)
        avg = sum(vals) / len(vals)
        print(
            f"{stage:<{stage_w}}"
            f"{len(vals):>{col_w}}"
            f"{p50:>{col_w}.2f}"
            f"{p95:>{col_w}.2f}"
            f"{p99:>{col_w}.2f}"
            f"{avg:>{col_w}.2f}"
        )

    print(sep)
    total = sum(len(v) for v in buckets.values())
    print(f"Total PERF lines parsed: {total}\n")


if __name__ == "__main__":
    main()
