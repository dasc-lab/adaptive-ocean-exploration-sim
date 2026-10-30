#!/usr/bin/env python3
"""Validate and combine compact Monte Carlo condition outputs.

Usage:
    python revision_sims/aggregate_mc_compact_results.py SWEEP_ROOT [OUTPUT_DIR]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

STRATEGIES = {
    "transect", "ergo_nonadaptive", "ergo_adaptive",
    "bb_ipp_nonadaptive", "bb_ipp_adaptive",
    "ergo_ground_truth", "bb_ipp_ground_truth",
}
LS_VALUES = {0.2, 0.75, 1.0}
LT_VALUES = {30.0, 75.0, 120.0}
W_VALUES = {0.0, 0.5, 1.0, 1.5}
SEEDS = set(range(1234, 1264))


def read_all(root: Path, filename: str) -> tuple[list[str], list[dict[str, str]]]:
    paths = sorted(p for p in root.rglob(filename) if "aggregate" not in p.parts)
    if not paths:
        raise SystemExit(f"No {filename} files found below {root}")
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if header is None:
                header = fields
            elif fields != header:
                raise SystemExit(f"Header mismatch in {path}")
            rows.extend(reader)
    return header or [], rows


def write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) == 3 else root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    trial_header, trials = read_all(root, "trial_metrics.csv")
    keys = [(f(r, "Ls"), f(r, "Lt"), f(r, "W_Rated"), r["Strategy"], int(r["Seed"])) for r in trials]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise SystemExit(f"Duplicate compact trial rows, first entries: {duplicates[:5]}")

    expected = {
        (ls, lt, w, strategy, seed)
        for ls in LS_VALUES for lt in LT_VALUES for w in W_VALUES
        for strategy in STRATEGIES for seed in SEEDS
    }
    actual = set(keys)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise SystemExit(
            f"Incomplete sweep: {len(missing)} missing and {len(extra)} unexpected rows; "
            f"first missing={list(sorted(missing))[:3]}, first extra={list(sorted(extra))[:3]}"
        )

    summary_header, summaries = read_all(root, "summary_metrics.csv")
    plot_header, plot_rows = read_all(root, "plot_summary_data.csv")
    if len(summaries) != 3 * 3 * 4 * 7 or len(plot_rows) != 3 * 3 * 4 * 7:
        raise SystemExit(
            f"Expected 252 aggregate rows; found {len(summaries)} summary and {len(plot_rows)} plot rows"
        )

    trial_sort = lambda r: (f(r, "Ls"), f(r, "Lt"), f(r, "W_Rated"), r["Strategy"], int(r["Seed"]))
    summary_sort = lambda r: (f(r, "Ls"), f(r, "Lt"), f(r, "W_Rated"), r["Strategy"])
    write_rows(out / "trial_metrics_all.csv", trial_header, sorted(trials, key=trial_sort))
    write_rows(out / "summary_metrics_all.csv", summary_header, sorted(summaries, key=summary_sort))
    write_rows(out / "plot_summary_data_all.csv", plot_header, sorted(plot_rows, key=summary_sort))
    print(f"Validated {len(trials)} trials across 36 conditions and wrote compact aggregates to {out}")


if __name__ == "__main__":
    main()
