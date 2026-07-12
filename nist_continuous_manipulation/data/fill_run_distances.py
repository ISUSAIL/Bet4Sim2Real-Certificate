"""Compute the average EOAT-to-marker distance error for every run of every
replicate and fill the 'distance' column of the By-Run-Order factorial CSV.

Reuses the single-replicate machinery in parse_log_events.py: for each
replicate it locates the run log and the OTS take CSVs, resolves every
"Searching for targetMarker" event to an EOAT position and its ground-truth
marker, then averages the per-marker L2 distance error over each run's 6
markers to get one distance per (replicate, run).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from parse_log_events import (
    annotate_with_take,
    load_marker_gt,
    load_takes,
    parse_events,
    parse_run_order,
    write_events_csv,
)

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"


def find_replicate_paths(replicate: int):
    """Return (log_path, ots_dir) for a replicate under data/raw/Replicate N."""
    rep_dir = RAW_DIR / f"Replicate {replicate}"
    logs = sorted(rep_dir.glob("Program Output/run_tests_nist_output_*.txt"))
    if not logs:
        raise FileNotFoundError(f"No run_tests_nist_output log under {rep_dir}")
    ots_dir = rep_dir / "OTS" / "CSV" / "Custom Axis Convention"
    if not ots_dir.is_dir():
        raise FileNotFoundError(f"No OTS CSV directory at {ots_dir}")
    return logs[0], ots_dir


def run_distances_for_replicate(replicate, run_side_map, marker_gt):
    """Return {run_number: average_distance_error_mm} for one replicate, and
    write that replicate's per-event CSV under its own folder."""
    log_path, ots_dir = find_replicate_paths(replicate)
    takes = load_takes(ots_dir)
    name_to_run = {t["name"]: t["run_number"] for t in takes}

    events = parse_events(log_path)
    annotate_with_take(events, takes, run_side_map=run_side_map, marker_gt=marker_gt)

    replicate_dir = log_path.parent.parent
    write_events_csv(events, replicate_dir / "events.csv")

    dists = defaultdict(list)
    for e in events:
        d = e.get("distance_error_mm")
        if d is None or e.get("run_name") is None:
            continue
        dists[name_to_run[e["run_name"]]].append(d)
    return {run: sum(v) / len(v) for run, v in dists.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_run_order = RAW_DIR / "Continuous_Mobile_Manipulator_Experiment_Run_Order_06-07-2022.xlsx"
    parser.add_argument("--run-order", type=Path, default=default_run_order)
    parser.add_argument("--fiducials-gt", type=Path, default=RAW_DIR / "rmma_fiducials_gt.csv")
    default_factorial = DATA_DIR / "Continuous_Mobile_Manipulator_Experiment_Factorial_06-07-2022_By_Run_Order.csv"
    parser.add_argument("--factorial", type=Path, default=default_factorial,
                        help="By-Run-Order factorial CSV to fill in place")
    args = parser.parse_args()

    run_order = parse_run_order(args.run_order)
    marker_gt = load_marker_gt(args.fiducials_gt)

    # (replicate, run) -> average distance error
    distances = {}
    for replicate in sorted(run_order):
        run_side_map = run_order[replicate]
        try:
            per_run = run_distances_for_replicate(replicate, run_side_map, marker_gt)
        except FileNotFoundError as exc:
            print(f"Replicate {replicate}: SKIPPED ({exc})")
            continue
        for run, avg in sorted(per_run.items()):
            distances[(replicate, run)] = avg
        print(f"Replicate {replicate}: " + "  ".join(
            f"R{run}={per_run[run]:.2f}" for run in sorted(per_run)))

    # Fill the factorial CSV, checking Side agreement as a run-order sanity check.
    with args.factorial.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if "distance" not in fieldnames:
        fieldnames.append("distance")

    filled = 0
    for r in rows:
        rep, run = int(r["Replicate"]), int(r["Run"])
        avg = distances.get((rep, run))
        if avg is None:
            r["distance"] = ""
            continue
        expected_side = run_order.get(rep, {}).get(run)
        if expected_side is not None and str(expected_side) != str(r["Side"]):
            print(f"  WARNING: Replicate {rep} Run {run} side mismatch "
                  f"(factorial={r['Side']} vs run-order={expected_side})")
        r["distance"] = avg
        filled += 1

    with args.factorial.open("w", newline="\r\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFilled 'distance' for {filled}/{len(rows)} rows -> {args.factorial}")


if __name__ == "__main__":
    main()
