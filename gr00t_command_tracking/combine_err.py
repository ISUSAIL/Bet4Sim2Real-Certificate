"""Combine the per-axis tracking errors in 2_err.csv into a single scalar.

err2 = 0.25 * |err_vx| + 0.25 * |err_vy| + 0.5 * |err_yaw|
"""

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "2_err.csv"
OUTPUT_CSV = DATA_DIR / "err2.csv"
WEIGHTS = {"err_vx": 0.25, "err_vy": 0.25, "err_yaw": 0.5}


def main():
    with INPUT_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing = [name for name in WEIGHTS if name not in rows[0]]
    if missing:
        raise KeyError(f"{INPUT_CSV} is missing columns: {missing}")

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "err2"])
        for row in rows:
            err2 = sum(weight * abs(float(row[name])) for name, weight in WEIGHTS.items())
            writer.writerow([f"{float(row['t']):.6f}", f"{err2:.6f}"])

    print(f"wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
