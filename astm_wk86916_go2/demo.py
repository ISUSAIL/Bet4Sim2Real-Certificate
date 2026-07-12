import csv
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "synthetic"
sys.path.insert(0, str(SYNTHETIC_DIR))

from method import concentration  # noqa: E402
from method import e_value  # noqa: E402
from method import p_value  # noqa: E402
from method import sim2real  # noqa: E402
from method import vincent  # noqa: E402
from method.distributions import BetaSkewed  # noqa: E402

import importlib.util  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "UnitreeGo2.csv"
CONFIDENCE = 0.95
NORMALIZATION_BOUNDS = {
    "pos_error_l2": (0.0, 0.10),
    "orientation_error_geodesic": (0.0, 0.20),
}
VINCENT_GAPS = (0.05, 0.10, 0.20)
HORIZONS = (5, 10, 20, 30)
VINCENT_SIMULATORS = {
    "pos_error_l2": BetaSkewed(2.0, 10.0),
    "orientation_error_geodesic": BetaSkewed(2.0, 4.0),
}


def load_synthetic_demo():
    spec = importlib.util.spec_from_file_location("synthetic_demo_config", SYNTHETIC_DIR / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArrayDistribution:
    def __init__(self, samples, name="robot_samples"):
        self.samples = np.asarray(samples, dtype=float).ravel()
        self._name = name

    def sample(self, n=1):
        if n > self.samples.size:
            raise ValueError("requested more samples than available")
        return self.samples[:n]

    def true_mean(self):
        return float(np.mean(self.samples))

    def true_variance(self):
        return float(np.var(self.samples))

    @property
    def name(self):
        return self._name


def read_robot_data():
    data = np.genfromtxt(INPUT_CSV, delimiter=",", names=True)
    rows = {}
    metadata = []
    for column, (lower, upper) in NORMALIZATION_BOUNDS.items():
        raw = np.asarray(data[column], dtype=float)
        normalized = np.clip((raw - lower) / (upper - lower), 0.0, 1.0)
        rows[column] = {
            "raw": raw,
            "normalized": normalized,
            "lower": lower,
            "upper": upper,
            "scale": upper - lower,
        }
        metadata.append({
            "measure": column,
            "normalization_lower": lower,
            "normalization_upper": upper,
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "raw_mean": float(np.mean(raw)),
            "normalized_mean": float(np.mean(normalized)),
            "n_samples": raw.size,
        })
    return rows, metadata


def certificate_rows(measure, samples, method, bank, certificate, scale):
    widths = np.nan_to_num(certificate[:, 1] - certificate[:, 0], nan=1.0, posinf=1.0)
    means = np.cumsum(samples) / np.arange(1, samples.size + 1)
    rows = []
    for idx, width in enumerate(widths):
        rows.append({
            "measure": measure,
            "method": method,
            "bank": bank,
            "n": idx + 1,
            "mean_norm": float(means[idx]),
            "lower_norm": float(certificate[idx, 0]),
            "upper_norm": float(certificate[idx, 1]),
            "width_norm": float(width),
            "width_original": float(width * scale),
        })
    return rows


def run_methods_for_measure(measure, info, banks, eta_by_bank):
    samples = info["normalized"]
    n_samples = samples.size
    rows = []

    for bank_name, bank in banks.items():
        refine = bank_name not in ("Sim_35", "Sim_7_biased")
        cert = sim2real.bounds_from_samples(
            samples,
            bank,
            confidence=CONFIDENCE,
            eta=eta_by_bank[bank_name],
            refine=refine,
            kappa=1.0,
        )
        rows.extend(certificate_rows(measure, samples, "sim2real", bank_name, cert, info["scale"]))

    baseline_specs = [
        ("e_value_wsr", "none", e_value.bounds_from_samples(samples, confidence=CONFIDENCE, method="wsr", refine=False)),
        ("e_value_constant_0.25", "none", e_value.bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.25,),
            refine=False,
        )),
        ("e_value_constant_0.5", "none", e_value.bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.5,),
            refine=False,
        )),
        ("hoeffding", "none", concentration.hoeffding_bounds_from_samples(samples, confidence=CONFIDENCE)),
        ("empirical_bernstein", "none", concentration.empirical_bernstein_bounds_from_samples(samples, confidence=CONFIDENCE)),
        ("t_test", "none", p_value.bounds_from_samples(samples, confidence=CONFIDENCE, method="student_t")),
        ("z_test", "none", p_value.bounds_from_samples(samples, confidence=CONFIDENCE, method="normal")),
        ("sequential_t_test", "none", p_value.bounds_from_samples(samples, confidence=CONFIDENCE, method="sequential_t")),
    ]
    for method, bank, cert in baseline_specs:
        rows.extend(certificate_rows(measure, samples, method, bank, cert, info["scale"]))

    real_distribution = ArrayDistribution(samples, name=measure)
    sim_distribution = VINCENT_SIMULATORS[measure]
    for gap in VINCENT_GAPS:
        cert = vincent.certificate(
            real_distribution,
            sim_distribution,
            seed=0,
            n_samples=n_samples,
            confidence=CONFIDENCE,
            sim2real_gap_upper=gap,
            sim2real_gap_lower=gap,
        )
        rows.extend(certificate_rows(measure, samples, "vincent", f"gap_{gap:g}", cert, info["scale"]))

    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["measure"], row["method"], row["bank"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (measure, method, bank), values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: int(row["n"]))
        item = {
            "measure": measure,
            "method": method,
            "bank": bank,
            "final_width_norm": float(values[-1]["width_norm"]),
            "mean_width_norm": float(np.mean([row["width_norm"] for row in values])),
            "final_width_original": float(values[-1]["width_original"]),
            "mean_width_original": float(np.mean([row["width_original"] for row in values])),
        }
        for horizon in HORIZONS:
            if horizon <= len(values):
                item[f"width_norm_{horizon}"] = float(values[horizon - 1]["width_norm"])
                item[f"width_original_{horizon}"] = float(values[horizon - 1]["width_original"])
        out.append(item)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    synthetic_demo = load_synthetic_demo()
    banks = synthetic_demo.sim_banks()
    eta_by_bank = synthetic_demo.SIM2REAL_ETA_BY_BANK

    data, metadata = read_robot_data()
    rows = []
    for measure, info in data.items():
        rows.extend(run_methods_for_measure(measure, info, banks, eta_by_bank))

    write_csv(DATA_DIR / "normalization_metadata.csv", metadata)
    write_csv(DATA_DIR / "certificate_widths.csv", rows)
    write_csv(DATA_DIR / "summary.csv", summarize(rows))
    print(f"saved {DATA_DIR / 'normalization_metadata.csv'}")
    print(f"saved {DATA_DIR / 'certificate_widths.csv'}")
    print(f"saved {DATA_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
