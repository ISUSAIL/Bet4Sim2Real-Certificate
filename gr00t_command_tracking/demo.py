import csv
import importlib.util
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


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded by path on purpose. `synthetic/method` is a real package and this
# directory holds a same-named `method/` folder, so a plain
# `from method.suresim import ...` binds the local namespace directory first
# and every `from method import ...` above then fails to resolve.
suresim = load_module_from_path("gr00t_suresim", Path(__file__).resolve().parent / "method" / "suresim.py")


DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "vel_error_real.csv"
SIM_pair_CSV = DATA_DIR / "paired_sim_5.csv"
SIM_aug_CSV = DATA_DIR / "aug_sim_5.csv"
# (csv, column, measure) per source. The paired sim rollout carries the same
# weighted error under a different column name, so it shares the err2
# normalization window and stays directly comparable to the real sequence.
# Only the real sequence is certified; sim is normalized for reporting/plots.
SEQUENCES = {
    "real": (INPUT_CSV, "err2", "err2"),
    "sim_pair": (SIM_pair_CSV, "err_weighted", "err2"),
    "sim_aug": (SIM_aug_CSV, "err_weighted", "err2"),
}
CONFIDENCE = 0.95
NORMALIZATION_BOUNDS = {
    # err2 is the weighted sum of per-axis error magnitudes, so it is
    # non-negative and its window starts at 0. The upper end sits just above the
    # observed max (1.072) so the right tail stays unclipped.
    "err2": (0.0, 1.1),
}
VINCENT_GAPS = (0.05, 0.10, 0.20, 0.30)
# Banks from the shared synthetic bank set that this demo does not report. The
# biased bank is a deliberately misspecified stress case for the synthetic
# study; it is not part of the sim2real story for this dataset.
SKIPPED_BANKS = ("Sim_7_biased",)
# Candidate-mean grid used to bracket the certificate endpoints. The shared
# default in method/e_value.py has 0.02 spacing, which is coarser than the
# certificates this dataset reaches: when the accepted region falls between two
# nodes the search returns nothing and the width collapses to the full range.
# err2 concentrates around 0.153 normalized and its sim2real certificates get
# down to ~0.005 wide, so even 0.01 spacing loses some of the banks; 0.002
# keeps a node inside the region for every bank that has a non-empty one.
GRID = np.round(np.arange(0.0001, 1.0, 0.002), 3)
HORIZONS = (5, 10, 20, 30, 100, 300, 600)
VINCENT_SIMULATORS = {
    # Right-skewed like the data, with mean 2/13 = 0.154 matching the
    # normalized mean of err2 so the simulator is roughly well specified.
    "err2": BetaSkewed(2.0, 11.0),
}
# SureSim's prediction-powered interval. Y_gold is the real rollout, Y_gold_sim
# is the paired sim rollout (same commands, so the two line up index by index),
# and Y_sim is the augmented sim rollout used as the unlabeled pool.
SURESIM_LABELED = "sim_pair"
SURESIM_UNLABELED = "sim_aug"
SURESIM_ALPHA = 1.0 - CONFIDENCE
SURESIM_C = 0.05
# ppi_uniform shuffles its stacked sample through the global numpy RNG, so the
# certificate is re-randomized on every call. Seeding per step keeps the run
# reproducible and keeps each step's interval independent of how many steps the
# sweep happens to cover.
SURESIM_SEED = 0


def load_synthetic_demo():
    return load_module_from_path("synthetic_demo_config", SYNTHETIC_DIR / "demo.py")


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


def normalize_series(path, column, measure, source):
    raw = np.asarray(np.genfromtxt(path, delimiter=",", names=True)[column], dtype=float)
    lower, upper = NORMALIZATION_BOUNDS[measure]
    normalized = np.clip((raw - lower) / (upper - lower), 0.0, 1.0)
    info = {
        "raw": raw,
        "normalized": normalized,
        "lower": lower,
        "upper": upper,
        "scale": upper - lower,
    }
    meta = {
        "source": source,
        "measure": measure,
        "column": column,
        "normalization_lower": lower,
        "normalization_upper": upper,
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "raw_mean": float(np.mean(raw)),
        "normalized_mean": float(np.mean(normalized)),
        "n_samples": raw.size,
    }
    return info, meta


def read_robot_data():
    rows = {}
    sim_rows = {}
    metadata = []
    for source, (path, column, measure) in SEQUENCES.items():
        info, meta = normalize_series(path, column, measure, source)
        metadata.append(meta)
        if source == "real":
            rows[measure] = info
        else:
            sim_rows.setdefault(measure, {})[source] = info
    return rows, sim_rows, metadata


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


def suresim_certificate(y_gold, y_gold_sim, y_sim):
    """Prediction-powered certificate path over a growing real rollout.

    Row n - 1 is the interval after n real samples, matching the convention the
    other methods use here. The labeled pair grows together (y_gold[:n] against
    y_gold_sim[:n]) because the rectifier y_gold - y_gold_sim is only defined on
    matched commands, while the unlabeled pool y_sim is used in full at every
    step.
    """
    certificate = np.empty((y_gold.size, 2), dtype=float)
    for n in range(1, y_gold.size + 1):
        np.random.seed(SURESIM_SEED + n)
        lower, upper = suresim.ppi_uniform(
            y_gold[:n],
            y_gold_sim[:n],
            y_sim,
            alpha=SURESIM_ALPHA,
            c=SURESIM_C,
        )
        certificate[n - 1] = (lower, upper)
    # The estimand is the mean of a [0, 1]-normalized error, and every other
    # method here is already confined to that support by its candidate grid.
    # PPI is not: its rectified sample lives on [-(1 + N/n), 2 + N/n], which at
    # n = 1 is ~1400 wide, so intersecting with [0, 1] puts it on equal footing
    # instead of letting the early steps dominate the axis.
    return np.clip(certificate, 0.0, 1.0)


def run_methods_for_measure(measure, info, sim_info, banks, eta_by_bank):
    samples = info["normalized"]
    n_samples = samples.size
    rows = []

    for bank_name, bank in banks.items():
        cert = sim2real.bounds_from_samples(
            samples,
            bank,
            grid=GRID,
            confidence=CONFIDENCE,
            eta=eta_by_bank[bank_name],
            refine=True,
            tol=1e-5,
        )
        rows.extend(certificate_rows(measure, samples, "sim2real", bank_name, cert, info["scale"]))

    baseline_specs = [
        ("e_value_wsr", "none", e_value.bounds_from_samples(
            samples,
            grid=GRID,
            confidence=CONFIDENCE,
            method="wsr",
            refine=True,
            tol=1e-5,
        )),
        ("e_value_constant_0.25", "none", e_value.bounds_from_samples(
            samples,
            grid=GRID,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.25,),
            refine=True,
            tol=1e-5,
        )),
        ("e_value_constant_0.5", "none", e_value.bounds_from_samples(
            samples,
            grid=GRID,
            confidence=CONFIDENCE,
            method="constant",
            constant_lambdas=(0.5,),
            refine=True,
            tol=1e-5,
        )),
        ("hoeffding", "none", concentration.hoeffding_bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
        )),
        ("empirical_bernstein", "none", concentration.empirical_bernstein_bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
        )),
        ("t_test", "none", p_value.bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
            method="student_t",
        )),
        ("z_test", "none", p_value.bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
            method="normal",
        )),
        ("sequential_t_test", "none", p_value.bounds_from_samples(
            samples,
            confidence=CONFIDENCE,
            method="sequential_t",
        )),
    ]
    for method, bank, cert in baseline_specs:
        rows.extend(certificate_rows(measure, samples, method, bank, cert, info["scale"]))

    labeled = sim_info.get(SURESIM_LABELED)
    unlabeled = sim_info.get(SURESIM_UNLABELED)
    if labeled is not None and unlabeled is not None:
        cert = suresim_certificate(
            samples,
            labeled["normalized"][:n_samples],
            unlabeled["normalized"],
        )
        rows.extend(certificate_rows(measure, samples, "suresim", SURESIM_UNLABELED, cert, info["scale"]))

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
    banks = {
        name: bank
        for name, bank in synthetic_demo.sim_banks().items()
        if name not in SKIPPED_BANKS
    }
    eta_by_bank = synthetic_demo.SIM2REAL_ETA_BY_BANK

    data, sim_data, metadata = read_robot_data()
    rows = []
    for measure, info in data.items():
        rows.extend(
            run_methods_for_measure(measure, info, sim_data.get(measure, {}), banks, eta_by_bank)
        )

    write_csv(DATA_DIR / "normalization_metadata.csv", metadata)
    write_csv(DATA_DIR / "certificate_widths.csv", rows)
    write_csv(DATA_DIR / "summary.csv", summarize(rows))
    print(f"input  {INPUT_CSV}")
    print(f"saved {DATA_DIR / 'normalization_metadata.csv'}")
    print(f"saved {DATA_DIR / 'certificate_widths.csv'}")
    print(f"saved {DATA_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
