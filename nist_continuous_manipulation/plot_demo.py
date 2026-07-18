import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import NullFormatter


DATA_DIR = Path(__file__).resolve().parent / "data"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


SIM2REAL_COLORS = {
    "Sim_35": "#78a6cf",
    "Sim_252": "#2f79b7",
    "Sim_756": "#083d77",
    "Sim_10080": "#002f6c",
    "Sim_7_biased": "#b8d5ea",
}

CONCENTRATION_COLORS = {
    "hoeffding": "#31a354",
    "empirical_bernstein": "#006d2c",
}

PVALUE_COLORS = {
    "t_test": "#969696",
    "z_test": "#636363",
    "sequential_t_test": "#252525",
}

VINCENT_COLORS = {
    0.05: "#7a0177",
    0.10: "#ae017e",
    0.20: "#dd3497",
    0.30: "#f768a1",
}


MEASURE_LABELS = {
    "Interception_Rate": ("Interception rate", "certificate width (percentage points)"),
    "Search_Time": ("Search time", "certificate width (s)"),
    "distance": ("Distance error", "certificate width (mm)"),
}


PLOT_SCALE = {
    "Interception_Rate": 100.0,
    "Search_Time": 1.0,
    "distance": 1.0,
}


Y_TICKS = {
    "Interception_Rate": [5, 10, 20, 50, 100],
    "Search_Time": [0.5, 1, 2, 5, 10],
    "distance": [2, 5, 10, 20, 30],
}


def plain_tick(value, _):
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bank_sort_key(name):
    return (name.endswith("_biased"), int(name.split("_")[1]))


def method_order(rows):
    sim_banks = sorted({r["bank"] for r in rows if r["method"] == "sim2real"}, key=bank_sort_key)
    order = [("sim2real", bank) for bank in sim_banks]
    order.extend([
        ("e_value_wsr", "none"),
        ("e_value_constant_0.25", "none"),
        ("e_value_constant_0.5", "none"),
        ("hoeffding", "none"),
        ("empirical_bernstein", "none"),
        ("t_test", "none"),
        ("z_test", "none"),
        ("sequential_t_test", "none"),
    ])
    vincent = sorted(
        {r["bank"] for r in rows if r["method"] == "vincent"},
        key=lambda name: float(name.split("_")[-1]),
    )
    order.extend([("vincent", bank) for bank in vincent])
    return order


def label(method, bank):
    if method == "sim2real":
        return f"Proposed ({bank})"
    if method == "e_value_wsr":
        return "e-value (WSR)"
    if method == "e_value_constant_0.25":
        return r"e-value ($\lambda_t=0.25$)"
    if method == "e_value_constant_0.5":
        return r"e-value ($\lambda_t=0.5$)"
    if method == "hoeffding":
        return "Hoeffding"
    if method == "empirical_bernstein":
        return "empirical Bernstein"
    if method == "t_test":
        return "t-test"
    if method == "z_test":
        return "z-test"
    if method == "sequential_t_test":
        return "sequential t-test"
    if method == "vincent":
        return f"Vincent et al. (gap {bank.split('_')[-1]})"
    return method


def style_for(method, bank):
    if method == "sim2real":
        style = {"color": SIM2REAL_COLORS.get(bank, "#1f77b4"), "linewidth": 2.2, "alpha": 0.92}
        if bank.endswith("_biased"):
            style.update({"linestyle": "--", "alpha": 0.55})
        if bank == "Sim_10080":
            style.update({"linewidth": 2.8})
        return style
    if method.startswith("e_value"):
        if method == "e_value_wsr":
            return {"color": "#d95f02", "linewidth": 2.1, "alpha": 0.95}
        if method.endswith("0.25"):
            return {"color": "#fdb863", "linewidth": 1.6, "alpha": 0.78, "linestyle": ":"}
        return {"color": "#e08214", "linewidth": 1.8, "alpha": 0.86, "linestyle": "-."}
    if method in ("hoeffding", "empirical_bernstein"):
        return {
            "color": CONCENTRATION_COLORS[method],
            "linewidth": 1.8,
            "alpha": 0.88,
            "linestyle": "--" if method == "hoeffding" else "-.",
        }
    if method in ("t_test", "z_test", "sequential_t_test"):
        styles = {"t_test": ":", "z_test": "-.", "sequential_t_test": "--"}
        return {"color": PVALUE_COLORS[method], "linewidth": 1.9, "alpha": 0.86, "linestyle": styles[method]}
    if method == "vincent":
        return {
            "color": VINCENT_COLORS.get(float(bank.split("_")[-1]), "#b45a4a"),
            "linewidth": 1.7,
            "alpha": 0.84,
            "linestyle": "--",
        }
    return {}


def plot_width_curves():
    rows = read_csv(DATA_DIR / "certificate_widths.csv")
    order = method_order(rows)
    measures = list(dict.fromkeys(row["measure"] for row in rows))
    fig, axes = plt.subplots(1, len(measures), figsize=(7.2 * len(measures), 5.8), sharey=False)
    if len(measures) == 1:
        axes = [axes]

    legend_entries = {}
    for ax, measure in zip(axes, measures):
        plotted_widths = []
        for method, bank in order:
            series = [
                row for row in rows
                if row["measure"] == measure and row["method"] == method and row["bank"] == bank
            ]
            if not series:
                continue
            x = np.array([int(row["n"]) for row in series])
            y = np.array([float(row["width_original"]) for row in series]) * PLOT_SCALE.get(measure, 1.0)
            plotted_widths.extend(y[np.isfinite(y) & (y > 0)])
            idx = np.argsort(x)
            line, = ax.plot(
                x[idx],
                y[idx],
                marker="o",
                markersize=3.5,
                label=label(method, bank),
                **style_for(method, bank),
            )
            legend_entries.setdefault((method, bank), (line, label(method, bank)))
        title, y_label = MEASURE_LABELS.get(measure, (measure, "certificate width"))
        ax.set_title(title)
        ax.set_xlabel("samples")
        ax.set_ylabel(y_label, labelpad=10)
        ax.set_yscale("log")
        if plotted_widths:
            ymin = min(plotted_widths)
            ymax = max(plotted_widths)
            ax.set_ylim(ymin / 1.25, ymax * 1.18)
        if measure in Y_TICKS:
            ax.yaxis.set_major_locator(FixedLocator(Y_TICKS[measure]))
            ax.yaxis.set_major_formatter(FuncFormatter(plain_tick))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25)

    handles, labels = zip(*[legend_entries[key] for key in order if key in legend_entries])
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.35, top=0.88, wspace=0.27)
    save = DATA_DIR / "width_curves.png"
    fig.savefig(save, dpi=180, bbox_inches="tight")
    print(f"saved {save}")


def plot_normalized_sequences():
    metadata = {row["measure"]: row for row in read_csv(DATA_DIR / "normalization_metadata.csv")}
    data = np.genfromtxt(DATA_DIR / "measurement.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, len(metadata), figsize=(6.2 * len(metadata), 3.7), sharey=True)
    if len(metadata) == 1:
        axes = [axes]
    for ax, measure in zip(axes, metadata):
        lower = float(metadata[measure]["normalization_lower"])
        upper = float(metadata[measure]["normalization_upper"])
        raw = np.asarray(data[measure], dtype=float)
        normalized = np.clip((raw - lower) / (upper - lower), 0.0, 1.0)
        x = np.arange(1, normalized.size + 1)
        ax.plot(x, normalized, marker="o", color="#222222", linewidth=1.8)
        ax.set_title(MEASURE_LABELS.get(measure, (measure, ""))[0])
        ax.set_xlabel("samples")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("normalized measure")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.84, wspace=0.08)
    save = DATA_DIR / "normalized_sequences.png"
    fig.savefig(save, dpi=180, bbox_inches="tight")
    print(f"saved {save}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    plot_width_curves()
    plot_normalized_sequences()


if __name__ == "__main__":
    main()
