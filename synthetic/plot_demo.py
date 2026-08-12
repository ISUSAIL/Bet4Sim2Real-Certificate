import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm

from demo import real_sets
from demo import sim_banks


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


LABELS = {
    "sim2real": "sim2real",
    "e_value_wsr": "e-value WSR",
    "e_value_constant_0.25": "e-value constant 0.25",
    "e_value_constant_0.5": "e-value constant 0.5",
    "hoeffding": "Hoeffding",
    "empirical_bernstein": "empirical Bernstein",
    "t_test": "t-test",
    "z_test": "z-test",
    "sequential_t_test": "sequential t-test",
    "vincent": "Vincent",
}


SIM2REAL_COLORS = {
    "Sim_35": "#78a6cf",
    "Sim_252": "#2f79b7",
    "Sim_504": "#0f5f9f",
    "Sim_756": "#083d77",
    "Sim_1260": "#0050a4",
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


SIM_MARKERS = {
    "Sim_35": "^",
    "Sim_252": "s",
    "Sim_504": "D",
    "Sim_756": "P",
    "Sim_1260": "D",
    "Sim_10080": "D",
    "Sim_7_biased": "X",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label(method, bank):
    if method == "sim2real":
        if bank == "Ideal":
            return "Proposed (ideal Kelly)"
        return f"Proposed ({bank})"
    if method == "vincent":
        if bank == "Ideal":
            return "Vincent et al. (ideal)"
        return f"Vincent et al. (gap {bank.split('_')[-1]})"
    if method == "e_value_wsr":
        return "e-value (WSR)"
    if method == "e_value_constant_0.25":
        return r"e-value ($\lambda_t=0.25$)"
    if method == "e_value_constant_0.5":
        return r"e-value ($\lambda_t=0.5$)"
    if method == "t_test":
        return "t-test"
    if method == "z_test":
        return "z-test"
    if method == "sequential_t_test":
        return "sequential t-test"
    return LABELS.get(method, method)


def method_order(rows):
    sim_banks = sorted(
        {r["bank"] for r in rows if r["method"] == "sim2real" and r["bank"] != "Ideal"},
        key=lambda x: (x.endswith("_biased"), int(x.split("_")[1])),
    )
    order = [("sim2real", bank) for bank in sim_banks]
    if any(r["method"] == "sim2real" and r["bank"] == "Ideal" for r in rows):
        order.append(("sim2real", "Ideal"))
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
    vincent_banks = sorted(
        {r["bank"] for r in rows if r["method"] == "vincent"},
        key=lambda x: (x == "Ideal", float("inf") if x == "Ideal" else float(x.split("_")[-1])),
    )
    order.extend([("vincent", bank) for bank in vincent_banks])
    return order


def real_set_order(rows):
    return sorted({row["real_set"] for row in rows}, key=lambda name: int(name.split("_")[1]))


def style_for(method, bank, highlighted_bank):
    if method == "sim2real":
        if bank == "Ideal":
            return {
                "color": "#00a6c8",
                "linewidth": 3.0,
                "alpha": 0.95,
                "linestyle": "-",
            }
        style = {
            "color": SIM2REAL_COLORS.get(bank, "#1f77b4"),
            "linewidth": 2.0,
            "alpha": 0.88,
        }
        if bank == highlighted_bank:
            style.update({"linewidth": 3.0, "alpha": 1.0})
        if bank.endswith("_biased"):
            style.update({"linestyle": "--", "alpha": 0.58, "linewidth": 2.0})
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
        styles = {
            "t_test": ":",
            "z_test": "-.",
            "sequential_t_test": "--",
        }
        return {"color": PVALUE_COLORS[method], "linewidth": 1.9, "alpha": 0.86, "linestyle": styles[method]}
    if method == "vincent":
        if bank == "Ideal":
            return {
                "color": "#4d004b",
                "linewidth": 2.3,
                "alpha": 0.86,
                "linestyle": ":",
            }
        gap = float(bank.split("_")[-1])
        return {
            "color": VINCENT_COLORS.get(gap, "#b45a4a"),
            "linewidth": 1.7,
            "alpha": 0.84,
            "linestyle": "--",
        }
    return {}


def plot_width_curves(path, save):
    rows = read_csv(path)
    order = method_order(rows)
    real_sets = real_set_order(rows)
    fig, axes_grid = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=False)
    axes = axes_grid.ravel()
    legend_ax = axes[-1]
    legend_ax.axis("off")
    highlighted_bank = [bank for method, bank in order if method == "sim2real" and not bank.endswith("_biased")][-1]
    legend_entries = {}

    for ax, real_set in zip(axes, real_sets):
        plotted_widths = []
        for method, bank in order:
            series = [r for r in rows if r["real_set"] == real_set and r["method"] == method and r["bank"] == bank]
            if not series:
                continue
            n = np.array([int(r["n"]) for r in series])
            width = np.array([float(r["mean_width"]) for r in series])
            plotted_widths.extend(width[np.isfinite(width) & (width > 0)])
            idx = np.argsort(n)
            line, = ax.plot(
                n[idx],
                width[idx],
                marker="o",
                markersize=4,
                label=label(method, bank),
                **style_for(method, bank, highlighted_bank),
            )
            line_label = label(method, bank)
            legend_entries.setdefault((method, bank), (line, line_label))

        ax.set_title(real_set)
        ax.set_xlabel("samples")
        ax.set_yscale("log")
        if plotted_widths:
            ymin = min(plotted_widths)
            ymax = max(plotted_widths)
            ax.set_ylim(ymin / 1.18, ymax * 1.18)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("mean certificate width")
    axes[2].set_ylabel("mean certificate width")
    groups = [
        ("Proposed", [(method, bank) for method, bank in order if method == "sim2real"], (0.02, 0.98)),
        ("e-values", [(method, bank) for method, bank in order if method.startswith("e_value")], (0.52, 0.98)),
        ("Concentration", [(method, bank) for method, bank in order if method in (
            "hoeffding",
            "empirical_bernstein",
        )], (0.52, 0.62)),
        ("p-values", [(method, bank) for method, bank in order if method in (
            "t_test",
            "z_test",
            "sequential_t_test",
        )], (0.52, 0.40)),
        ("Vincent et al.", [(method, bank) for method, bank in order if method == "vincent"], (0.02, 0.38)),
    ]
    for title, keys, anchor in groups:
        entries = [legend_entries[key] for key in keys if key in legend_entries]
        if not entries:
            continue
        group_handles, group_labels = zip(*entries)
        legend_ax.text(anchor[0], anchor[1], title, transform=legend_ax.transAxes, fontsize=15, fontweight="bold", va="top")
        legend = legend_ax.legend(
            group_handles,
            group_labels,
            frameon=False,
            fontsize=12,
            loc="upper left",
            bbox_to_anchor=(anchor[0], anchor[1] - 0.06),
            bbox_transform=legend_ax.transAxes,
            handlelength=2.5,
            labelspacing=0.55,
            borderaxespad=0.0,
        )
        legend_ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save, dpi=180)
    print(f"saved {save}")


def plot_coverage_curves(path, save):
    rows = read_csv(path)
    order = method_order(rows)
    real_sets = real_set_order(rows)
    horizons = sorted({int(row["n"]) for row in rows})
    row_keys = [key for key in order if any(r["method"] == key[0] and r["bank"] == key[1] for r in rows)]
    fig, axes = plt.subplots(1, len(real_sets), figsize=(4.9 * len(real_sets), 5.6), sharey=True)
    if len(real_sets) == 1:
        axes = [axes]

    norm = TwoSlopeNorm(vmin=0.70, vcenter=0.95, vmax=1.00)
    im = None

    for ax, real_set in zip(axes, real_sets):
        grid = np.full((len(row_keys), len(horizons)), np.nan)
        for row in rows:
            key = (row["method"], row["bank"])
            if row["real_set"] != real_set or key not in row_keys:
                continue
            grid[row_keys.index(key), horizons.index(int(row["n"]))] = float(row["coverage"])

        im = ax.imshow(grid, aspect="auto", cmap="RdBu", norm=norm)
        ax.set_title(real_set)
        ax.set_xlabel("samples")
        ax.set_xticks(range(len(horizons)), [str(h) for h in horizons])
        ax.set_yticks(range(len(row_keys)), [label(*key) for key in row_keys])

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                text_color = "white" if value < 0.82 or value >= 0.97 else "#222222"
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=9, color=text_color)

    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.12, top=0.90, wspace=0.08)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("coverage; centered at 95%")
    fig.savefig(save, dpi=180)
    print(f"saved {save}")


def plot_eta_ablation(path, save):
    rows = read_csv(path)
    real_sets = real_set_order(rows)
    banks = sorted(
        {row["bank"] for row in rows},
        key=lambda name: (name.endswith("_biased"), int(name.split("_")[1])),
    )
    horizons = [10, 20, 50, 100, 200]
    etas = sorted({float(row["eta"]) for row in rows})
    positive = []
    for row in rows:
        for horizon in horizons:
            value = float(row[f"width_{horizon}"])
            if value > 0:
                positive.append(value)

    fig, axes = plt.subplots(
        len(banks),
        len(real_sets),
        figsize=(4.1 * len(real_sets), 2.55 * len(banks)),
        sharex=True,
        sharey=True,
    )
    if len(banks) == 1:
        axes = np.array([axes])
    if len(real_sets) == 1:
        axes = axes[:, None]

    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    im = None
    for i, bank in enumerate(banks):
        for j, real_set in enumerate(real_sets):
            ax = axes[i, j]
            grid = np.full((len(horizons), len(etas)), np.nan)
            for row in rows:
                if row["real_set"] != real_set or row["bank"] != bank:
                    continue
                eta_idx = etas.index(float(row["eta"]))
                for h_idx, horizon in enumerate(horizons):
                    grid[h_idx, eta_idx] = float(row[f"width_{horizon}"])

            im = ax.imshow(grid, aspect="auto", cmap="viridis_r", norm=norm)
            if i == 0:
                ax.set_title(real_set, fontsize=22)
            if j == 0:
                ax.set_ylabel(f"{bank}\nsamples", fontsize=20)
            if i == len(banks) - 1:
                ax.set_xlabel(r"$\eta$", fontsize=26)
            ax.set_xticks(range(len(etas)), [f"{eta:g}" for eta in etas], rotation=45, ha="right", fontsize=16)
            ax.set_yticks(range(len(horizons)), [str(horizon) for horizon in horizons], fontsize=16)

            for h_idx in range(len(horizons)):
                for eta_idx in range(len(etas)):
                    value = grid[h_idx, eta_idx]
                    ax.text(eta_idx, h_idx, f"{value:.2f}", ha="center", va="center", fontsize=11, color="white")

    fig.subplots_adjust(left=0.095, right=0.865, bottom=0.095, top=0.94, hspace=0.16, wspace=0.08)
    cax = fig.add_axes([0.89, 0.18, 0.022, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("mean certificate width", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    fig.savefig(save, dpi=180)
    print(f"saved {save}")


def distribution_points(named_distributions):
    points = []
    for name, distribution in named_distributions:
        points.append({
            "name": name,
            "mean": float(distribution.true_mean()),
            "variance": float(distribution.true_variance()),
        })
    return points


def nearest_sim_distances(real_points, sim_points):
    if not real_points or not sim_points:
        return [], []
    real_xy = np.array([[point["mean"], point["variance"]] for point in real_points])
    sim_xy = np.array([[point["mean"], point["variance"]] for point in sim_points])
    scale = np.array([
        max(np.ptp(np.r_[real_xy[:, 0], sim_xy[:, 0]]), 1e-6),
        max(np.ptp(np.r_[real_xy[:, 1], sim_xy[:, 1]]), 1e-6),
    ])
    distances = np.linalg.norm((real_xy[:, None, :] - sim_xy[None, :, :]) / scale, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(real_points)), nearest_idx]
    return nearest_idx, nearest_dist


def plot_distribution_geometry_by_bank(save):
    real_groups = real_sets()
    banks = sim_banks()
    n_rows = len(real_groups)
    n_cols = len(banks)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.25 * n_cols, 2.95 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, (real_name, real_distributions) in enumerate(real_groups.items()):
        real_points = distribution_points(real_distributions)
        for col_idx, (bank_name, bank) in enumerate(banks.items()):
            ax = axes[row_idx, col_idx]
            sim_points = distribution_points([
                (f"{bank_name}_{idx:03d}", distribution)
                for idx, distribution in enumerate(bank)
            ])
            nearest_idx, nearest_dist = nearest_sim_distances(real_points, sim_points)

            sim_means = np.array([point["mean"] for point in sim_points])
            sim_vars = np.array([point["variance"] for point in sim_points])
            real_means = np.array([point["mean"] for point in real_points])
            real_vars = np.array([point["variance"] for point in real_points])

            ax.scatter(
                sim_means,
                sim_vars,
                s=18,
                marker=SIM_MARKERS.get(bank_name, "o"),
                color=SIM2REAL_COLORS.get(bank_name, "#1f77b4"),
                edgecolors="none",
                alpha=0.42,
                rasterized=True,
            )
            ax.scatter(
                real_means,
                real_vars,
                s=38,
                marker="o",
                facecolors="#111111",
                edgecolors="white",
                linewidths=0.55,
                alpha=0.96,
                zorder=5,
            )

            for real_point, idx in zip(real_points, nearest_idx):
                sim_point = sim_points[int(idx)]
                ax.plot(
                    [real_point["mean"], sim_point["mean"]],
                    [real_point["variance"], sim_point["variance"]],
                    color="#555555",
                    linewidth=0.45,
                    alpha=0.20,
                    zorder=1,
                )

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.006, 0.256)
            ax.grid(alpha=0.22)

            if row_idx == 0:
                ax.set_title(bank_name, fontsize=22)
            if col_idx == 0:
                ax.set_ylabel(f"{real_name}\nvariance", fontsize=21)
            if row_idx == n_rows - 1:
                ax.set_xlabel("mean", fontsize=21)
            ax.tick_params(axis="both", labelsize=17)

    real_handle = axes[0, 0].scatter([], [], s=48, marker="o", facecolors="#111111", edgecolors="white", linewidths=0.55)
    sim_handle = axes[0, 0].scatter([], [], s=34, marker="s", color="#2f79b7", edgecolors="none", alpha=0.55)
    line_handle, = axes[0, 0].plot([], [], color="#555555", linewidth=0.7, alpha=0.35)
    fig.legend(
        [real_handle, sim_handle, line_handle],
        ["Real distribution", "Simulator", "nearest simulator link"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=20,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.165, top=0.91, hspace=0.08, wspace=0.06)
    fig.savefig(save, dpi=180)
    print(f"saved {save}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    plot_distribution_geometry_by_bank(os.path.join(DATA_DIR, "distribution_geometry_by_bank.png"))
    plot_width_curves(os.path.join(DATA_DIR, "width_curves.csv"), os.path.join(DATA_DIR, "width_curves.png"))
    plot_coverage_curves(os.path.join(DATA_DIR, "coverage_curves.csv"), os.path.join(DATA_DIR, "coverage_curves.png"))
    plot_eta_ablation(os.path.join(DATA_DIR, "eta_ablation.csv"), os.path.join(DATA_DIR, "eta_ablation.png"))


if __name__ == "__main__":
    main()
