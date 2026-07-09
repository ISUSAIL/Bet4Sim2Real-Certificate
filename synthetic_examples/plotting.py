"""Plotting for the Sim-to-Real betting confidence-sequence demo.

Loads the results produced by ``data_generation.py`` and renders the two
figures:

  1. Bound width over n: proposed vs. ideal and data-driven (Real_8 x Sim_30).
  2. Confidence-sequence width heatmap: every method x real set.

Run ``python data_generation.py`` first to produce ``bound_width.npz`` and
``cross_product.npz``.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_bound_width(path="bound_width.npz", save=None):
    """Figure 1: mean confidence-sequence width over n for each sizing."""
    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    widths = data["widths"]
    coverages = data["coverages"]
    T = int(data["T"])
    curves = {name: (widths[i], coverages[i]) for i, name in enumerate(names)}

    n = np.arange(1, T + 1)
    style = {
        "ideal (oracle)":    dict(color="k", ls="--"),
        "data-driven (WSR)": dict(color="0.6"),
        "proposed":          dict(color="C0", lw=2.4),
    }
    plt.figure(figsize=(7.2, 4.3))
    for name, (w, cov) in curves.items():
        plt.plot(n, w, label=f"{name}  (coverage {cov:.2f})", **style.get(name, {}))
    plt.xlabel("real draws n")
    plt.ylabel("mean confidence-sequence width")
    plt.title("Bound width: proposed vs. ideal and data-driven  (Real_8 x Sim_30)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
        print(f"saved {save}")

    # Variance-reduction summary at a few horizons.
    wsr, prop, ideal = curves["data-driven (WSR)"][0], curves["proposed"][0], curves["ideal (oracle)"][0]
    for nn in (30, 100, 200):
        closed = (wsr[nn-1] - prop[nn-1]) / (wsr[nn-1] - ideal[nn-1])
        print(f"n={nn:3d}:  WSR {wsr[nn-1]:.3f} -> proposed {prop[nn-1]:.3f} "
              f"-> ideal {ideal[nn-1]:.3f}  ({closed:.0%} closed)")


def plot_cross_product(path="cross_product.npz", save=None):
    """Figure 2: width@T / coverage heatmap for every method x real set."""
    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    width = data["width"]
    cover = data["cover"]
    set_order = list(data["set_order"])

    # rank rows best -> worst by average width relative to the oracle, and color by that ratio
    ideal = width[names.index("Ideal Kelly")]
    order = np.argsort((width / ideal).mean(axis=1))
    names = [names[i] for i in order]
    width, cover = width[order], cover[order]
    ratio = width / width[names.index("Ideal Kelly")]

    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(names) + 1.6))
    im = ax.imshow(ratio, cmap="RdYlGn_r", vmin=1.0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(set_order)), set_order)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(set_order)):
            ax.text(j, i, f"{width[i, j]:.3f}\ncov {cover[i, j]:.0%}",
                    ha="center", va="center", fontsize=8)
    ax.set_title("Confidence-sequence width  (every method x real set)")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("width / ideal  (1 = oracle)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f"saved {save}")


if __name__ == "__main__":
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)
    plot_bound_width(save=os.path.join(out_dir, "bound_width.png"))
    plot_cross_product(save=os.path.join(out_dir, "cross_product.png"))
    plt.show()
