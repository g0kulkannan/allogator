"""Fig. S2D: attention-summarisation grid.

Compares the nine evaluated strategies: {mean, max, min, median} ×
{sum, max}, plus max-sym × sum. Canonical (mean × sum) is highlighted in
royal blue.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

plt.rcParams["font.size"] = 11

OUT = FIGURES / "figS2d_attn_gamma_grid"

# (method dir name, axis label)
VARIANTS = [
    ("ESM1b",             "mean / sum"),
    ("ESM1b_mean_max",    "mean / max"),
    ("ESM1b_max_sum",     "max / sum"),
    ("ESM1b_max_max",     "max / max"),
    ("ESM1b_min_sum",     "min / sum"),
    ("ESM1b_min_max",     "min / max"),
    ("ESM1b_median_sum",  "median / sum"),
    ("ESM1b_median_max",  "median / max"),
    ("ESM1b_max_sym_sum", "max-sym / sum"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aurocs = {}
    for method, _ in VARIANTS:
        try:
            aurocs[method] = list(load_auroc_metrics(method).values())
        except FileNotFoundError:
            aurocs[method] = []
            print(f"  (skip {method}: no cached AUROC)")

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    medians = []
    for i, (method, _) in enumerate(VARIANTS):
        v = aurocs[method]
        if not v:
            medians.append(np.nan); continue
        x = [i + rng.uniform(-0.18, 0.18) for _ in v]
        is_canon = method == "ESM1b"
        ax.scatter(x, v,
                   s=12 if is_canon else 10,
                   alpha=0.85 if is_canon else 0.55,
                   color="royalblue" if is_canon else "#bbbbbb",
                   edgecolor="black" if is_canon else "none",
                   linewidth=0.3 if is_canon else 0,
                   zorder=3 if is_canon else 2)
        medians.append(float(np.median(v)))
    for i, m in enumerate(medians):
        if np.isnan(m): continue
        ax.hlines(m, i - 0.28, i + 0.28, colors="black", linewidth=1.8, zorder=4)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4)
    ax.set_xlim(-0.5, len(VARIANTS) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([lbl for _, lbl in VARIANTS], rotation=90)
    ax.set_ylabel("AUROC")
    plt.tight_layout()
    fig.savefig(OUT / "attn_gamma_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for (method, lbl), m in zip(VARIANTS, medians):
        if aurocs[method]:
            print(f"  {lbl:18s}  n={len(aurocs[method]):3d}  median={m:.4f}")
    print(f"saved {OUT/'attn_gamma_grid.png'}")


if __name__ == "__main__":
    main()
