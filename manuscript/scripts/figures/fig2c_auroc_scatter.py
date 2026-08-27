"""Fig. 2C and Fig. S2 panels: per-protein AUROC scatter, ESM-1b on the
y-axis vs. each comparator on the x-axis.

One dot per benchmark protein. The diagonal divides the plot into two
triangles — points above the diagonal are proteins where ESM-1b scores
higher than the comparator. The annotation shows the result of a paired
one-sided t-test on per-protein AUROC across the common set of proteins.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

plt.rcParams["font.size"] = 13

OUT = FIGURES / "fig2c_auroc_scatter"

# (comparator method name, x-axis label, output filename)
COMPARISONS = [
    ("Ohm",         "Network model (Ohm) AUROC",        "ESM1b_vs_Ohm.png"),
    ("EVcouplings", "EVcouplings AUROC",                "ESM1b_vs_EVcouplings.png"),
    ("ProtT5",      "ProtT5 AUROC",                     "ESM1b_vs_ProtT5.png"),
    ("Distance",    "3D-distance AUROC",                "ESM1b_vs_Distance.png"),
    ("ESMpp",       "ESM++ AUROC",                      "ESM1b_vs_ESMpp.png"),
    ("ESM2_650M",   "ESM-2 (650M) AUROC",               "ESM1b_vs_ESM2.png"),
    ("Random",      "Random baseline AUROC",            "ESM1b_vs_Random.png"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        esm1b = load_auroc_metrics("ESM1b")
    except FileNotFoundError:
        print("missing cached AUROCs for ESM1b — run scoring first")
        return

    for method, xlabel, fname in COMPARISONS:
        try:
            other = load_auroc_metrics(method)
        except FileNotFoundError:
            print(f"skip {method}: no cached AUROC")
            continue
        common = sorted(set(esm1b) & set(other))
        if len(common) < 5:
            print(f"skip {method}: only {len(common)} common proteins")
            continue
        a = np.array([esm1b[u] for u in common], dtype=float)
        b = np.array([other[u] for u in common], dtype=float)
        t, p_two = ttest_rel(a, b)
        p_gt = (p_two / 2.0) if t > 0 else (1.0 - p_two / 2.0)
        rel = ">" if p_gt < 0.05 else "≈"
        ann = f"ESM-1b {rel} {method}, p={p_gt:.3f}"

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_patch(plt.Polygon([[0, 0], [1, 0], [1, 1]], color="#f0f0f0", zorder=0))
        ax.add_patch(plt.Polygon([[0, 0], [0, 1], [1, 1]], color="#bfe5fc", zorder=0))
        ax.scatter(b, a, color="black", s=70, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=2)
        ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.6, zorder=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel); ax.set_ylabel("ESM-1b AUROC")
        ax.text(0.97, 0.05, ann, ha="right", va="bottom",
                transform=ax.transAxes, fontsize=13,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#888", alpha=0.95), zorder=3)
        plt.tight_layout()
        fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}: {ann}  (n={len(common)})")


if __name__ == "__main__":
    main()
