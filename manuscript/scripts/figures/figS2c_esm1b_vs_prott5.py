"""Fig. S2C: protein-by-protein AUROC, ESM-1b vs. ProtT5.

Reuses the same plotting style as Fig. 2C — diagonal divides the
panel, paired one-sided t-test in the corner.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

OUT = FIGURES / "figS2c_esm1b_vs_prott5"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        a_dict = load_auroc_metrics("ESM1b")
        b_dict = load_auroc_metrics("ProtT5")
    except FileNotFoundError as e:
        print(f"missing data: {e}"); return
    common = sorted(set(a_dict) & set(b_dict))
    a = np.array([a_dict[u] for u in common], dtype=float)
    b = np.array([b_dict[u] for u in common], dtype=float)
    if len(common) < 5:
        print(f"only {len(common)} proteins — skip"); return
    t, p_two = ttest_rel(a, b)
    p_gt = (p_two / 2.0) if t > 0 else (1.0 - p_two / 2.0)
    rel = ">" if p_gt < 0.05 else "≈"
    ann = f"ESM-1b {rel} ProtT5, p = {p_gt:.3f}"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Polygon([[0, 0], [1, 0], [1, 1]], color="#f0f0f0", zorder=0))
    ax.add_patch(plt.Polygon([[0, 0], [0, 1], [1, 1]], color="#bfe5fc", zorder=0))
    ax.scatter(b, a, color="black", s=70, alpha=0.85,
               edgecolor="white", linewidth=0.8, zorder=2)
    ax.plot([0, 1], [0, 1], "--", color="k", alpha=0.6, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("ProtT5 AUROC"); ax.set_ylabel("ESM-1b AUROC")
    ax.text(0.97, 0.05, ann, ha="right", va="bottom",
            transform=ax.transAxes, fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888", alpha=0.95))
    plt.tight_layout()
    fig.savefig(OUT / "esm1b_vs_prott5.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  n={len(common)}: {ann}")
    print(f"saved {OUT/'esm1b_vs_prott5.png'}")


if __name__ == "__main__":
    main()
