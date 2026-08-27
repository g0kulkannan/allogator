"""Fig. S6: ESM-1b attention at protein–protein interfaces.

Each line connects a protein's mean ESM-1b attention at non-interface
candidates to its mean attention at interface candidates, defined as residues
within 5 Å of any partner-chain atom. Horizontal bars show group medians.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from reproduction.paths import FIGURES

OUT = FIGURES / "figS6_interface_bias"
SRC = OUT / "interface_per_protein.csv"


def _sign_test(d: np.ndarray) -> tuple[int, int, float]:
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n = n_pos + n_neg
    if n == 0:
        return n_pos, n_neg, float("nan")
    return n_pos, n_neg, float(
        stats.binomtest(n_pos, n, alternative="two-sided").pvalue)


def _slope_plot(stat_n, stat_i) -> tuple[int, int, float]:
    n_pos, n_neg, p = _sign_test(stat_i - stat_n)
    fig, ax = plt.subplots(figsize=(6.5, 7))
    rng = np.random.RandomState(0)
    jitter = rng.normal(0, 0.018, size=len(stat_n))
    for j in range(len(stat_n)):
        d = stat_i[j] - stat_n[j]
        color = "#4c72b0" if d > 0 else ("#c44e52" if d < 0 else "#888")
        ax.plot([0 + jitter[j], 1 + jitter[j]],
                [stat_n[j], stat_i[j]],
                color=color, alpha=0.4, linewidth=1.2, zorder=1)
    ax.scatter(np.zeros_like(stat_n) + jitter, stat_n, s=70,
               color="#888", edgecolor="black", linewidth=0.7, zorder=2)
    ax.scatter(np.ones_like(stat_i) + jitter, stat_i, s=70,
               color="#4c72b0", edgecolor="black", linewidth=0.7, zorder=2)
    for xp, y, c in [(0, np.median(stat_n), "#333"),
                      (1, np.median(stat_i), "#1f4e79")]:
        ax.plot([xp - 0.2, xp + 0.2], [y, y], color=c, lw=4, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["non-interface", "interface"],
                                              fontsize=15)
    ax.tick_params(axis="y", labelsize=13); ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Per-protein mean ESM-1b score", fontsize=15)
    ax.set_ylim(-0.005, 0.075)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figureS6.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return n_pos, n_neg, p


def main() -> None:
    if not SRC.exists():
        print(
            f"missing {SRC} — run "
            "manuscript/scripts/scoring/run_interface.py first"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    if df.empty:
        print("no multimers analysed"); return
    n_pos, n_neg, p = _slope_plot(
        df["mean_non_interface"].values,
        df["mean_interface"].values,
    )
    print(
        f"mean interface > non-interface for {n_pos}/{n_pos + n_neg} proteins; "
        f"two-sided sign-test p={p:.6g}"
    )
    print(f"saved {OUT/'figureS6.png'}")


if __name__ == "__main__":
    main()
