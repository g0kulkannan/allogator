"""Fig. S7: per-protein AUROC vs. MSA depth per residue (N_eff / L).

(a) Per-protein AUROC against N_eff/L (log x) for ESM-1b, EVcouplings, and
    Ohm across the 109-protein benchmark, with a per-method LOWESS trend.
(b) Median per-protein AUROC by method on the low-depth (N_eff/L < 1)
    proteins all three methods can score, with IQR error bars and individual
    proteins overlaid.

ESM-1b is highlighted in color; EVcouplings is orange and Ohm gray.

Self-contained (reads cached AUROC JSONs + N_eff CSV; no allobench import).

Output:
    manuscript/figures/figS7_neff_per_L/figS7_neff_per_L.png
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, wilcoxon, ttest_rel, binomtest
from statsmodels.nonparametric.smoothers_lowess import lowess

from reproduction.paths import BENCHMARK, CACHED, FIGURES

OUT = FIGURES / "figS7_neff_per_L"

METHODS = [
    ("ESM1b",       "ESM-1b",      "royalblue"),
    ("EVcouplings", "EVcouplings", "#e07b00"),   # orange
    ("Ohm",         "Ohm",         "#777777"),   # gray
]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.5,
})


def load_auroc(method: str) -> dict:
    raw = json.loads((CACHED / method / "per_protein_auroc.json").read_text())
    out = {}
    for u, v in raw.items():
        val = v["AUROC"] if isinstance(v, dict) and "AUROC" in v else v
        if isinstance(val, list):
            val = val[0]
        if val is not None:
            out[u] = float(val)
    return out


def _lowess(x, y, frac=0.55):
    order = np.argsort(x)
    return x[order], lowess(
        y[order], np.log10(x[order]), frac=frac, return_sorted=False
    )


def main() -> None:
    aurocs = {m: load_auroc(m) for m, _, _ in METHODS}
    neff = pd.read_csv(BENCHMARK / "neff_colabfold_uniref.csv").dropna(subset=["neff", "length"])
    neff = neff.set_index("uniprot")
    neff["nl"] = neff["neff"] / neff["length"]

    fig = plt.figure(figsize=(8.4, 3.9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.95, 1], wspace=0.32,
                           left=0.10, right=0.985, top=0.9, bottom=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- panel a: AUROC vs N_eff/L ----
    plotted_max, plotted_min, plotted_ids = -np.inf, np.inf, set()
    for method, label, color in METHODS:
        common = sorted(set(aurocs[method]) & set(neff.index))
        plotted_ids.update(common)
        x = neff.loc[common, "nl"].to_numpy(float)
        y = np.array([aurocs[method][u] for u in common], dtype=float)
        plotted_max = max(plotted_max, x.max())
        plotted_min = min(plotted_min, x.min())
        rho, _ = spearmanr(x, y)
        ax_a.scatter(x, y, s=16, alpha=0.6, color=color, edgecolor="black",
                     linewidth=0.3, label=f"{label}  (n={len(common)}, ρ={rho:+.2f})", zorder=3)
        # LOWESS trend, drawn only over the data-dense interior: skip the first
        # and last two points, where the tails are too sparse for a meaningful
        # local fit.
        xs, ys = _lowess(x, y)
        k = 2
        if len(xs) > 2 * k:
            xs, ys = xs[k:-k], ys[k:-k]
        ax_a.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95, zorder=4)
    ax_a.set_xscale("log")
    ax_a.set_xlim(plotted_min * 0.9, plotted_max * 1.03)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlabel(r"N$_{\mathrm{eff}}$ / L  (effective sequences per residue)")
    ax_a.set_ylabel("Per-protein AUROC")
    ax_a.grid(True, which="major", linestyle=":", alpha=0.3)
    ax_a.legend(loc="lower right", framealpha=0.95)

    # ---- panel b: bars plus individual points on matched low-depth subset ----
    bench = set(aurocs["ESM1b"]) & set(neff.index)
    low = [u for u in bench if neff.loc[u, "nl"] < 1.0]
    matched = sorted(u for u in low if all(u in aurocs[m] for m, _, _ in METHODS))
    vals = {m: np.array([aurocs[m][u] for u in matched]) for m, _, _ in METHODS}
    medians = {m: float(np.median(vals[m])) for m, _, _ in METHODS}
    quartiles = {m: np.percentile(vals[m], [25, 75]) for m, _, _ in METHODS}

    point_offsets = np.linspace(-0.15, 0.15, len(matched))
    for i, (m, lab, color) in enumerate(METHODS):
        ax_b.bar(i, medians[m], width=0.66, color=color, alpha=0.72,
                 edgecolor="black", linewidth=0.8, zorder=2)
        ax_b.scatter(i + point_offsets, vals[m], s=28, color=color,
                     alpha=1.0, edgecolor="black", linewidth=0.55, zorder=3)
        q1, q3 = quartiles[m]
        ax_b.errorbar(
            i,
            medians[m],
            yerr=np.array([[medians[m] - q1], [q3 - medians[m]]]),
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )
    ax_b.set_xticks(range(len(METHODS)))
    ax_b.set_xticklabels([lab for _, lab, _ in METHODS], rotation=30, ha="right")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel(f"Median per-protein AUROC (matched n={len(matched)})")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    for ax, lab in [(ax_a, "a"), (ax_b, "b")]:
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-46, 8),
                    textcoords="offset points", fontsize=14, fontweight="bold",
                    va="bottom", ha="left")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figS7_neff_per_L.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # ---- significance (one-sided, ESM-1b greater) on the matched subset ----
    print(f"matched n={len(matched)}: {matched}")
    print("medians:", {k: round(v, 3) for k, v in medians.items()})
    print("means:  ", {m: round(float(vals[m].mean()), 3) for m, _, _ in METHODS})
    e = vals["ESM1b"]
    for m in ("EVcouplings", "Ohm"):
        o = vals[m]
        wins = int((e > o).sum())
        t_two = ttest_rel(e, o).pvalue
        t_one = t_two / 2 if e.mean() > o.mean() else 1 - t_two / 2
        w_one = wilcoxon(e, o, alternative="greater").pvalue
        s_one = binomtest(wins, len(e), 0.5, alternative="greater").pvalue
        print(f"ESM-1b vs {m}: wins {wins}/{len(e)} | paired t one-sided p={t_one:.3f} | "
              f"Wilcoxon one-sided p={w_one:.3f} | sign-test one-sided p={s_one:.3f}")
    print(f"saved {OUT / 'figS7_neff_per_L.png'}")


if __name__ == "__main__":
    main()
