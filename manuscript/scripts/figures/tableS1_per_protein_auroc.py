"""Table S1: per-protein AUROC for Ohm, EVcouplings, and ESM-1b.

Writes ``per_protein_auroc.csv`` and a three-column PNG table. The highest
AUROC for each protein is highlighted in green.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

OUT = FIGURES / "tableS1_per_protein_auroc"
METHODS = ["Ohm", "EVcouplings", "ESM1b"]
GRAY_HEADER = "#dcdcdc"
GRAY_ROW    = "#f3f3f3"
WHITE       = "#ffffff"
GREEN_BEST  = "#bce3b8"


def _collect() -> tuple[list[str], np.ndarray]:
    metrics = {m: load_auroc_metrics(m) for m in METHODS}
    proteins = sorted(set().union(*metrics.values()))
    mat = np.full((len(proteins), len(METHODS)), np.nan)
    for i, u in enumerate(proteins):
        for j, m in enumerate(METHODS):
            if u in metrics[m]:
                mat[i, j] = metrics[m][u]
    return proteins, mat


def _table(proteins, mat) -> None:
    n = len(proteins)
    best = np.full(n, -1, dtype=int)
    for i in range(n):
        if np.all(np.isnan(mat[i])): continue
        best[i] = int(np.nanargmax(mat[i]))

    N = 3
    chunk = int(np.ceil(n / N))
    rows_per = chunk + 1
    fig_h = 0.22 * rows_per + 0.4
    fig, axes = plt.subplots(1, N, figsize=(13.5, fig_h),
                              gridspec_kw={"wspace": 0.18})
    col_x = [0.0, 1.4, 2.2, 3.2, 4.0]

    def draw(ax, row, col_idx, text, fill, weight="normal", align="center"):
        x0, x1 = col_x[col_idx], col_x[col_idx + 1]
        ax.add_patch(Rectangle((x0, row), x1 - x0, 1.0, facecolor=fill,
                                edgecolor="#bbbbbb", linewidth=0.4, zorder=1))
        if text is None or text == "": return
        if align == "center":
            ax.text((x0 + x1)/2, row + 0.5, text, ha="center", va="center",
                    fontsize=8, fontweight=weight, color="black", zorder=2)
        else:
            ax.text(x0 + 0.06, row + 0.5, text, ha="left", va="center",
                    fontsize=8, fontweight=weight, color="black", zorder=2)

    for c in range(N):
        ax = axes[c]
        ax.set_xlim(0, 4); ax.set_ylim(0, rows_per); ax.invert_yaxis()
        ax.set_aspect("auto"); ax.axis("off")
        draw(ax, 0, 0, "UniProt",     GRAY_HEADER, "bold", "left")
        for j, m in enumerate(METHODS):
            draw(ax, 0, j + 1, m, GRAY_HEADER, "bold")
        i0, i1 = c * chunk, min(n, (c + 1) * chunk)
        for local_i, i in enumerate(range(i0, i1)):
            row = local_i + 1
            base = GRAY_ROW if local_i % 2 == 0 else WHITE
            draw(ax, row, 0, proteins[i], base, "normal", "left")
            for j in range(len(METHODS)):
                v = mat[i, j]
                txt = f"{v:.3f}" if not np.isnan(v) else "—"
                fill = GREEN_BEST if j == best[i] else base
                weight = "bold" if j == best[i] else "normal"
                draw(ax, row, j + 1, txt, fill, weight)

    fig.savefig(OUT / "tableS1.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proteins, mat = _collect()
    if not proteins:
        print("no AUROCs to plot"); return
    pd.DataFrame(mat, index=proteins, columns=METHODS).to_csv(
        OUT / "per_protein_auroc.csv", index_label="UniProt")
    _table(proteins, mat)
    for j, m in enumerate(METHODS):
        col = mat[:, j]; col = col[~np.isnan(col)]
        if col.size:
            print(f"  {m:13s} n={len(col):3d} mean={col.mean():.4f} "
                  f"median={np.median(col):.4f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
