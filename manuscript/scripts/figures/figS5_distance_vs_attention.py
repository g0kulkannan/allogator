"""Fig. S5: attention rank vs. 3D distance to the active site.

Pooled across the benchmark, each dot is one candidate residue. Non-
allosteric residues are gray, allosteric residues are coloured by their
within-protein attention rank percentile (plasma_r colormap, dark
purple = highest rank). The dotted horizontal line at the 90th rank
percentile illustrates that allosteric residues are enriched in the
high-attention band across the full distance range.

A second variant (``allo_per_protein.png``) plots one row per protein
with allosteric residues placed by distance and coloured by within-
protein attention rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from reproduction.paths import FIGURES

OUT = FIGURES / "figS5_distance_vs_attention"
SRC = OUT / "attention_rank_vs_distance.csv"


def _pooled(df: pd.DataFrame) -> None:
    nonallo = df[df.allo == 0]
    allo    = df[df.allo == 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(nonallo.distance_3D, nonallo.attention_rank_pct,
               s=4, c="lightgray", alpha=0.32,
               label=f"Non-allosteric (n={len(nonallo):,})")
    sc = ax.scatter(allo.distance_3D, allo.attention_rank_pct,
                     c=allo.attention_rank_pct, cmap="plasma_r",
                     vmin=0.0, vmax=1.0, s=18, alpha=0.9,
                     edgecolor="black", linewidth=0.25,
                     label=f"Allosteric (n={len(allo)})")
    ax.set_xlabel("3D distance: candidate Cα → nearest active Cα (Å)")
    ax.set_ylabel("ESM-1b attention rank percentile (within protein)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.9, color="#666", linestyle=":", alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Allosteric attention rank percentile")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT / "rank_vs_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _per_protein(allo: pd.DataFrame) -> None:
    order = sorted(allo.uniprot.unique())
    pos_of = {u: i for i, u in enumerate(order)}
    allo = allo.assign(row=allo.uniprot.map(pos_of))
    n = len(order)
    fig_h = max(6.5, 0.09 * n)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    for i in range(n):
        ax.hlines(i, 0, allo.distance_3D.max() * 1.02,
                  color="#eeeeee", linewidth=0.4, zorder=0)
    sc = ax.scatter(allo.distance_3D, allo.row,
                    c=allo.attention_rank_pct, cmap="plasma_r",
                    vmin=0.0, vmax=1.0, s=22, alpha=0.92,
                    edgecolor="black", linewidth=0.25, zorder=2)
    ax.set_xlabel("3D distance: allosteric Cα → nearest active Cα (Å)")
    ax.set_ylabel(f"Protein (n={n}, sorted alphabetically by UniProt)")
    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=5.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.invert_yaxis()
    ax.set_xlim(0, allo.distance_3D.max() * 1.02)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, pad=0.012, fraction=0.025, aspect=40)
    cbar.set_label("ESM-1b attention rank percentile\n(within protein, dark = highest)")
    plt.tight_layout()
    fig.savefig(OUT / "allo_per_protein.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SRC.exists():
        print(
            f"missing {SRC} — run "
            "manuscript/scripts/scoring/run_attention_distance.py first"
        )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    _pooled(df)
    allo = df[df.allo == 1].copy()
    if len(allo):
        _per_protein(allo)
    rho, p = spearmanr(df.distance_3D, df.attention_rank_pct)
    print(f"pooled Spearman ρ = {rho:.3f}  p = {p:.2e}  "
          f"(allo: {int(df.allo.sum())}, total: {len(df)})")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
