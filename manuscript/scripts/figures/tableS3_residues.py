"""Supplementary Table S3: active-site and allosteric-site residues
for every benchmark protein.

Outputs both a CSV (machine-readable) and a rendered table image.
"""
from __future__ import annotations
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reproduction.io import load_benchmark, load_manifest
from reproduction.paths import FIGURES

OUT = FIGURES / "tableS3_residues"
GRAY_HEADER = "#dcdcdc"
GRAY_ROW    = "#f3f3f3"
WHITE       = "#ffffff"


def _build_csv() -> pd.DataFrame:
    benchmark = load_benchmark()
    manifest = load_manifest()
    uni_to_pdb = dict(zip(manifest.uniprot, manifest.pdb))
    rows = []
    for u in sorted(benchmark):
        a = sorted(int(r) for r in benchmark[u]["active"])
        l = sorted(int(r) for r in benchmark[u]["allo"])
        rows.append({
            "UniProt": u,
            "PDB": uni_to_pdb.get(u, ""),
            "n_active": len(a),
            "active_site_residues": ", ".join(str(r) for r in a),
            "n_allosteric": len(l),
            "allosteric_site_residues": ", ".join(str(r) for r in l),
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tableS3_residues.csv", index=False)
    return df


def _render(df: pd.DataFrame) -> None:
    widths = {"UniProt": 0.9, "PDB": 0.6, "active": 3.3, "allo": 4.4}
    total_w = sum(widths.values())
    col_x = {}; x = 0.0
    for name, w in widths.items():
        col_x[name] = (x, x + w); x += w

    chars_per_unit = 13
    def wrap(text: str, col_w: float) -> str:
        w_chars = max(8, int(col_w * chars_per_unit) - 1)
        return textwrap.fill(text, width=w_chars,
                             break_long_words=False, break_on_hyphens=False)

    wrapped = []
    for _, r in df.iterrows():
        wrapped.append({"UniProt": r.UniProt, "PDB": r.PDB,
                        "active": wrap(r.active_site_residues, widths["active"]),
                        "allo":   wrap(r.allosteric_site_residues, widths["allo"])})

    # At the rendered page scale, 0.32 data units is shorter than a 7.5-point
    # line and causes wrapped residue lists to overlap. Keep one full text line
    # per 0.58 units so the PNG remains readable when zoomed.
    LINE_H = 0.58; PAD = 0.22; base = 1.15
    row_heights = []
    for r in wrapped:
        nl = max(1, r["active"].count("\n") + 1, r["allo"].count("\n") + 1)
        row_heights.append(max(base, LINE_H * nl + PAD * 2))
    HEADER_H = 1.1
    total_h = HEADER_H + sum(row_heights)
    UNIT = 0.22
    fig, ax = plt.subplots(figsize=(total_w * UNIT * 4.2, total_h * UNIT))
    ax.set_xlim(0, total_w); ax.set_ylim(0, total_h)
    ax.invert_yaxis(); ax.set_aspect("auto"); ax.axis("off")

    def draw(y0, h, col, text, fill, weight="normal", align="left",
             fontsize=7.5):
        x0, x1 = col_x[col]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=fill,
                                edgecolor="#bbbbbb", linewidth=0.4, zorder=1))
        if text in ("", None): return
        ha = "left" if align == "left" else "center"
        tx = x0 + 0.08 if align == "left" else (x0 + x1) / 2
        ax.text(tx, y0 + h/2, text, ha=ha, va="center",
                fontsize=fontsize, fontweight=weight, color="black", zorder=2)

    y = 0.0
    draw(y, HEADER_H, "UniProt", "UniProt", GRAY_HEADER, "bold", "center", 9)
    draw(y, HEADER_H, "PDB",     "PDB",     GRAY_HEADER, "bold", "center", 9)
    draw(y, HEADER_H, "active",  "Active-site residues",     GRAY_HEADER, "bold", "center", 9)
    draw(y, HEADER_H, "allo",    "Allosteric-site residues", GRAY_HEADER, "bold", "center", 9)
    y = HEADER_H
    for i, ((_, r), wr, h) in enumerate(zip(df.iterrows(), wrapped, row_heights)):
        fill = GRAY_ROW if i % 2 == 0 else WHITE
        draw(y, h, "UniProt", wr["UniProt"], fill, "bold", "left", 8)
        draw(y, h, "PDB",     wr["PDB"],     fill, "normal", "center", 8)
        draw(y, h, "active",  wr["active"],  fill, "normal", "left", 7.5)
        draw(y, h, "allo",    wr["allo"],    fill, "normal", "left", 7.5)
        y += h
    plt.tight_layout()
    fig.savefig(OUT / "tableS3_residues.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = _build_csv()
    _render(df)
    print(f"proteins: {len(df)}")
    print(f"  active residues  total: {df.n_active.sum()}  "
          f"mean/protein={df.n_active.mean():.1f}")
    print(f"  allosteric total: {df.n_allosteric.sum()}  "
          f"mean/protein={df.n_allosteric.mean():.1f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
