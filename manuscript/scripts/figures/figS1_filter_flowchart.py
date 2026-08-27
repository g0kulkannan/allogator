"""Fig. S1: 109-protein benchmark filter-chain flowchart.

The protein counts at each step summarize construction of the benchmark.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from reproduction.paths import FIGURES

OUT = FIGURES / "figS1_filter_flowchart"

# (description, count_entering_step, count_dropped_at_step)
STEPS = [
    ("AlloBench.csv\n(per-modulator rows, SIFTS-mapped to UniProt residues)",
     427, None),
    ("Build allosteric labels: union of per-modulator PDB-contact residues\n"
     "on the target UniProt chain only;\n"
     "drop UniProts left with no allosteric residues",
     427, 20),
    ("Sequence length ≤ 1022 aa\n(ESM-1b positional-embedding cap)",
     407, 53),
    ("Active = M-CSA catalytic residues ∪ UniProt FT 'Active site';\n"
     "drop UniProts with no catalytic annotation",
     354, 178),
    ("Drop UniProts whose allosteric modulator overlaps the substrate site\n"
     "(AlloBench `site_overlap = Yes` for any row)",
     176, 40),
    ("Manual removal of 12 curator-flagged UniProts\n"
     "(missing modulators, mis-classified substrates, isoform duplicates)",
     136, 12),
    ("Gene-name dedup: per gene, keep the UniProt with the most allosteric\n"
     "residues (tiebreak longer sequence, then alphabetical UniProt)",
     124, 12),
    ("Sequence-similarity dedup: MMseqs2 cluster at ≥ 70 % identity,\n"
     "≥ 80 % bidirectional coverage; keep most-allosteric representative",
     112, 3),
    ("Final benchmark set", 109, None),
]


def _validate_counts() -> None:
    """Check that the displayed step counts are internally consistent."""
    remaining = STEPS[0][1]
    for _label, entering, dropped in STEPS[1:-1]:
        assert entering == remaining
        assert dropped is not None
        remaining -= dropped
    assert remaining == STEPS[-1][1]

BOX_W = 6.0
BOX_H = 1.05
LEFT_CX = 4.3
RIGHT_CX = 11.7
N_LEFT = 5
LEFT_TOP = 9.0
ROW_PITCH = 1.65


def _col_xy(i):
    if i < N_LEFT:
        return (LEFT_CX, LEFT_TOP - i * ROW_PITCH)
    j = i - N_LEFT
    return (RIGHT_CX, LEFT_TOP - j * ROW_PITCH)


def _edge_label(ax, x, y, n_in, n_drop):
    if n_drop is not None:
        text = f"{n_in} → drop {n_drop} → {n_in - n_drop}"
    else:
        text = f"n = {n_in}"
    ax.text(x, y, text, ha="left", va="center", fontsize=9,
            color="#0c1e4a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#bbb", alpha=0.94), zorder=5)


def main() -> None:
    _validate_counts()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 16); ax.set_ylim(1.5, 10.1); ax.set_axis_off()

    box_centers = []
    for i, (label, n_in, _n_drop) in enumerate(STEPS):
        cx, cy = _col_xy(i)
        is_start = i == 0
        is_final = i == len(STEPS) - 1
        fc = "#2563eb" if is_final else ("#1e40af" if is_start else "#dbeafe")
        ec = "#1e40af"
        tc = "white" if is_start or is_final else "#0c1e4a"
        box = FancyBboxPatch((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                              boxstyle="round,pad=0.04,rounding_size=0.12",
                              facecolor=fc, edgecolor=ec, linewidth=1.6,
                              zorder=3)
        ax.add_patch(box)
        if is_start or is_final:
            ax.text(cx, cy + 0.18, label, ha="center", va="center",
                    fontsize=10.5, color=tc, fontweight="bold", zorder=4)
            ax.text(cx, cy - 0.28, f"n = {n_in}", ha="center", va="center",
                    fontsize=12, color=tc, fontweight="bold", zorder=4)
        else:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=9, color=tc, zorder=4)
        box_centers.append((cx, cy))

    for i in range(len(STEPS) - 1):
        src_cx, src_cy = box_centers[i]
        dst_cx, dst_cy = box_centers[i + 1]
        if i == N_LEFT - 1:
            col_gap_x = (LEFT_CX + RIGHT_CX) / 2
            src_right_x = src_cx + BOX_W/2 + 0.05
            dst_left_x  = dst_cx - BOX_W/2 - 0.05
            ax.plot([src_right_x, col_gap_x], [src_cy, src_cy],
                    color="#0c1e4a", linewidth=2.0, zorder=2)
            ax.plot([col_gap_x, col_gap_x], [src_cy, dst_cy],
                    color="#0c1e4a", linewidth=2.0, zorder=2)
            ax.add_patch(FancyArrowPatch(
                (col_gap_x, dst_cy), (dst_left_x, dst_cy),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2.0, color="#0c1e4a", zorder=2))
            _edge_label(ax, src_right_x + 0.10, src_cy + 0.30,
                        STEPS[i + 1][1], STEPS[i + 1][2])
        else:
            ax.add_patch(FancyArrowPatch(
                (src_cx, src_cy - BOX_H/2 - 0.02),
                (dst_cx, dst_cy + BOX_H/2 + 0.02),
                arrowstyle="-|>", mutation_scale=18,
                linewidth=2.0, color="#0c1e4a", zorder=2))
            mid_y = (src_cy + dst_cy) / 2
            _edge_label(ax, src_cx + 0.15, mid_y,
                        STEPS[i + 1][1], STEPS[i + 1][2])

    ax.text(8.0, 9.85, "Benchmark filter chain (drop steps only)",
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#0c1e4a")
    plt.tight_layout()
    fig.savefig(OUT / "filter_flowchart.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT/'filter_flowchart.png'}")


if __name__ == "__main__":
    main()
