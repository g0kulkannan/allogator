#!/usr/bin/env python3
"""Generate the plotted panels of Figs. 3, S8, and S9.

Structural views are assembled separately in PyMOL. Every scatter,
distribution, composition, domain track, and statistic shown here is built
from the case-study outputs.
Run ``build_case_study_outputs.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

POTENCY_CUTOFF = 0.87
EFFICACY_CUTOFF = 0.75
WT_COLOR = "#f59e42"
ALTERED_COLOR = "#171796"
CLINICAL_COLOR = "#8e0a91"
UNKNOWN_COLOR = "#858585"
TOP_COLOR = "#dd1c77"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _add_signaling_thresholds(ax: plt.Axes) -> None:
    ax.axvline(POTENCY_CUTOFF, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(EFFICACY_CUTOFF, color="#333333", linestyle="--", linewidth=1)
    ax.text(-0.62, 1.32, "wild-type-like\nsignaling", ha="left", va="top", fontsize=8)
    ax.text(2.50, 1.32, "low potency", ha="right", va="top", fontsize=8)
    ax.text(-0.62, 0.23, "low efficacy", ha="left", va="bottom", fontsize=8)
    ax.text(2.50, 0.23, "low potency\nlow efficacy", ha="right", va="bottom", fontsize=8)
    ax.set_xlim(-0.7, 2.6)
    ax.set_ylim(0.2, 1.4)
    ax.set_xlabel("Normalised logEC50")
    ax.set_ylabel("Normalised amplitude")


def _attention_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    rank_column: str,
    title: str,
) -> None:
    clean = frame.dropna(
        subset=["normalised_logEC50", "normalised_amplitude", rank_column]
    )
    rank_percentile = (clean[rank_column] - 1) / max(len(clean) - 1, 1)
    scatter = ax.scatter(
        clean["normalised_logEC50"],
        clean["normalised_amplitude"],
        c=rank_percentile,
        cmap="plasma_r",
        vmin=0,
        vmax=1,
        s=15,
        alpha=0.9,
        linewidths=0,
    )
    _add_signaling_thresholds(ax)
    ax.set_title(f"{title} (n={len(clean)})", fontsize=10)
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Within-method rank percentile", fontsize=8)
    colorbar.set_ticks([0, 1])
    colorbar.ax.tick_params(labelsize=7)


def _signaling_group(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame["potency"].eq("WT-like") & frame["efficacy"].eq("WT-like"),
        "WT-like signaling",
        "Altered signaling",
    )


def plot_figure3() -> None:
    pharmacology = pd.read_csv(OUTPUTS / "b2ar_plot_data.csv")
    stats = _read_json(OUTPUTS / "b2ar_statistics.json")
    scatter_data = pharmacology.dropna(subset=["attention"]).copy()

    group_data = pharmacology.loc[
        pharmacology["expression_low"].eq(False)
        & pharmacology["attention"].notna()
        & pharmacology["potency"].notna()
        & pharmacology["efficacy"].notna()
    ].copy()
    group_data["signaling_group"] = _signaling_group(group_data)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    _attention_scatter(axes[0], scatter_data, "attention_rank", "ESM-1b attention")
    axes[0].set_title("", loc="center")
    axes[0].set_title("A  B2AR signaling colored by ESM-1b rank", loc="left", fontsize=10)

    groups = ["WT-like signaling", "Altered signaling"]
    values = [group_data.loc[group_data["signaling_group"].eq(g), "attention"] for g in groups]
    violins = axes[1].violinplot(values, positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(violins["bodies"], ["#e4a518", "#c95369"]):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.9)
    rng = np.random.default_rng(13)
    for position, vals in enumerate(values):
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        axes[1].scatter(
            position + jitter,
            vals,
            s=7,
            color="#202020",
            alpha=0.65,
            linewidths=0,
            zorder=3,
        )

    # Highlight the full top five, including expression-low motif variants
    # that are not part of the signaling-group comparison.
    top_five = scatter_data.nlargest(5, "attention").copy()
    top_five["plot_x"] = np.where(
        top_five["potency"].eq("WT-like") & top_five["efficacy"].eq("WT-like"),
        0.0,
        1.0,
    )
    axes[1].scatter(
        top_five["plot_x"],
        top_five["attention"],
        s=24,
        color=TOP_COLOR,
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )
    for row in top_five.loc[top_five["motif"].notna()].itertuples():
        axes[1].annotate(
            f"{row.GPCRdb}\n({row.motif})",
            (row.plot_x, row.attention),
            xytext=(-8, 3),
            textcoords="offset points",
            fontsize=7,
            ha="right",
        )

    mw = stats["signaling_group_test"]
    axes[1].text(
        0.02,
        0.98,
        f"two-sided Mann–Whitney U\np={mw['p']:.3g}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes[1].set_xticks([0, 1], [f"WT-like\n(n={len(values[0])})", f"Altered\n(n={len(values[1])})"], rotation=25)
    axes[1].set_ylabel("ESM-1b attention")
    axes[1].set_title("B  Attention by signaling phenotype", loc="left", fontsize=10)
    _style_axis(axes[1])
    fig.tight_layout()
    _save(fig, "figure3")


def _non_wt_like(frame: pd.DataFrame) -> pd.Series:
    return frame["normalised_logEC50"].ge(POTENCY_CUTOFF) | frame[
        "normalised_amplitude"
    ].lt(EFFICACY_CUTOFF)


def plot_figure_s8() -> None:
    pharmacology = pd.read_csv(OUTPUTS / "b2ar_plot_data.csv")
    statistics = _read_json(OUTPUTS / "b2ar_statistics.json")
    methods = [
        ("ESM-1b", "attention", "attention_rank", "attention", "esm1b_attention"),
        ("Ohm", "ohm_ACI", "ohm_ACI_rank", "ohm", "ohm"),
        ("EVcouplings", "evc_score", "evc_score_rank", "evcouplings", "evcouplings"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 10.0), gridspec_kw={"width_ratios": [1.45, 1]})

    for row_index, (
        name,
        score_column,
        rank_column,
        correlation_key,
        top50_key,
    ) in enumerate(methods):
        clean = pharmacology.dropna(
            subset=[score_column, rank_column, "normalised_logEC50", "normalised_amplitude"]
        ).copy()
        _attention_scatter(axes[row_index, 0], clean, rank_column, name)
        rho_x = spearmanr(clean[score_column], clean["normalised_logEC50"])
        rho_y = spearmanr(clean[score_column], clean["normalised_amplitude"])
        expected_correlation = statistics[correlation_key]
        if (
            int(expected_correlation["n"]) != len(clean)
            or not np.isclose(
                rho_x.statistic,
                expected_correlation["normalised_logEC50"]["spearman_rho"],
            )
            or not np.isclose(
                rho_y.statistic,
                expected_correlation["normalised_amplitude"]["spearman_rho"],
            )
        ):
            raise ValueError(f"Figure S8 {name} correlations differ from builder output")

        top = clean.nlargest(50, score_column).copy()
        top["non_WT_like"] = _non_wt_like(top)
        altered = int(top["non_WT_like"].sum())
        wt_like = int(len(top) - altered)
        expected_top50 = statistics["top50_quadrant_enrichment"]["methods"][
            top50_key
        ]
        if altered != int(expected_top50["non_wt_like_n"]):
            raise ValueError(f"Figure S8 {name} top-50 count differs from builder output")
        axes[row_index, 1].pie(
            [wt_like, altered],
            labels=["WT-like", "non-WT-like"],
            colors=[WT_COLOR, ALTERED_COLOR],
            startangle=110,
            autopct=lambda pct, total=50: f"{pct:.0f}%\n({pct * total / 100:.0f})",
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
        )
        axes[row_index, 1].set_title(f"Top 50 {name} residues", fontsize=10)

    fig.tight_layout()
    _save(fig, "figureS8")


def _plot_domain_track(ax: plt.Axes) -> None:
    domains = [
        (1, 555, "NTD", "#4f4a92"),
        (556, 623, "DBD", "#278f9c"),
        (624, 668, "H", "#2cad8e"),
        (669, 920, "LBD", "#76cf4f"),
    ]
    for start, end, label, color in domains:
        ax.barh(0, end - start + 1, left=start, height=0.72, color=color, edgecolor="none")
        ax.text((start + end) / 2, 0, label, ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(1, 920)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("Domains", rotation=0, ha="right", va="center", labelpad=42)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_figure_s9() -> None:
    points = pd.read_csv(OUTPUTS / "ar_figureS9_plot_data.csv")
    coverage = pd.read_csv(OUTPUTS / "ar_pdb_coverage.csv")
    plddt = pd.read_csv(OUTPUTS / "ar_af3_plddt.csv")
    ar_stats = _read_json(OUTPUTS / "ar_statistics.json")
    active = sorted(
        int(x) for x in ar_stats["active_site_residue_weights"]
    )

    fig = plt.figure(figsize=(10.5, 7.0))
    grid = fig.add_gridspec(4, 1, height_ratios=[0.42, 0.42, 0.65, 3.1], hspace=0.32)
    ax_domain = fig.add_subplot(grid[0])
    ax_coverage = fig.add_subplot(grid[1])
    ax_plddt = fig.add_subplot(grid[2])
    ax_attention = fig.add_subplot(grid[3])

    _plot_domain_track(ax_domain)
    coverage_values = coverage["present_in_any_structure"].astype(int).to_numpy()[None, :]
    ax_coverage.imshow(
        coverage_values,
        aspect="auto",
        interpolation="nearest",
        extent=[1, len(coverage), 0, 1],
        cmap=ListedColormap(["#ffad38", "#6210a5"]),
        vmin=0,
        vmax=1,
    )
    ax_coverage.set_xlim(1, 920)
    ax_coverage.set_yticks([])
    ax_coverage.set_xticks([])
    ax_coverage.set_ylabel("PDB coverage", rotation=0, ha="right", va="center", labelpad=42)

    plddt_image = ax_plddt.imshow(
        plddt["pLDDT"].to_numpy()[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=[1, len(plddt), 0, 1],
        cmap="plasma_r",
        vmin=0,
        vmax=100,
    )
    ax_plddt.set_xlim(1, 920)
    ax_plddt.set_yticks([])
    ax_plddt.set_xticks([])
    ax_plddt.set_ylabel("AF3 pLDDT", rotation=0, ha="right", va="center", labelpad=42)
    cbar = fig.colorbar(plddt_image, ax=ax_plddt, fraction=0.018, pad=0.012)
    cbar.set_ticks([0, 100])

    unknown = points.loc[points["clinical_status"].eq("unknown")]
    clinical = points.loc[points["clinical_status"].eq("clinical_variant")]
    ax_attention.scatter(
        unknown["residue_number"],
        unknown["attention_score"],
        s=14,
        color=UNKNOWN_COLOR,
        alpha=0.82,
        linewidths=0,
        label=f"Unknown (n={len(unknown)})",
    )
    ax_attention.scatter(
        clinical["residue_number"],
        clinical["attention_score"],
        s=15,
        color=CLINICAL_COLOR,
        alpha=0.9,
        linewidths=0,
        label=f"Clinical variant (n={len(clinical)})",
    )
    for residue in active:
        ax_attention.scatter(
            residue,
            -0.003,
            marker="*",
            s=58,
            color="#ff2d8f",
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )
    ax_attention.scatter([], [], marker="*", s=58, color="#ff2d8f", label="Testosterone-binding")
    cutoff = float(ar_stats["attention_cutoff"])
    ax_attention.axhline(cutoff, color="#333333", linestyle="--", linewidth=1)
    for item in ar_stats["high_attention_DBD_residues"]:
        residue = int(item["residue"][1:])
        score = float(item["attention_score"])
        ax_attention.annotate(
            item["residue"],
            (residue, score),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9,
        )
    ax_attention.set_xlim(1, 920)
    ax_attention.set_ylim(-0.008, max(0.2, points["attention_score"].max() * 1.1))
    ax_attention.set_xlabel("Residue number")
    ax_attention.set_ylabel("Attention to androgen-binding residues")
    ax_attention.legend(loc="upper left", frameon=True, fontsize=8)
    _style_axis(ax_attention)
    ax_domain.set_title("A  Androgen-receptor domains, structure coverage, and AF3 confidence", loc="left", fontsize=11)
    ax_attention.set_title("B  Attention to the androgen-binding site", loc="left", fontsize=11)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.10)
    _save(fig, "figureS9")


def main() -> None:
    plot_figure3()
    plot_figure_s8()
    plot_figure_s9()
    print(f"Saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
