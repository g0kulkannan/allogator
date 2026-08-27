"""Figure S4: ESM-1b single-head and single-layer performance.

The publication panel contains a 33 x 20 heatmap of median per-protein
AUROC for individual attention heads and an adjacent 33 x 1 strip with
the median AUROC obtained by averaging the 20 heads within each layer.
Table columns are stored zero-based; displayed model layers and heads are
numbered 0--32 and 1--20, respectively.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduction.paths import CACHED, FIGURES


OUT = FIGURES / "figS4_per_layer"
N_PROTEINS = 109
N_LAYERS = 33
N_HEADS = 20
COLOR_MIN = 0.47
COLOR_MAX = 0.64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=CACHED,
        help="directory containing the PerLayer analysis tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="directory for publication figure files and summary table",
    )
    return parser.parse_args()


def _load_complete_table(path, n_columns: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    frame = pd.read_csv(path, index_col=0)
    frame.columns = [int(x) for x in frame.columns]
    frame = frame.reindex(columns=range(n_columns))
    if frame.shape != (N_PROTEINS, n_columns):
        raise ValueError(
            f"{path}: expected {(N_PROTEINS, n_columns)}, found {frame.shape}"
        )
    if frame.isna().any().any():
        raise ValueError(f"{path}: publication table contains missing values")
    return frame


def _render_publication_panel(
    per_head_median: np.ndarray,
    per_layer_median: np.ndarray,
    output_dir: Path,
) -> tuple[float, float]:
    # Preserve the fixed scale used in the manuscript artwork.
    # Values outside this interval are intentionally clipped by imshow.
    vmin = COLOR_MIN
    vmax = COLOR_MAX

    fig = plt.figure(figsize=(10.6, 9.4))
    grid = fig.add_gridspec(
        1, 3,
        width_ratios=(20, 1.25, 0.72),
        left=0.08,
        right=0.93,
        bottom=0.08,
        top=0.88,
        wspace=0.12,
    )
    heat_ax = fig.add_subplot(grid[0, 0])
    layer_ax = fig.add_subplot(grid[0, 1], sharey=heat_ax)
    color_ax = fig.add_subplot(grid[0, 2])

    image = heat_ax.imshow(
        per_head_median,
        cmap="plasma_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    layer_ax.imshow(
        per_layer_median[:, None],
        cmap="plasma_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    heat_ax.set_title(
        f"Median per-head AUROC across {N_PROTEINS} proteins",
        fontsize=16,
        pad=12,
    )
    heat_ax.set_xlabel("Attention head", fontsize=13)
    heat_ax.set_ylabel("Layer", fontsize=13)
    heat_ax.set_xticks(range(N_HEADS), labels=range(1, N_HEADS + 1))
    heat_ax.set_yticks(range(N_LAYERS), labels=range(N_LAYERS))
    heat_ax.tick_params(labelsize=10)

    layer_ax.set_xticks([0], labels=["Layer\n(median)"])
    layer_ax.tick_params(axis="x", labelsize=10, pad=8)
    layer_ax.tick_params(axis="y", left=False, labelleft=False)

    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("AUROC", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(
        f"Plasma colormap range: {vmin:.2f}–{vmax:.2f}",
        color="#777777",
        fontsize=11,
        y=0.965,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figureS4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def main() -> None:
    args = _parse_args()
    layer_path = args.cache_root / "PerLayer" / "per_layer_auroc.csv"
    head_path = args.cache_root / "PerLayer" / "per_head_auroc.csv"
    layer_frame = _load_complete_table(layer_path, N_LAYERS)
    head_frame = _load_complete_table(head_path, N_LAYERS * N_HEADS)

    per_head_median = np.median(head_frame.to_numpy(), axis=0).reshape(
        N_LAYERS, N_HEADS
    )
    per_layer_median = np.median(layer_frame.to_numpy(), axis=0)
    best_layer = int(np.argmax(per_layer_median))
    vmin, vmax = _render_publication_panel(
        per_head_median, per_layer_median, args.output_dir
    )

    summary = pd.DataFrame(
        {
            "displayed_layer": np.arange(N_LAYERS),
            "table_column": np.arange(N_LAYERS),
            "median_auroc": per_layer_median,
        }
    )
    summary.to_csv(args.output_dir / "per_layer_median_summary.csv", index=False)
    print(f"  best layer: {best_layer}")
    print(f"  color scale: {vmin:.2f}–{vmax:.2f}")
    print(f"saved {args.output_dir/'figureS4.png'}")


if __name__ == "__main__":
    main()
