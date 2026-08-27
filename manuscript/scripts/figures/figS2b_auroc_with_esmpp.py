"""Fig. S2B: per-protein AUROC scatter across all methods.

One column per method, one dot per protein, median marked as a thick
black tick. ESM-1b is highlighted in royal blue against light-gray
comparators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

plt.rcParams["font.size"] = 11

METHODS = ["Ohm", "EVcouplings", "ESM1b", "ESMpp", "ESM2_650M",
           "ProtT5", "ESM1b_contacts", "Distance", "Random"]
LABELS  = ["Ohm", "EVcouplings", "ESM1b", "ESM++", "ESM2 (650M)",
           "ProtT5", "ESM1b contacts", "3D distance", "ESM1b randomized"]
OUT = FIGURES / "figS2b_auroc"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contact-cache-root",
        type=Path,
        default=None,
        help="optional cache root containing ESM1b_contacts/",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def _load_contact_metrics(cache_root: Path) -> dict[str, float]:
    raw = json.loads(
        (cache_root / "ESM1b_contacts" / "per_protein_auroc.json").read_text()
    )
    return {
        accession: float(value["AUROC"][0])
        for accession, value in raw.items()
        if isinstance(value, dict)
        and isinstance(value.get("AUROC"), list)
        and value["AUROC"]
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aurocs = {}
    for m in METHODS:
        try:
            if m == "ESM1b_contacts" and args.contact_cache_root is not None:
                aurocs[m] = _load_contact_metrics(args.contact_cache_root)
            else:
                aurocs[m] = load_auroc_metrics(m)
        except FileNotFoundError:
            aurocs[m] = {}
            print(f"  (skip {m}: no cached AUROC)")

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    medians = []
    for i, m in enumerate(METHODS):
        v = list(aurocs[m].values())
        if not v:
            medians.append(np.nan); continue
        x = [i + rng.uniform(-0.18, 0.18) for _ in v]
        is_esm1b = m == "ESM1b"
        ax.scatter(x, v,
                   s=12 if is_esm1b else 10,
                   alpha=0.85 if is_esm1b else 0.55,
                   color="royalblue" if is_esm1b else "#bbbbbb",
                   edgecolor="black" if is_esm1b else "none",
                   linewidth=0.3 if is_esm1b else 0,
                   zorder=3 if is_esm1b else 2)
        medians.append(float(np.median(v)))
    for i, med in enumerate(medians):
        if np.isnan(med): continue
        ax.hlines(med, i - 0.28, i + 0.28, colors="black", linewidth=1.8, zorder=4)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4)
    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(LABELS, rotation=90)
    ax.set_ylabel("AUROC")
    plt.tight_layout()
    fig.savefig(args.output_dir / "auroc_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for m, lbl in zip(METHODS, LABELS):
        v = list(aurocs[m].values())
        if v:
            print(f"  {lbl:14s}: n={len(v):3d} mean={np.mean(v):.4f} "
                  f"median={np.median(v):.4f}")
    print(f"saved {args.output_dir/'auroc_scatter.png'}")


if __name__ == "__main__":
    main()
