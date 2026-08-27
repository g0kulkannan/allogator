"""Fig. S3: ESM-2 model-scale effect.

One jitter column per ESM-2 size, light → dark blue with model size,
black tick at each median.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduction.io import load_auroc_metrics
from reproduction.paths import FIGURES

SIZES = ["8M", "35M", "150M", "650M", "3B", "15B"]
OUT = FIGURES / "figS3_esm2_scale"
plt.rcParams["font.size"] = 11


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aurocs: dict[str, list[float]] = {}
    have = []
    for s in SIZES:
        try:
            aurocs[s] = list(load_auroc_metrics(f"ESM2_{s}").values())
            have.append(s)
        except FileNotFoundError:
            print(f"  (skip ESM2_{s}: no cached AUROC)")
    if not have:
        print("no ESM-2 sizes available — run scoring first"); return

    colors = plt.cm.Blues(np.linspace(0.35, 0.92, len(have)))
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    medians = []
    for i, (s, c) in enumerate(zip(have, colors)):
        v = aurocs[s]
        x = [i + rng.uniform(-0.18, 0.18) for _ in v]
        ax.scatter(x, v, s=12, color=c, alpha=0.85,
                   edgecolor="black", linewidth=0.25, zorder=3)
        medians.append(float(np.median(v)))
    for i, m in enumerate(medians):
        ax.hlines(m, i - 0.28, i + 0.28, colors="black", linewidth=1.8, zorder=4)
    ax.set_xlim(-0.5, len(have) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels(have, rotation=90)
    ax.set_xlabel("ESM-2 model size")
    ax.set_ylabel("AUROC")
    plt.tight_layout()
    fig.savefig(OUT / "esm2_scale.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for s, m in zip(have, medians):
        print(f"  ESM-2 {s:5s}  n={len(aurocs[s]):3d}  mean={np.mean(aurocs[s]):.4f}  "
              f"median={m:.4f}")
    print(f"saved {OUT/'esm2_scale.png'}")


if __name__ == "__main__":
    main()
