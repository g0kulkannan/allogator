"""Fig. S2A: 9-method permutation-significance overlap matrix.

Cells = number of proteins where both row and column methods reach
permutation p < 0.05. Diagonal = total significant per method.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from reproduction.io import load_benchmark, load_residue_scores
from reproduction.paths import FIGURES, auroc_json
from reproduction.scoring import candidate_mask

METHODS = ["Ohm", "EVcouplings", "ESM1b", "ESMpp", "ESM2_650M", "ProtT5",
           "ESM1b_contacts", "Distance", "Random"]
LABELS  = ["Ohm", "EVcouplings", "ESM1b", "ESM++", "ESM2 (650M)",
           "ProtT5", "ESM1b contacts", "3D distance", "ESM1b randomized"]
N_PERM = 1000
SEED = 42
OUT = FIGURES / "figS2a_overlap_compact"


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


def _significant(
    method: str,
    benchmark,
    contact_cache_root: Path | None = None,
) -> set[str]:
    if method == "ESM1b_contacts" and contact_cache_root is not None:
        p_path = contact_cache_root / method / "per_protein_auroc.json"
    else:
        p_path = auroc_json(method)
    raw = json.loads(p_path.read_text()) if p_path.exists() else {}
    ps: dict[str, float] = {}
    for u, v in raw.items():
        if isinstance(v, dict) and "AUROC" in v:
            val = v["AUROC"]
            if isinstance(val, list) and len(val) >= 2 and val[1] is not None:
                ps[u] = float(val[1])
    if ps:
        return {u for u, p in ps.items() if p < 0.05}
    # Recompute from residue scores if AUROC json doesn't carry the p
    try:
        if method == "ESM1b_contacts" and contact_cache_root is not None:
            raw_scores = json.loads(
                (contact_cache_root / method / "residue_scores.json").read_text()
            )
            rs = {
                accession: {int(position): float(score) for position, score in values.items()}
                for accession, values in raw_scores.items()
            }
        else:
            rs = load_residue_scores(method)
    except FileNotFoundError:
        return set()
    rng = np.random.default_rng(SEED)
    sig = set()
    for u, scores in rs.items():
        if u not in benchmark: continue
        entry = benchmark[u]
        L = len(entry["sequence"])
        mask = candidate_mask(L, entry["active"])
        cand = [p for p in range(1, L + 1) if mask[p - 1]]
        s = np.array([scores.get(p, np.nan) for p in cand], dtype=float)
        y = np.array([1 if p in entry["allo"] else 0 for p in cand], dtype=int)
        if y.sum() in (0, len(y)): continue
        if not np.isfinite(s).any(): continue
        s = np.where(np.isnan(s), np.nanmedian(s), s)
        auroc = roc_auc_score(y, s)
        ranks = rankdata(s, method="average")
        n_positive = int(y.sum())
        n_negative = int(y.size - n_positive)
        offset = n_positive * (n_positive + 1) / 2
        denominator = n_positive * n_negative
        yt = y.copy()
        at_least_observed = 0
        for _ in range(N_PERM):
            rng.shuffle(yt)
            permuted = (float(ranks[yt == 1].sum()) - offset) / denominator
            at_least_observed += int(permuted >= auroc)
        if at_least_observed / N_PERM < 0.05:
            sig.add(u)
    return sig


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark()
    sig = {
        m: _significant(m, benchmark, args.contact_cache_root) for m in METHODS
    }
    n = len(METHODS)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            if j < i: continue
            mat[i, j] = len(sig[a]) if i == j else len(sig[a] & sig[b])
    df = pd.DataFrame(mat, index=LABELS, columns=LABELS)
    df.to_csv(args.output_dir / "matrix.csv")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                linewidths=0.4,
                annot_kws={"weight": "bold", "size": 11},
                mask=df.isna(), vmin=0, ax=ax)
    for i in range(n):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False,
                                edgecolor="royalblue", lw=2.0))
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()
    fig.savefig(args.output_dir / "overlap_compact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for m, label in zip(METHODS, LABELS):
        print(f"  {label:14s}: {len(sig[m])}")
    print(f"saved {args.output_dir/'overlap_compact.png'}")


if __name__ == "__main__":
    main()
