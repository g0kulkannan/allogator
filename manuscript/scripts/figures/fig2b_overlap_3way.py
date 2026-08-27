"""Fig. 2B: three-way overlap matrix.

Diagonal = number of benchmark proteins where the method's per-protein
AUROC reaches permutation p < 0.05. Off-diagonal = number of proteins
where both methods reach that threshold.

The Ohm and EVcouplings AUROC tables contain permutation p-values
(``AUROC: [auroc, p_value]``). When a method table does not contain them,
this script calculates the permutation p-values from its residue scores.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import roc_auc_score

from reproduction.io import load_benchmark, load_residue_scores
from reproduction.paths import FIGURES, auroc_json
from reproduction.scoring import candidate_mask

MODELS = ["Ohm", "EVcouplings", "ESM1b"]
LABELS = ["Network model\n(Ohm)",
          "Coevolutionary\nanalysis (EVcouplings)",
          "Language model\n(ESM-1b)"]
N_PERM = 1000
SEED = 42
OUT = FIGURES / "fig2b_overlap_3way"


def _perm_p_value(method: str, benchmark) -> dict[str, float]:
    """Re-derive permutation p-values for a method from its residue
    score JSON."""
    rs = load_residue_scores(method)
    out: dict[str, float] = {}
    rng = np.random.default_rng(SEED)
    for u, score_dict in rs.items():
        if u not in benchmark: continue
        entry = benchmark[u]
        L = len(entry["sequence"])
        mask = candidate_mask(L, entry["active"])
        cand = [p for p in range(1, L + 1) if mask[p - 1]]
        s = np.array([score_dict.get(p, np.nan) for p in cand], dtype=float)
        y = np.array([1 if p in entry["allo"] else 0 for p in cand], dtype=int)
        if y.sum() in (0, len(y)): continue
        if not np.isfinite(s).any(): continue
        s = np.where(np.isnan(s), np.nanmedian(s), s)
        auroc = roc_auc_score(y, s)
        yt = y.copy(); perm = np.zeros(N_PERM)
        for i in range(N_PERM):
            rng.shuffle(yt); perm[i] = roc_auc_score(yt, s)
        out[u] = float((perm >= auroc).sum() / N_PERM)
    return out


def _significant_set(method: str, benchmark) -> set[str]:
    """{uniprot : p < 0.05} for one method. Recomputes permutation
    p-values from the residue-score JSON when the cached AUROC file
    doesn't already carry one."""
    p_path = auroc_json(method)
    raw = json.loads(p_path.read_text()) if p_path.exists() else {}
    ps: dict[str, float] = {}
    for u, v in raw.items():
        if isinstance(v, dict) and "AUROC" in v:
            val = v["AUROC"]
            if isinstance(val, list) and len(val) >= 2 and val[1] is not None:
                ps[u] = float(val[1])
    if not ps:
        ps = _perm_p_value(method, benchmark)
    return {u for u, p in ps.items() if p < 0.05}


def main() -> None:
    benchmark = load_benchmark()
    significant = {m: _significant_set(m, benchmark) for m in MODELS}
    OUT.mkdir(parents=True, exist_ok=True)

    n = len(MODELS)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if j < i: continue
            mat[i, j] = len(significant[a]) if i == j \
                        else len(significant[a] & significant[b])
    df = pd.DataFrame(mat, index=LABELS, columns=LABELS)
    df.to_csv(OUT / "matrix.csv")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues",
                cbar=False, linewidths=0.5,
                annot_kws={"weight": "bold", "size": 22},
                mask=df.isna(), vmin=0, ax=ax)
    for i in range(n):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False,
                                edgecolor="royalblue", lw=3))
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.title("Proteins with permutation p < 0.05\n"
              "diagonal: total / off-diagonal: shared")
    plt.tight_layout()
    plt.savefig(OUT / "overlap_3way.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string())
    print(f"\nsaved {OUT/'overlap_3way.png'}")


if __name__ == "__main__":
    main()
