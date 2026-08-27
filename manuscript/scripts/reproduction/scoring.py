"""Per-protein AUROC scoring and paired comparisons.

The benchmark is positive-unlabeled: residues in `allo` are confident
positives, everything else (apart from active-site and sequence-adjacent
positions) is treated as a putative negative. Per-protein AUROC is the
canonical evaluation metric. Scores are never pooled across proteins.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score

from allobench.candidates import candidate_mask


def per_protein_auroc(scores: Dict[int, float],
                      allo: Iterable[int],
                      active: Iterable[int],
                      seq_length: int) -> float | None:
    """AUROC over candidate positions only. Returns None if all candidate
    labels are the same class."""
    mask = candidate_mask(seq_length, active)
    cand = np.where(mask)[0]                # 0-indexed candidate positions
    if cand.size == 0: return None
    allo_set = {int(r) for r in allo}
    y = np.array([1 if (p+1) in allo_set else 0 for p in cand], dtype=int)
    if y.sum() == 0 or y.sum() == len(y): return None
    s = np.array([scores.get(int(p+1), np.nan) for p in cand], dtype=float)
    if not np.isfinite(s).any(): return None
    # impute missing positions with the per-protein median so they rank
    # neutrally rather than at the top/bottom.
    if np.isnan(s).any():
        s = np.where(np.isnan(s), np.nanmedian(s), s)
    return float(roc_auc_score(y, s))


def summarise_paired(a_name: str, a: Dict[str, float],
                      b_name: str, b: Dict[str, float]) -> pd.DataFrame:
    """Paired one-sided t-test (a > b) on per-protein AUROC across the
    intersection of proteins scored by both methods. Also reports the
    Wilcoxon signed-rank one-sided p-value as a non-parametric check."""
    common = sorted(set(a) & set(b))
    va = np.array([a[u] for u in common], dtype=float)
    vb = np.array([b[u] for u in common], dtype=float)
    d  = va - vb
    if len(common) < 3:
        return pd.DataFrame([{"a": a_name, "b": b_name, "n": len(common),
                               "mean_a": float(va.mean()) if len(va) else np.nan,
                               "mean_b": float(vb.mean()) if len(vb) else np.nan,
                               "mean_d": float(d.mean()) if len(d) else np.nan,
                               "t": np.nan, "p_t_one_sided": np.nan,
                               "p_wilcoxon_one_sided": np.nan}])
    t, p2 = ttest_rel(va, vb)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    try:
        _, pw = wilcoxon(va, vb, alternative="greater")
    except ValueError:
        pw = float("nan")
    return pd.DataFrame([{
        "a": a_name, "b": b_name, "n": len(common),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
        "mean_d": float(d.mean()),
        "a_better_n": int((d > 0).sum()),
        "t": float(t), "p_t_one_sided": float(p1),
        "p_wilcoxon_one_sided": float(pw),
    }])
