"""Score the benchmark with EVcouplings sequence-coupling scores.

For each protein we load the per-pair coupling table emitted by the
EVcouplings server (CSV with i, j, cn columns), reindex it to the full
UniProt sequence length, impute unresolved entries with the global
minimum minus 10 (so they rank last), then sum the columns of the
active-site residues to obtain a per-residue allosteric score.

Sequence-adjacent residues (within ±1 of an active site) are dropped
before AUROC. Permutation p-values use 1000 label shuffles.

Input tables live under ``manuscript/data/predictions_raw/evcouplings/{uniprot}.csv``
(or ``.csv.gz`` — pandas reads either transparently). The analysis uses the
three columns ``i``, ``j``, and ``cn``; see
``manuscript/scripts/build/import_evcouplings.py`` for the slim-and-compress step.

Output:
    manuscript/data/cached_results/EVcouplings/per_protein_auroc.json
    manuscript/data/cached_results/EVcouplings/residue_scores.json
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from reproduction.io import load_benchmark
from reproduction.paths import CACHED, PREDICTIONS_RAW

METHOD = "EVcouplings"
EVC_DIR = PREDICTIONS_RAW / "evcouplings"
N_PERM = 1000
SEED = 42


def _auroc_with_perm(scores: dict[int, float],
                      allo: set[int],
                      n_perm: int = N_PERM,
                      seed: int = SEED) -> tuple[float | None, float | None]:
    if not scores: return None, None
    items = sorted(scores.items())
    y = np.array([1 if r in allo else 0 for r, _ in items], dtype=int)
    s = np.array([v for _, v in items], dtype=float)
    if y.sum() == 0 or y.sum() == len(y): return None, None
    auroc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    yt = y.copy(); perms = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(yt); perms[i] = roc_auc_score(yt, s)
    return auroc, float((perms >= auroc).sum() / n_perm)


def main() -> None:
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}; scores: dict = {}; skipped = []

    for u in tqdm(sorted(benchmark), desc="EVcouplings"):
        entry = benchmark[u]
        active = sorted(entry["active"]); allo = entry["allo"]
        L = len(entry["sequence"])
        csv = EVC_DIR / f"{u}.csv"
        if not csv.exists():
            csv = EVC_DIR / f"{u}.csv.gz"
        if not csv.exists():
            skipped.append(f"{u}: missing CSV"); continue
        df = pd.read_csv(csv, usecols=["i", "j", "cn"])
        tab = df.pivot(index="i", columns="j", values="cn")
        tab = tab.fillna(tab.T)
        full = list(range(1, L + 1))
        tab = tab.reindex(index=full, columns=full)
        mn = float(np.nanmin(tab.values))
        tab = tab.fillna(mn - 10.0)
        avail = [a for a in active if a in tab.columns]
        if not avail:
            skipped.append(f"{u}: no active in EVC table"); continue
        sums = tab[avail].sum(axis=1)
        score_map = {int(r): float(s) for r, s in sums.items()}
        # adjacency filter
        score_map = {r: s for r, s in score_map.items()
                     if all(abs(r - a) >= 2 for a in active)}
        auroc, p = _auroc_with_perm(score_map, allo)
        if auroc is None:
            skipped.append(f"{u}: degenerate y"); continue
        metrics[u] = {"AUROC": [auroc, p]}
        scores[u]  = {str(r): float(v) for r, v in score_map.items()}

    (out_dir / "per_protein_auroc.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "residue_scores.json").write_text(json.dumps(scores, indent=2))
    if skipped:
        (out_dir / "skipped.txt").write_text("\n".join(skipped))
    if metrics:
        v = [m["AUROC"][0] for m in metrics.values()]
        n_sig = sum(1 for m in metrics.values() if m["AUROC"][1] is not None and m["AUROC"][1] < 0.05)
        print(f"[evc] n={len(v)} mean AUROC={np.mean(v):.4f} "
              f"sig={n_sig}/{len(v)} at p<0.05")


if __name__ == "__main__":
    main()
