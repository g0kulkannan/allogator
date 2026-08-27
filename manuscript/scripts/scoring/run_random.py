"""Random baseline: replace the attention matrix with draws from |N(0,1)|
and follow the same active-row-sum recipe as ESM-1b.

Seeded for reproducibility. Per-protein AUROCs should average ~0.5.
"""
from __future__ import annotations
import json

import numpy as np

from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc

METHOD = "Random"
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}; scores: dict = {}

    for u in sorted(benchmark):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        # Keep NumPy's default float64 draw: this is the seeded publication
        # baseline and changing the draw dtype changes the random stream.
        mat = np.abs(rng.standard_normal((L, L)))
        act_idx = np.asarray([a - 1 for a in active], dtype=np.int64)
        per_res = mat[act_idx, :].sum(axis=0)
        score_map = {int(p + 1): float(per_res[p]) for p in range(L)}
        auroc = per_protein_auroc(score_map, allo, active, L)
        if auroc is None: continue
        metrics[u] = {"AUROC": [auroc]}
        mask = candidate_mask(L, active)
        scores[u]  = {str(p + 1): float(per_res[p])
                      for p in range(L) if mask[p]}

    (out_dir / "per_protein_auroc.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "residue_scores.json").write_text(json.dumps(scores, indent=2))
    v = [m["AUROC"][0] for m in metrics.values()]
    print(f"[random] n={len(v)} mean AUROC={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
