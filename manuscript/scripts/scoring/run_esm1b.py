"""Score the 109-protein benchmark with ESM-1b mean attention.

Canonical attention summarisation: mean over (33 layers × 20 heads),
then sum over active-site rows for each candidate column. Output:

    manuscript/data/cached_results/ESM1b/per_protein_auroc.json
    manuscript/data/cached_results/ESM1b/residue_scores.json

Resumable — protein AUROCs and residue scores are flushed every
``FLUSH_EVERY`` proteins.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from allobench.attention import (
    attention_scores_from_tensor,
    attention_tensor,
    load_model,
)
from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc


METHOD = "ESM1b"
FLUSH_EVERY = 5


def main() -> None:
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "per_protein_auroc.json"
    scores_path  = out_dir / "residue_scores.json"

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    scores  = json.loads(scores_path.read_text())  if scores_path.exists()  else {}
    todo = [u for u in sorted(benchmark)
            if u not in metrics or u not in scores]
    print(f"[esm1b] {len(benchmark)} proteins; cached {len(metrics)}; "
          f"running {len(todo)}")
    if not todo:
        return

    model, alphabet, n_layers, device = load_model("ESM1b")
    print(f"[esm1b] model on {device}")

    since_flush = 0
    t0 = time.time()
    for u in tqdm(todo, desc="esm1b"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L: continue
        if max(allo, default=0) > L:   continue
        try:
            attn = attention_tensor(model, alphabet, device, seq, n_layers)
        except RuntimeError as e:
            print(f"[esm1b] skip {u}: {e}")
            continue

        per_res = attention_scores_from_tensor(attn, active)
        score_map = {int(p + 1): float(per_res[p]) for p in range(L)}

        auroc = per_protein_auroc(score_map, allo, active, L)
        if auroc is None:
            continue
        metrics[u] = {"AUROC": [auroc]}
        scores[u]  = {str(p + 1): float(per_res[p])
                      for p in range(L) if candidate_mask(L, active)[p]}

        since_flush += 1
        if since_flush >= FLUSH_EVERY:
            metrics_path.write_text(json.dumps(metrics, indent=2))
            scores_path.write_text(json.dumps(scores, indent=2))
            since_flush = 0

    metrics_path.write_text(json.dumps(metrics, indent=2))
    scores_path.write_text(json.dumps(scores, indent=2))
    vals = [v["AUROC"][0] for v in metrics.values()]
    print(f"[esm1b] done in {(time.time() - t0)/60:.1f} min; "
          f"n={len(vals)} mean AUROC={np.mean(vals):.4f} "
          f"median={np.median(vals):.4f}")


if __name__ == "__main__":
    main()
