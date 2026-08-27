"""Per-layer and per-head AUROC for ESM-1b.

For Fig. S4: rank each residue by (sum over active rows) of the
attention matrix at each (layer, head) pair, and compute per-protein
AUROC. We emit two CSV tables:

    manuscript/data/cached_results/PerLayer/per_layer_auroc.csv     # 109 × 33
    manuscript/data/cached_results/PerLayer/per_head_auroc.csv      # 109 × 660

The 660-column ordering is row-major: layer 0 head 1, layer 0 head 2, …
"""
from __future__ import annotations
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from allobench.attention import load_model, attention_tensor
from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask

METHOD = "PerLayer"
N_LAYERS = 33
N_HEADS = 20


def main() -> None:
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    per_layer_path = out_dir / "per_layer_auroc.csv"
    per_head_path  = out_dir / "per_head_auroc.csv"

    if per_layer_path.exists() and per_head_path.exists():
        layer_df = pd.read_csv(per_layer_path, index_col=0)
        head_df  = pd.read_csv(per_head_path, index_col=0)
        layer_df.columns = [int(c) for c in layer_df.columns]
        head_df.columns  = [int(c) for c in head_df.columns]
    else:
        layer_df = pd.DataFrame(index=sorted(benchmark),
                                 columns=range(N_LAYERS), dtype=float)
        head_df  = pd.DataFrame(index=sorted(benchmark),
                                 columns=range(N_LAYERS * N_HEADS), dtype=float)

    todo = [u for u in sorted(benchmark)
            if layer_df.loc[u].isna().any() or head_df.loc[u].isna().any()]
    print(f"[perlayer] running on {len(todo)} proteins")
    if not todo:
        return

    model, alphabet, n_layers, device = load_model("ESM1b")
    print(f"[perlayer] model on {device}")

    t0 = time.time()
    for u in tqdm(todo, desc="per-layer"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        try:
            attn = attention_tensor(model, alphabet, device, seq, n_layers)
        except RuntimeError as e:
            print(f"[perlayer] skip {u}: {e}"); continue
        mask = candidate_mask(L, active)
        cand_idx = np.where(mask)[0]
        act_idx  = np.asarray([a - 1 for a in active], dtype=np.int64)
        labels = np.array([1 if (p + 1) in allo else 0 for p in cand_idx], dtype=int)
        if labels.sum() == 0 or labels.sum() == len(labels): continue

        # full attn shape: (n_layers, n_heads, L+2, L+2)
        residue_block = attn[:, :, 1:-1, 1:-1]
        sub = residue_block[:, :,
                              act_idx[:, None], cand_idx[None, :]]  # (NL,NH,na,nc)
        head_scores = sub.sum(axis=2)                                # (NL,NH,nc)
        for lyr in range(N_LAYERS):
            for hd in range(N_HEADS):
                try:
                    head_df.loc[u, lyr * N_HEADS + hd] = float(
                        roc_auc_score(labels, head_scores[lyr, hd]))
                except ValueError:
                    pass
            try:
                layer_df.loc[u, lyr] = float(
                    roc_auc_score(labels, head_scores[lyr].mean(axis=0)))
            except ValueError:
                pass

        # write checkpoints every few proteins
        if todo.index(u) % 5 == 0:
            layer_df.to_csv(per_layer_path)
            head_df.to_csv(per_head_path)

    layer_df.to_csv(per_layer_path)
    head_df.to_csv(per_head_path)
    print(f"[perlayer] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
