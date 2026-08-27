"""Attention × active-site summarisation grid for ESM-1b (Fig. S2D).

The figure compares mean, maximum, minimum, median, and symmetric-maximum
head aggregation with sum or maximum active-site aggregation. Symmetric
maximum is paired with sum aggregation. For the plotted ``mean / max`` cell,
the active-site summary is the equal-weight mean of the mean and maximum
active-site values. ``Mean × sum`` is the method used in Fig. 2 and is
computed by ``run_esm1b.py``; this script computes the other eight schemes
shown in Fig. S2D.

Single ESM-1b forward pass per protein; all nine plotted variants are derived
from the same tensor. Output one JSON pair per cell under
``manuscript/data/cached_results/ESM1b_<attn>_<gamma>/``.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from allobench.attention import load_model, attention_tensor
from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc

ATTN_KEYS  = ["mean", "max", "min", "median", "max_sym"]
GAMMA_KEYS = ["sum", "max"]
# ``mean × sum`` is computed by run_esm1b.py and stored under ESM1b/.
# Symmetric-maximum aggregation is evaluated with sum aggregation.
COMBOS = [(a, g) for a in ATTN_KEYS for g in GAMMA_KEYS
          if not (a == "max_sym" and g == "max")
          and not (a == "mean"    and g == "sum")]
FLUSH_EVERY = 5


def reduce_heads(basis: np.ndarray, key: str) -> np.ndarray:
    if key == "mean":     return basis.mean(axis=(0, 1))
    if key == "max":      return basis.max(axis=(0, 1))
    if key == "min":      return basis.min(axis=(0, 1))
    if key == "median":   return np.median(basis, axis=(0, 1))
    if key == "max_sym":
        base = basis.mean(axis=(0, 1))
        return np.maximum(base, base.T)
    raise ValueError(key)


def reduce_gamma(values: np.ndarray, key: str) -> float:
    if key == "sum": return float(values.sum())
    if key == "max": return float(values.max()) if values.size else 0.0
    raise ValueError(key)


def main() -> None:
    benchmark = load_benchmark()
    accums: dict[tuple[str, str], dict] = {}
    for attn_k, gamma_k in COMBOS:
        tag = f"ESM1b_{attn_k}_{gamma_k}"
        d = CACHED / tag; d.mkdir(parents=True, exist_ok=True)
        m_path = d / "per_protein_auroc.json"
        s_path = d / "residue_scores.json"
        metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
        scores  = json.loads(s_path.read_text()) if s_path.exists() else {}
        accums[(attn_k, gamma_k)] = {"metrics": metrics, "scores": scores,
                                     "m_path": m_path, "s_path": s_path}

    todo = [
        u for u in sorted(benchmark)
        if any(
            u not in accums[c]["metrics"] or u not in accums[c]["scores"]
            for c in COMBOS
        )
    ]
    print(f"[grid] {len(todo)} proteins need inference")
    if not todo:
        _flush(accums); _summary(accums); return

    model, alphabet, n_layers, device = load_model("ESM1b")
    print(f"[grid] model on {device}")

    since_flush = 0
    t0 = time.time()
    for u in tqdm(todo, desc="attn×gamma"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        try:
            attn = attention_tensor(model, alphabet, device, seq, n_layers)
        except RuntimeError as e:
            print(f"[grid] skip {u}: {e}"); continue
        mask = candidate_mask(L, active)
        cand_idx = np.where(mask)[0]
        act_idx  = np.asarray([a - 1 for a in active], dtype=np.int64)

        for attn_k, gamma_k in COMBOS:
            if (u in accums[(attn_k, gamma_k)]["metrics"]
                    and u in accums[(attn_k, gamma_k)]["scores"]):
                continue
            full = reduce_heads(attn, attn_k)[1:-1, 1:-1]
            sub = full[act_idx[:, None], cand_idx[None, :]]
            if (attn_k, gamma_k) == ("mean", "max"):
                # The Figure S2D mean/max control uses an equal-weight blend
                # of the active-site mean and maximum for each candidate.
                sc = (0.5 * (sub.mean(axis=0) + sub.max(axis=0))).astype(
                    np.float32
                )
            else:
                sc = np.array([reduce_gamma(sub[:, j], gamma_k)
                               for j in range(sub.shape[1])], dtype=np.float32)
            score_map = {int(cand_idx[j] + 1): float(sc[j])
                         for j in range(sub.shape[1])}
            auroc = per_protein_auroc(score_map, allo, active, L)
            if auroc is None: continue
            accums[(attn_k, gamma_k)]["metrics"][u] = {"AUROC": [auroc]}
            accums[(attn_k, gamma_k)]["scores"][u]  = {
                str(int(cand_idx[j] + 1)): float(sc[j])
                for j in range(sub.shape[1])}

        del attn
        since_flush += 1
        if since_flush >= FLUSH_EVERY:
            _flush(accums); since_flush = 0

    _flush(accums)
    print(f"[grid] done in {(time.time() - t0)/60:.1f} min")
    _summary(accums)


def _flush(accums):
    for d in accums.values():
        d["m_path"].write_text(json.dumps(d["metrics"], indent=2))
        d["s_path"].write_text(json.dumps(d["scores"], indent=2))


def _summary(accums):
    print("\n=== mean AUROC per variant ===")
    for (a_k, g_k), d in sorted(accums.items()):
        v = [m["AUROC"][0] for m in d["metrics"].values()]
        if v:
            print(f"  ESM1b_{a_k}_{g_k:<3}  n={len(v):3d}  "
                  f"mean={np.mean(v):.4f}  median={np.median(v):.4f}")


if __name__ == "__main__":
    main()
