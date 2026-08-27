"""Score the benchmark with the chosen ESM-2 sizes (8M → 15B) using the
same mean(heads × layers) ↦ sum(active rows) recipe as ESM-1b.

Output:

    manuscript/data/cached_results/ESM2_<size>/per_protein_auroc.json
    manuscript/data/cached_results/ESM2_<size>/residue_scores.json

CLI:

    python run_esm2_scale.py                  # all six sizes
    python run_esm2_scale.py --sizes 3B 15B   # subset
    python run_esm2_scale.py --sizes 15B --fp16    # fp16 for tight VRAM
    python run_esm2_scale.py --device cuda    # force a backend

Approximate VRAM budgets (fp32 / fp16, batch = 1):

    8M     1 / 1 GB     150M   4 / 2 GB     3B    25 / 13 GB
    35M    1 / 1 GB     650M   8 / 4 GB     15B  60 / 30 GB

Per-protein AUROCs and residue scores are flushed every ``FLUSH_EVERY``
proteins, so an interrupted run resumes where it left off.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from allobench.attention import (load_model, attention_tensor, free_model,
                                  get_device)
from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc

ALL_SIZES = ["8M", "35M", "150M", "650M", "3B", "15B"]
FLUSH_EVERY = 5


def _resolve_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return get_device()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA backend found")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested but no MPS backend found")
        return torch.device("mps")
    return torch.device(name)


def score_one_size(size: str, benchmark: dict,
                    device: torch.device, dtype: torch.dtype) -> None:
    tag = f"ESM2_{size}"
    out_dir = CACHED / tag; out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "per_protein_auroc.json"
    s_path = out_dir / "residue_scores.json"
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    scores  = json.loads(s_path.read_text()) if s_path.exists() else {}
    todo = [u for u in sorted(benchmark)
            if u not in metrics or u not in scores]
    print(f"[{tag}] cached {len(metrics)}; running {len(todo)}")
    if not todo:
        return

    # load on the user-selected device + dtype rather than the helper's
    # auto-pick, so cluster jobs can pin to CUDA and 15B can run fp16.
    import esm
    builder = {
        "8M":   esm.pretrained.esm2_t6_8M_UR50D,
        "35M":  esm.pretrained.esm2_t12_35M_UR50D,
        "150M": esm.pretrained.esm2_t30_150M_UR50D,
        "650M": esm.pretrained.esm2_t33_650M_UR50D,
        "3B":   esm.pretrained.esm2_t36_3B_UR50D,
        "15B":  esm.pretrained.esm2_t48_15B_UR50D,
    }[size]
    model, alphabet = builder()
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters(): p.requires_grad_(False)
    n_layers = sum(1 for _ in model.layers)
    print(f"[{tag}] model on {device} dtype={dtype} ({n_layers} layers)")

    since_flush = 0; t0 = time.time()
    for u in tqdm(todo, desc=tag):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        try:
            attn = attention_tensor(model, alphabet, device, seq, n_layers)
        except RuntimeError as e:
            print(f"[{tag}] skip {u}: {e}"); continue
        full = attn.mean(axis=(0, 1))[1:-1, 1:-1]
        act_idx = np.asarray([a - 1 for a in active], dtype=np.int64)
        per_res = full[act_idx, :].sum(axis=0)
        mask = candidate_mask(L, active)
        score_map = {int(p + 1): float(per_res[p]) for p in range(L)}
        auroc = per_protein_auroc(score_map, allo, active, L)
        if auroc is None: continue
        metrics[u] = {"AUROC": [auroc]}
        scores[u]  = {str(p + 1): float(per_res[p])
                      for p in range(L) if mask[p]}
        since_flush += 1
        if since_flush >= FLUSH_EVERY:
            m_path.write_text(json.dumps(metrics, indent=2))
            s_path.write_text(json.dumps(scores, indent=2))
            since_flush = 0

    m_path.write_text(json.dumps(metrics, indent=2))
    s_path.write_text(json.dumps(scores, indent=2))
    v = [m["AUROC"][0] for m in metrics.values()]
    print(f"[{tag}] done in {(time.time() - t0)/60:.1f} min; "
          f"n={len(v)} mean={np.mean(v):.4f}")
    free_model(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=ALL_SIZES,
                    choices=ALL_SIZES,
                    help="which ESM-2 sizes to score (default: all six)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="compute backend; 'auto' tries MPS → CUDA → CPU")
    ap.add_argument("--fp16", action="store_true",
                    help="run inference in fp16 (recommended for 3B / 15B "
                         "on cards with <40 GB VRAM; not supported on CPU)")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    if dtype is torch.float16 and device.type == "cpu":
        raise SystemExit("--fp16 not supported on CPU; drop --fp16 or use a GPU")

    benchmark = load_benchmark()
    for size in args.sizes:
        score_one_size(size, benchmark, device, dtype)


if __name__ == "__main__":
    main()
