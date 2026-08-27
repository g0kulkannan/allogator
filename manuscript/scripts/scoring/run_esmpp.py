"""Score the benchmark with ESM++ (Synthyra/ESMplusplus_large).

ESM++ is a faithful open re-implementation of ESM-C (~600M parameters,
33 layers × 18 heads). Uses the same mean-attention summarisation as
ESM-1b and ESM-2.

Output:
    manuscript/data/cached_results/ESMpp/per_protein_auroc.json
    manuscript/data/cached_results/ESMpp/residue_scores.json
"""
from __future__ import annotations
import argparse
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc, json, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc

MODEL_NAME = "Synthyra/ESMplusplus_large"
SCORING_REVISION = "94b8ccff33b994b0b47842315c7d9bc43bf3b487"
METHOD = "ESMpp"
FLUSH_EVERY = 5


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available():         return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16",
        help="inference dtype; the released cache was calculated in float16",
    )
    parser.add_argument(
        "--revision", default=SCORING_REVISION,
        help="immutable Hugging Face model revision used by this script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import transformers
    from transformers import AutoModelForMaskedLM
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "per_protein_auroc.json"
    s_path = out_dir / "residue_scores.json"
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    scores  = json.loads(s_path.read_text()) if s_path.exists() else {}
    todo = [u for u in sorted(benchmark)
            if u not in metrics or u not in scores]
    print(f"[esm++] cached {len(metrics)}; running {len(todo)}")
    if not todo: return

    device = get_device(args.device)
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype is torch.float16:
        raise SystemExit("float16 ESM++ inference is unsupported on CPU; use --dtype float32")
    print(f"[esm++] loading {MODEL_NAME}@{args.revision} on {device} ({dtype})")
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, revision=args.revision, trust_remote_code=True,
        torch_dtype=dtype
    ).to(device).eval()
    run_metadata = {
        "method": "ESM++ large",
        "model": MODEL_NAME,
        "requested_model_revision": args.revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "inference_dtype": args.dtype,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "score_summary": (
            "Mean attention across layers and heads, summed over active-site "
            "query rows"
        ),
    }
    (out_dir / "method_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n"
    )
    tokenizer = model.tokenizer
    for p in model.parameters(): p.requires_grad_(False)

    since_flush = 0
    t0 = time.time()
    for u in tqdm(todo, desc="ESM++"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        try:
            enc = tokenizer([seq], return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_attentions=True)
                attn_stack = torch.stack(out.attentions, dim=0)[:, 0]
                attn = attn_stack[:, :, 1:L+1, 1:L+1]
                mean_attn = attn.mean(dim=(0, 1)).cpu().float().numpy().astype(np.float32)
                del out, attn_stack, attn, enc
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                print(f"[esm++] OOM on {u} (L={L}); skipping")
                if device.type == "mps": torch.mps.empty_cache()
                gc.collect(); continue
            raise

        act_idx = np.asarray([a - 1 for a in active], dtype=np.int64)
        per_res = mean_attn[act_idx, :].sum(axis=0)
        score_map = {int(p + 1): float(per_res[p]) for p in range(L)}
        auroc = per_protein_auroc(score_map, allo, active, L)
        if auroc is None: continue
        metrics[u] = {"AUROC": [auroc]}
        mask = candidate_mask(L, active)
        scores[u]  = {str(p + 1): float(per_res[p])
                      for p in range(L) if mask[p]}
        since_flush += 1
        if since_flush >= FLUSH_EVERY:
            m_path.write_text(json.dumps(metrics, indent=2))
            s_path.write_text(json.dumps(scores, indent=2))
            since_flush = 0
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    m_path.write_text(json.dumps(metrics, indent=2))
    s_path.write_text(json.dumps(scores, indent=2))
    v = [m["AUROC"][0] for m in metrics.values()]
    print(f"[esm++] done in {(time.time() - t0)/60:.1f} min; "
          f"n={len(v)} mean={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
