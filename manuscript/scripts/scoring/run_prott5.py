"""Score the benchmark with ProtT5 (Rostlab/prot_t5_xl_half_uniref50-enc).

ProtT5 is a T5-encoder masked-language model trained on UniRef50.
Same mean-attention recipe as ESM-1b. Attention shape from the
Transformers model is (n_layers, n_heads, L+1, L+1) — EOS at L+1.

Output:
    manuscript/data/cached_results/ProtT5/per_protein_auroc.json
    manuscript/data/cached_results/ProtT5/residue_scores.json
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc, json, re, time

import numpy as np
import torch
from tqdm import tqdm

from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask, per_protein_auroc

METHOD = "ProtT5"
FLUSH_EVERY = 5


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    from transformers import T5Tokenizer, T5EncoderModel
    benchmark = load_benchmark()
    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "per_protein_auroc.json"
    s_path = out_dir / "residue_scores.json"
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    scores  = json.loads(s_path.read_text()) if s_path.exists() else {}
    todo = [u for u in sorted(benchmark)
            if u not in metrics or u not in scores]
    print(f"[prott5] cached {len(metrics)}; running {len(todo)}")
    if not todo: return

    device = get_device()
    print(f"[prott5] loading model on {device}")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        output_attentions=True).to(device)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    since_flush = 0; t0 = time.time()
    for u in tqdm(todo, desc="ProtT5"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        if max(active, default=0) > L or max(allo, default=0) > L: continue
        try:
            seq_clean = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
            ids = tokenizer(seq_clean, add_special_tokens=True,
                            padding=False, return_tensors="pt")
            ids = {k: v.to(device) for k, v in ids.items()}
            with torch.no_grad():
                out = model(**ids, output_attentions=True)
                attn_stack = torch.stack(out.attentions, dim=0)[:, 0]
                mean_attn = attn_stack.mean(dim=(0, 1)).cpu().numpy()
                mean_attn = mean_attn[:L, :L].astype(np.float32)
                del out, attn_stack, ids
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                print(f"[prott5] OOM on {u} (L={L}); skipping")
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

    m_path.write_text(json.dumps(metrics, indent=2))
    s_path.write_text(json.dumps(scores, indent=2))
    v = [m["AUROC"][0] for m in metrics.values()]
    print(f"[prott5] done in {(time.time() - t0)/60:.1f} min; "
          f"n={len(v)} mean={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
