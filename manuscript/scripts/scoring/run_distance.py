"""3D-distance baseline.

For each candidate residue, the predictor score is the negative of the
minimum Cα–Cα distance to any active-site Cα across every protomer in
the biological assembly. Residues unresolved in the structure are
imputed with the per-protein median score so they rank neutrally.

Input structures: PDBrenum-renumbered multi-chain CIFs in
``manuscript/data/structures_multichain/``.

Output:
    manuscript/data/cached_results/Distance/per_protein_auroc.json
    manuscript/data/cached_results/Distance/residue_scores.json
    manuscript/data/cached_results/Distance/per_protein_metrics.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from reproduction.io import load_benchmark, load_manifest
from reproduction.paths import CACHED, STRUCTURES_MULTICHAIN
from reproduction.scoring import candidate_mask, per_protein_auroc
from reproduction.structure import parse_ca, min_ca_distance_to_set

METHOD = "Distance"


def _cif_for(pdb_code: str) -> Path | None:
    for suffix in (".cif", ".cif.gz"):
        p = STRUCTURES_MULTICHAIN / f"{pdb_code.upper()}{suffix}"
        if p.exists():
            return p
    return None


def main() -> None:
    benchmark = load_benchmark()
    manifest = load_manifest()
    uni_to_pdb = dict(zip(manifest.uniprot, manifest.pdb))

    out_dir = CACHED / METHOD; out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "per_protein_auroc.json"
    s_path = out_dir / "residue_scores.json"
    metrics: dict = {}
    scores: dict = {}
    rows = []
    skipped = []

    for u in tqdm(sorted(benchmark), desc="distance"):
        entry = benchmark[u]
        seq = entry["sequence"]; active = sorted(entry["active"])
        allo = entry["allo"]; L = len(seq)
        pdb = uni_to_pdb.get(u)
        if not pdb:
            skipped.append(f"{u}: no PDB in manifest"); continue
        cif = _cif_for(pdb)
        if cif is None:
            skipped.append(f"{u}: missing CIF for {pdb}"); continue
        ca = parse_ca(cif)
        if not ca:
            skipped.append(f"{u}: parse failed for {pdb}"); continue
        active_in_cif = [a for a in active if a in ca]
        if not active_in_cif:
            skipped.append(f"{u}: no active residues in CIF"); continue

        mask = candidate_mask(L, active)
        cand_positions = [int(p + 1) for p in range(L) if mask[p]]
        dists = min_ca_distance_to_set(ca, active_in_cif, cand_positions)
        # negative distance ⇒ closer-is-better; impute missing with median
        finite = np.array([v for v in dists.values() if np.isfinite(v)])
        if finite.size == 0:
            skipped.append(f"{u}: no usable distances"); continue
        median_d = float(np.median(finite))
        score_map = {p: -(d if np.isfinite(d) else median_d)
                     for p, d in dists.items()}

        auroc = per_protein_auroc(score_map, allo, active, L)
        if auroc is None:
            continue
        metrics[u] = {"AUROC": [auroc]}
        scores[u]  = {str(p): float(score_map[p]) for p in cand_positions}

        # Per-protein summary numbers
        allo_in_cif = [r for r in allo if r in ca]
        if allo_in_cif:
            allo_d = np.array(list(min_ca_distance_to_set(
                ca, active_in_cif, allo_in_cif).values()))
            allo_d = allo_d[np.isfinite(allo_d)]
            rows.append({"uniprot": u, "pdb": pdb.upper(),
                         "n_active": len(active), "n_active_in_cif": len(active_in_cif),
                         "n_allo": len(allo), "n_allo_in_cif": len(allo_in_cif),
                         "mean_allo_dist": float(allo_d.mean()) if allo_d.size else np.nan,
                         "median_allo_dist": float(np.median(allo_d)) if allo_d.size else np.nan,
                         "min_allo_dist": float(allo_d.min()) if allo_d.size else np.nan,
                         "max_allo_dist": float(allo_d.max()) if allo_d.size else np.nan,
                         "auroc_distance": auroc})

    m_path.write_text(json.dumps(metrics, indent=2))
    s_path.write_text(json.dumps(scores, indent=2))
    if skipped:
        (out_dir / "skipped.txt").write_text("\n".join(skipped))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_protein_metrics.csv", index=False)
    v = [m["AUROC"][0] for m in metrics.values()]
    print(f"[distance] n={len(v)} mean AUROC={np.mean(v):.4f} median={np.median(v):.4f}")
    if skipped:
        print(f"[distance] skipped {len(skipped)} (see {out_dir/'skipped.txt'})")


if __name__ == "__main__":
    main()
