"""Pool per-residue attention rank vs. 3D distance to the active site.

For every candidate residue in the benchmark we record:
  - within-protein attention rank percentile (0 = lowest, 1 = highest)
  - minimum Cα–Cα distance to any active-site Cα across every protomer
    of the multi-chain assembly
  - allosteric label (1 if the residue is in the labelled allosteric
    set, 0 otherwise)

This long-format CSV feeds Fig. S5 and the per-protein allosteric
distance plot.

Output:
    manuscript/figures/figS5_distance_vs_attention/attention_rank_vs_distance.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

from reproduction.io import load_benchmark, load_manifest, load_residue_scores
from reproduction.paths import FIGURES, STRUCTURES_MULTICHAIN
from reproduction.structure import parse_ca, min_ca_distance_to_set


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
    scores = load_residue_scores("ESM1b")

    rows = []
    for u in tqdm(sorted(benchmark), desc="rank vs dist"):
        if u not in scores: continue
        cif = _cif_for(uni_to_pdb.get(u, ""))
        if cif is None: continue
        ca = parse_ca(cif)
        if not ca: continue
        active = sorted(set(benchmark[u]["active"]) & set(ca))
        if not active: continue
        score_dict = scores[u]                     # {resnum: score}
        cand = sorted(int(r) for r in score_dict)
        score_arr = np.array([score_dict[r] for r in cand], dtype=np.float64)
        ranks = rankdata(score_arr, method="average")
        pct = (ranks - 1) / max(len(ranks) - 1, 1)
        dist = min_ca_distance_to_set(ca, active, cand)
        allo_set = {int(a) for a in benchmark[u]["allo"]}
        for r, p, d in zip(cand, pct, [dist[c] for c in cand]):
            if not np.isfinite(d): continue
            rows.append({"uniprot": u, "residue": int(r),
                         "attention_rank_pct": float(p),
                         "distance_3D": float(d),
                         "allo": int(r in allo_set)})

    df = pd.DataFrame(rows)
    out_path = FIGURES / "figS5_distance_vs_attention" / "attention_rank_vs_distance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_allo = int(df["allo"].sum())
    print(f"[rank-dist] {len(df)} residues across {df.uniprot.nunique()} "
          f"proteins; {n_allo} allosteric")


if __name__ == "__main__":
    main()
