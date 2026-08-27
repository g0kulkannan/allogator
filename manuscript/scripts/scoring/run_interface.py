"""Quantify ESM-1b attention bias at protein–protein interfaces.

For each multimer in the benchmark we compare per-residue ESM-1b
attention to the active site between candidate residues at the
partner-chain interface (within 5 Å of any atom of any other chain in
the renumbered multi-chain assembly) and candidate residues elsewhere.

Outputs:
    manuscript/figures/figS6_interface_bias/interface_per_protein.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

from reproduction.io import load_benchmark, load_manifest, load_residue_scores
from reproduction.paths import FIGURES, STRUCTURES_MULTICHAIN

INTERFACE_CUTOFF = 5.0
MIN_CHAIN_LEN = 10
STANDARD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
               "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL"}


def _load_chains(cif_path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    parser = MMCIFParser(QUIET=True)
    try:
        struct = parser.get_structure("s", str(cif_path))
    except Exception:
        return {}
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for model in struct:
        for chain in model:
            residues: Dict[int, np.ndarray] = {}
            for res in chain:
                if res.id[0] != " " or res.resname not in STANDARD_AA: continue
                rn = res.id[1]
                if not isinstance(rn, int) or rn <= 0 or rn > 10000: continue
                coords = np.array([a.coord for a in res.get_atoms()],
                                  dtype=np.float32)
                if coords.size:
                    residues[rn] = coords
            if len(residues) >= MIN_CHAIN_LEN:
                out[chain.id] = residues
        break
    return out


def _pick_chain(chains, target_len: int) -> str:
    return min(sorted(chains),
               key=lambda c: (abs(len(chains[c]) - target_len), c))


def _interface_set(chains, canonical: str, cutoff: float) -> set:
    if len(chains) < 2: return set()
    other_atoms = np.concatenate([
        np.concatenate(list(chains[c].values()), axis=0)
        for c in chains if c != canonical
    ], axis=0)
    cutoff_sq = cutoff ** 2
    out = set()
    for rn, atoms in chains[canonical].items():
        diffs = atoms[:, None, :] - other_atoms[None, :, :]
        if np.einsum("ijk,ijk->ij", diffs, diffs).min() <= cutoff_sq:
            out.add(rn)
    return out


def main() -> None:
    benchmark = load_benchmark()
    manifest = load_manifest()
    uni_to_pdb = dict(zip(manifest.uniprot, manifest.pdb))
    scores = load_residue_scores("ESM1b")

    rows = []
    for u in tqdm(sorted(benchmark), desc="interface"):
        if u not in scores or u not in uni_to_pdb: continue
        cif_path = STRUCTURES_MULTICHAIN / f"{uni_to_pdb[u].upper()}.cif"
        if not cif_path.exists():
            cif_path = STRUCTURES_MULTICHAIN / f"{uni_to_pdb[u].upper()}.cif.gz"
            if not cif_path.exists(): continue
        chains = _load_chains(cif_path)
        if len(chains) < 2: continue
        canon = _pick_chain(chains, len(benchmark[u]["sequence"]))
        iface = _interface_set(chains, canon, INTERFACE_CUTOFF)
        if not iface: continue
        score_dict = scores[u]
        residues = np.array(sorted(score_dict))
        vals = np.array([score_dict[r] for r in residues], dtype=np.float64)
        in_iface = np.array([r in iface for r in residues])
        if in_iface.sum() == 0 or (~in_iface).sum() == 0: continue
        rows.append({
            "uniprot": u, "pdb": uni_to_pdb[u],
            "n_chains": len(chains),
            "n_interface": len(iface),
            "n_candidates": len(residues),
            "n_iface_in_candidates": int(in_iface.sum()),
            "median_interface":     float(np.median(vals[in_iface])),
            "median_non_interface": float(np.median(vals[~in_iface])),
            "diff": float(np.median(vals[in_iface]) - np.median(vals[~in_iface])),
            "mean_interface":       float(vals[in_iface].mean()),
            "mean_non_interface":   float(vals[~in_iface].mean()),
            "mean_diff":   float(vals[in_iface].mean() - vals[~in_iface].mean()),
            "n_allo": len(benchmark[u]["allo"]),
            "n_allo_at_interface": len({int(a) for a in benchmark[u]["allo"]} & iface),
        })

    df = pd.DataFrame(rows)
    out_path = FIGURES / "figS6_interface_bias" / "interface_per_protein.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[interface] analysed {len(df)} multimers; saved {out_path}")
    if len(df):
        higher = int((df["mean_diff"] > 0).sum())
        print(f"  interface > non-interface mean attention in {higher}/{len(df)} proteins")


if __name__ == "__main__":
    main()
