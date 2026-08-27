"""Score the benchmark with Ohm allosteric coupling intensities.

Ohm is run externally via the web server at
http://drugdiscovery.utah.edu/Ohm.html. The server emits one ACI value
per residue, in the same order the residues appear in the input PDB.
This script:

  1. Uses the representative PDB listed for each protein in
     ``manuscript/data/benchmark/manifest.csv`` and reads the corresponding Ohm output
     and stripped PDB file.
  2. Builds (residue_number, chain, score) tuples in PDB order.
  3. Retains only chains that map to the benchmark UniProt sequence.
     Homologous protomer copies are retained; unrelated partner chains are
     excluded from benchmark labels and scores.
  4. De-duplicates by (residue_number, chain), keeping the max score.
  5. Imputes residues unresolved in the PDB with worst_observed_score - 1
     so they rank below resolved residues.
  6. Drops residues within ±1 of any active-site residue (same adjacency
     rule as the language-model methods).
  7. Computes per-protein AUROC against the allosteric residue set,
     with a 1000-iteration permutation p-value.

The exact residue-chain instances used for AUROC calculation are written to
``chain_residue_scores.csv.gz``. Their observation order is retained so the
seeded permutation p-values can be recalculated exactly. If raw Ohm/PDB inputs
are absent, the script recalculates the metrics from this included table.
``residue_scores.json`` provides one value per UniProt position by taking the
maximum across mapped protein copies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from reproduction.io import load_benchmark, load_manifest
from reproduction.paths import CACHED, STRUCTURES, PREDICTIONS_RAW

METHOD = "Ohm"
OHM_DIR = PREDICTIONS_RAW / "ohm"
N_PERM = 1000
SEED = 42
MIN_CHAIN_OVERLAP = 10
MIN_CHAIN_IDENTITY = 0.80

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _first_model_lines(pdb_lines):
    """Return the first coordinate model from an NMR PDB, or all lines."""
    model_start = next(
        (index for index, line in enumerate(pdb_lines) if line.startswith("MODEL")),
        None,
    )
    if model_start is None:
        return pdb_lines
    model_end = next(
        (
            index
            for index, line in enumerate(pdb_lines[model_start + 1 :], model_start + 1)
            if line.startswith("ENDMDL")
        ),
        len(pdb_lines),
    )
    return pdb_lines[model_start + 1 : model_end]


def _parse_pdb_residue_order(pdb_lines):
    out, last = [], None
    for line in pdb_lines:
        head = line[0:4]
        if head not in ("ATOM", "HETA", "  AT"): continue
        if head == "HETA" and line[17:20] == "HOH": continue
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        chain = line[21]
        if (resi, chain) != last:
            out.append((resi, chain, head.strip()))
            last = (resi, chain)
    return out


def _target_chains(pdb_lines, sequence):
    """Return chains whose numbered residues map to the UniProt sequence."""
    chain_residues = {}
    seen = set()
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        try:
            position = int(line[22:26])
        except ValueError:
            continue
        chain = line[21]
        insertion_code = line[26]
        residue_name = line[17:20].strip()
        key = (chain, position, insertion_code)
        if key in seen or residue_name not in AA3_TO_AA1:
            continue
        seen.add(key)
        chain_residues.setdefault(chain, []).append(
            (position, AA3_TO_AA1[residue_name])
        )

    targets = set()
    for chain, residues in chain_residues.items():
        overlap = [(p, aa) for p, aa in residues if 1 <= p <= len(sequence)]
        if len(overlap) < MIN_CHAIN_OVERLAP:
            continue
        identity = sum(sequence[p - 1] == aa for p, aa in overlap) / len(overlap)
        if identity >= MIN_CHAIN_IDENTITY:
            targets.add(chain)
    return targets


def _dedup_max(triples):
    keep: dict[tuple[int, str], tuple[int, str, float]] = {}
    for r, c, s in triples:
        key = (r, c)
        if key not in keep or s > keep[key][2]:
            keep[key] = (r, c, s)
    return list(keep.values())


def _remove_adjacent(residues, active):
    return [t for t in residues
            if all(abs(t[0] - a) >= 2 for a in active)]


def _auroc_with_perm(triples, allo, n_perm=N_PERM, seed=SEED):
    y = np.array([1 if t[0] in allo else 0 for t in triples], dtype=int)
    s = np.array([t[2] for t in triples], dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None, None
    auroc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    yt = y.copy()
    perms = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(yt); perms[i] = roc_auc_score(yt, s)
    p = float((perms >= auroc).sum() / n_perm)
    return auroc, p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohm-dir", type=Path, default=OHM_DIR)
    parser.add_argument("--structure-dir", type=Path, default=STRUCTURES)
    parser.add_argument("--output-dir", type=Path, default=CACHED / METHOD)
    parser.add_argument(
        "--chain-table",
        type=Path,
        default=None,
        help="use an existing chain_residue_scores.csv.gz table",
    )
    return parser.parse_args()


def _score_chain_table(
    frame: pd.DataFrame,
    benchmark: dict,
) -> tuple[dict, dict]:
    required = {
        "uniprot",
        "pdb",
        "observation_index",
        "residue_number",
        "aci_score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ohm chain table is missing columns: {sorted(missing)}")

    metrics: dict = {}
    scores: dict = {}
    for uniprot, group in frame.groupby("uniprot", sort=True):
        if uniprot not in benchmark:
            raise ValueError(f"Ohm chain table contains unknown protein {uniprot}")
        group = group.sort_values("observation_index")
        residues = group["residue_number"].astype(int).to_numpy()
        values = group["aci_score"].astype(float).to_numpy()
        triples = [
            (int(residue), "", float(score))
            for residue, score in zip(residues, values)
        ]
        auroc, p_value = _auroc_with_perm(triples, benchmark[uniprot]["allo"])
        if auroc is None:
            raise ValueError(f"Degenerate Ohm labels for {uniprot}")
        pdb_values = group["pdb"].astype(str).unique()
        if len(pdb_values) != 1:
            raise ValueError(f"Ohm chain table has multiple PDBs for {uniprot}")
        metrics[uniprot] = {
            "AUROC": [auroc, p_value],
            "pdb": str(pdb_values[0]).upper(),
        }
        position_scores: dict[int, float] = {}
        for residue, score in zip(residues, values):
            position_scores[int(residue)] = max(
                position_scores.get(int(residue), float("-inf")), float(score)
            )
        scores[uniprot] = [
            [position, position_scores[position]]
            for position in sorted(position_scores)
        ]
    return metrics, scores


def main() -> None:
    args = _parse_args()
    benchmark = load_benchmark()
    manifest = load_manifest()
    selected_pdb = {
        str(row.uniprot): str(row.pdb).upper()
        for row in manifest.itertuples()
    }
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_table = args.chain_table or out_dir / "chain_residue_scores.csv.gz"
    aci_paths = sorted(args.ohm_dir.glob("*.txt"))
    if not aci_paths:
        if not chain_table.exists():
            raise SystemExit(
                f"No Ohm outputs found in {args.ohm_dir} and no chain table "
                f"found at {chain_table}"
            )
        chain_frame = pd.read_csv(chain_table, keep_default_na=False)
        metrics, scores = _score_chain_table(chain_frame, benchmark)
        (out_dir / "per_protein_auroc.json").write_text(
            json.dumps(metrics, indent=2)
        )
        (out_dir / "residue_scores.json").write_text(json.dumps(scores, indent=2))
        values = [item["AUROC"][0] for item in metrics.values()]
        significant = sum(
            item["AUROC"][1] is not None and item["AUROC"][1] < 0.05
            for item in metrics.values()
        )
        print(
            f"[ohm] n={len(values)} mean AUROC={np.mean(values):.4f} "
            f"sig={significant}/{len(values)} at p<0.05"
        )
        return

    metrics: dict = {}
    scores: dict = {}
    skipped = []
    chain_rows: list[dict] = []

    for aci_path in tqdm(aci_paths, desc="Ohm"):
        title = aci_path.stem
        if "_" not in title:
            skipped.append(f"{title}: bad filename"); continue
        uniprot, pdb = title.rsplit("_", 1)
        if uniprot not in benchmark:
            skipped.append(f"{title}: {uniprot} not in benchmark"); continue
        if pdb.upper() != selected_pdb.get(uniprot):
            continue
        pdb_path = args.structure_dir / f"{title}.pdb"
        if not pdb_path.exists():
            skipped.append(f"{title}: missing structure {pdb_path.name}"); continue

        entry = benchmark[uniprot]
        active = entry["active"]; allo = entry["allo"]
        seq_len = len(entry["sequence"])

        pdb_lines = _first_model_lines(pdb_path.read_text().splitlines())
        aci_lines = aci_path.read_text().splitlines()
        residues = _parse_pdb_residue_order(pdb_lines)
        if len(residues) != len(aci_lines):
            skipped.append(
                f"{title}: {len(aci_lines)} ACI values for "
                f"{len(residues)} PDB residues"
            )
            continue
        target_chains = _target_chains(pdb_lines, entry["sequence"])
        if not target_chains:
            skipped.append(f"{title}: no chain maps to {uniprot}"); continue
        triples = []
        for (r, c, _), score in zip(residues, aci_lines, strict=True):
            if c in target_chains:
                triples.append((r, c, float(score.strip())))
        triples = _dedup_max(triples)
        triples = [t for t in triples if t[0] < 9500]

        # Impute unresolved residues with worst_observed - 1
        resolved = {t[0] for t in triples}
        missing = set(range(1, seq_len + 1)) - resolved
        if missing and triples:
            worst = min(t[2] for t in triples) - 1.0
            triples += [(r, None, worst) for r in missing]

        triples = _remove_adjacent(triples, active)
        auroc, p = _auroc_with_perm(triples, allo)
        if auroc is None:
            skipped.append(f"{title}: degenerate y"); continue

        if uniprot in metrics:
            raise ValueError(f"Duplicate Ohm input for {uniprot} and PDB {pdb}")
        rec = {"AUROC": [auroc, p], "pdb": pdb.upper()}
        metrics[uniprot] = rec
        for observation_index, (residue, chain, score) in enumerate(triples, 1):
            chain_rows.append(
                {
                    "uniprot": uniprot,
                    "pdb": pdb.upper(),
                    "observation_index": observation_index,
                    "residue_number": int(residue),
                    "chain_id": "" if chain is None else chain,
                    "aci_score": float(score),
                    "is_imputed": chain is None,
                }
            )

        position_scores: dict[int, float] = {}
        for residue, _chain, score in triples:
            position_scores[int(residue)] = max(
                position_scores.get(int(residue), float("-inf")), float(score)
            )
        scores[uniprot] = [
            [position, position_scores[position]]
            for position in sorted(position_scores)
        ]

    if not metrics:
        raise SystemExit("No selected Ohm inputs could be scored")
    (out_dir / "per_protein_auroc.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "residue_scores.json").write_text(json.dumps(scores, indent=2))
    pd.DataFrame(chain_rows).to_csv(
        out_dir / "chain_residue_scores.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    if skipped:
        print(f"[ohm] skipped {len(skipped)} inputs")
    if metrics:
        v = [m["AUROC"][0] for m in metrics.values()]
        n_sig = sum(1 for m in metrics.values() if m["AUROC"][1] is not None and m["AUROC"][1] < 0.05)
        print(f"[ohm] n={len(v)} mean AUROC={np.mean(v):.4f} "
              f"sig={n_sig}/{len(v)} at p<0.05")


if __name__ == "__main__":
    main()
