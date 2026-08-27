"""Export an inspectable residue-level trace for one benchmark protein.

For Ohm, the trace maps every original ACI value to the corresponding PDB
residue and chain.  For EVcouplings, it reports the active-site pair scores
that contribute to each residue score.  For the sequence-model and baseline
methods, it reports the released per-residue score together with the benchmark
labels used for evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.io import load_benchmark, load_manifest, load_residue_scores
from reproduction.paths import CACHED, PREDICTIONS_RAW, STRUCTURES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protein", required=True, help="UniProt accession")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path (default: write CSV to standard output)",
    )
    return parser.parse_args()


def _labels(entry: dict, positions: list[int]) -> pd.DataFrame:
    sequence = entry["sequence"]
    active = {int(x) for x in entry["active"]}
    allosteric = {int(x) for x in entry["allo"]}
    return pd.DataFrame(
        {
            "residue_number": positions,
            "amino_acid": [sequence[p - 1] for p in positions],
            "is_active_site": [p in active for p in positions],
            "is_sequence_adjacent": [
                p not in active and any(abs(p - a) == 1 for a in active)
                for p in positions
            ],
            "is_allosteric": [p in allosteric for p in positions],
        }
    )


def _cached_trace(method: str, protein: str, entry: dict) -> pd.DataFrame:
    scores = load_residue_scores(method)
    if protein not in scores:
        raise SystemExit(f"{protein} is not present in the {method} score cache")
    positions = sorted(scores[protein])
    frame = _labels(entry, positions)
    frame.insert(0, "uniprot", protein)
    frame.insert(1, "method", method)
    frame["score"] = [scores[protein][p] for p in positions]
    frame["descending_rank"] = frame["score"].rank(
        ascending=False, method="average"
    )
    return frame


def _ohm_trace(protein: str, entry: dict, pdb: str) -> pd.DataFrame:
    # Import from the scoring implementation so inspection and transformation
    # use exactly the same PDB-order and target-chain rules.
    from run_ohm import (
        _first_model_lines,
        _parse_pdb_residue_order,
        _target_chains,
    )

    stem = f"{protein}_{pdb}"
    aci_path = PREDICTIONS_RAW / "ohm" / f"{stem}.txt"
    pdb_path = STRUCTURES / f"{stem}.pdb"
    if not aci_path.exists() or not pdb_path.exists():
        raise SystemExit(f"raw Ohm input pair is unavailable for {stem}")
    pdb_lines = _first_model_lines(pdb_path.read_text().splitlines())
    residues = _parse_pdb_residue_order(pdb_lines)
    values = [float(x) for x in aci_path.read_text().splitlines()]
    if len(residues) != len(values):
        raise SystemExit(
            f"{stem}: {len(values)} ACI values for {len(residues)} PDB residues"
        )
    target_chains = _target_chains(pdb_lines, entry["sequence"])
    cached = load_residue_scores("Ohm").get(protein, {})
    active = {int(x) for x in entry["active"]}
    allosteric = {int(x) for x in entry["allo"]}
    rows = []
    for index, ((position, chain, record_type), score) in enumerate(
        zip(residues, values), 1
    ):
        candidate = (
            chain in target_chains
            and position < 9500
            and all(abs(position - active_position) >= 2 for active_position in active)
        )
        rows.append(
            {
                "uniprot": protein,
                "pdb": pdb,
                "raw_observation_index": index,
                "residue_number": int(position),
                "chain_id": chain,
                "pdb_record_type": record_type,
                "raw_aci_score": float(score),
                "maps_to_target_sequence": chain in target_chains,
                "is_candidate_observation": candidate,
                "is_allosteric": int(position) in allosteric,
                "released_position_score": cached.get(int(position), np.nan),
            }
        )
    return pd.DataFrame(rows)


def _evcouplings_trace(protein: str, entry: dict) -> pd.DataFrame:
    raw_path = PREDICTIONS_RAW / "evcouplings" / f"{protein}.csv.gz"
    if not raw_path.exists():
        raw_path = raw_path.with_suffix("")
    if not raw_path.exists():
        raise SystemExit(f"raw EVcouplings table is unavailable for {protein}")
    pairs = pd.read_csv(raw_path, usecols=["i", "j", "cn"])
    length = len(entry["sequence"])
    active = sorted(int(x) for x in entry["active"])
    table = pairs.pivot(index="i", columns="j", values="cn")
    table = table.fillna(table.T)
    table = table.reindex(index=range(1, length + 1), columns=range(1, length + 1))
    minimum = float(np.nanmin(table.to_numpy()))
    imputed_value = minimum - 10.0
    raw_present = table.notna()
    filled = table.fillna(imputed_value)
    cached = load_residue_scores("EVcouplings")[protein]
    candidates = [
        position
        for position in range(1, length + 1)
        if all(abs(position - active_position) >= 2 for active_position in active)
    ]
    frame = _labels(entry, candidates)
    frame.insert(0, "uniprot", protein)
    frame.insert(1, "method", "EVcouplings")
    frame["raw_active_site_pairs"] = [
        int(raw_present.loc[position, active].sum()) for position in candidates
    ]
    frame["imputed_active_site_pairs"] = [
        len(active) - int(raw_present.loc[position, active].sum())
        for position in candidates
    ]
    frame["imputed_pair_value"] = imputed_value
    frame["calculated_score"] = [
        float(filled.loc[position, active].sum()) for position in candidates
    ]
    frame["released_score"] = [cached[position] for position in candidates]
    frame["matches_released_score"] = np.isclose(
        frame["calculated_score"], frame["released_score"], rtol=0, atol=1e-12
    )
    return frame


def main() -> None:
    args = _parse_args()
    benchmark = load_benchmark()
    protein = args.protein.strip()
    if protein not in benchmark:
        raise SystemExit(f"unknown benchmark protein: {protein}")
    method = args.method.strip()
    if method == "Ohm":
        manifest = load_manifest().set_index("uniprot")
        frame = _ohm_trace(protein, benchmark[protein], str(manifest.loc[protein, "pdb"]))
    elif method == "EVcouplings":
        frame = _evcouplings_trace(protein, benchmark[protein])
    else:
        if not (CACHED / method / "residue_scores.json").exists():
            available = sorted(
                path.parent.name for path in CACHED.glob("*/residue_scores.json")
            )
            raise SystemExit(
                f"unknown method {method!r}; available methods: {', '.join(available)}"
            )
        frame = _cached_trace(method, protein, benchmark[protein])
    if args.output is None:
        frame.to_csv(sys.stdout, index=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
