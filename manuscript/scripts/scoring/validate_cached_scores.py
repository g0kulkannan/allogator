"""Check that released residue scores reproduce every released AUROC.

This is a lightweight score-level check: it does not load language models.
It also verifies coverage of the included Ohm and EVcouplings source outputs.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reproduction.io import (
    load_auroc_metrics,
    load_benchmark,
    load_residue_scores,
)
from reproduction.paths import CACHED, PREDICTIONS_RAW, STRUCTURES
from reproduction.scoring import per_protein_auroc

EXPECTED_SCORE_COUNTS = {
    "Distance": 100,
    "ESM1b": 109,
    "ESM1b_contacts": 109,
    "ESM1b_max_max": 109,
    "ESM1b_max_sum": 109,
    "ESM1b_max_sym_sum": 109,
    "ESM1b_mean_max": 109,
    "ESM1b_median_max": 109,
    "ESM1b_median_sum": 109,
    "ESM1b_min_max": 109,
    "ESM1b_min_sum": 109,
    "ESM2_8M": 109,
    "ESM2_35M": 109,
    "ESM2_150M": 109,
    "ESM2_650M": 109,
    "ESM2_3B": 109,
    "ESM2_15B": 109,
    "ESMpp": 109,
    "EVcouplings": 109,
    "Ohm": 100,
    "ProtT5": 109,
    "Random": 109,
}


def _validate_standard(method: str, benchmark: dict) -> tuple[int, float]:
    released_scores = load_residue_scores(method)
    released_aurocs = load_auroc_metrics(method)
    if set(released_scores) != set(released_aurocs):
        raise ValueError("residue-score and AUROC protein sets differ")
    expected_count = EXPECTED_SCORE_COUNTS[method]
    if len(released_scores) != expected_count:
        raise ValueError(
            f"expected {expected_count} proteins, found {len(released_scores)}"
        )
    differences = []
    for protein, released in released_aurocs.items():
        entry = benchmark[protein]
        calculated = per_protein_auroc(
            released_scores[protein],
            entry["allo"],
            entry["active"],
            len(entry["sequence"]),
        )
        if calculated is None:
            raise ValueError(f"{method}/{protein}: AUROC is undefined")
        differences.append(abs(calculated - released))
    return len(released_aurocs), max(differences, default=0.0)


def _validate_ohm(benchmark: dict) -> tuple[int, float, float]:
    chain_path = CACHED / "Ohm" / "chain_residue_scores.csv.gz"
    frame = pd.read_csv(chain_path, keep_default_na=False)
    released_scores = load_residue_scores("Ohm")
    released_aurocs = load_auroc_metrics("Ohm")
    if set(released_scores) != set(released_aurocs):
        raise ValueError("Ohm residue-score and AUROC protein sets differ")
    if len(released_scores) != EXPECTED_SCORE_COUNTS["Ohm"]:
        raise ValueError(
            f"expected {EXPECTED_SCORE_COUNTS['Ohm']} Ohm proteins, "
            f"found {len(released_scores)}"
        )
    auroc_differences = []
    score_differences = []
    for protein, group in frame.groupby("uniprot", sort=True):
        group = group.sort_values("observation_index")
        labels = group["residue_number"].astype(int).isin(
            benchmark[protein]["allo"]
        ).astype(int)
        calculated = float(roc_auc_score(labels, group["aci_score"].astype(float)))
        auroc_differences.append(abs(calculated - released_aurocs[protein]))
        position_max = group.groupby("residue_number")["aci_score"].max()
        for position, score in released_scores[protein].items():
            score_differences.append(abs(float(position_max.loc[position]) - score))
    return (
        len(released_aurocs),
        max(auroc_differences, default=0.0),
        max(score_differences, default=0.0),
    )


def _check_raw_inputs(benchmark: dict) -> list[str]:
    errors = []
    expected = set(benchmark)
    evcouplings = {
        path.name.removesuffix(".csv.gz").removesuffix(".csv")
        for path in (PREDICTIONS_RAW / "evcouplings").glob("*.csv*")
    }
    if evcouplings != expected:
        errors.append(
            f"EVcouplings source tables: expected {len(expected)}, found "
            f"{len(evcouplings)}"
        )
    ohm_stems = {path.stem for path in (PREDICTIONS_RAW / "ohm").glob("*.txt")}
    pdb_stems = {path.stem for path in STRUCTURES.glob("*.pdb")}
    if len(ohm_stems) != EXPECTED_SCORE_COUNTS["Ohm"]:
        errors.append(
            f"Ohm source outputs: expected {EXPECTED_SCORE_COUNTS['Ohm']}, "
            f"found {len(ohm_stems)}"
        )
    missing_pairs = sorted(ohm_stems - pdb_stems)
    if missing_pairs:
        errors.append(f"Ohm outputs without paired PDBs: {missing_pairs}")
    chain_proteins = set(
        pd.read_csv(
            CACHED / "Ohm" / "chain_residue_scores.csv.gz",
            usecols=["uniprot"],
        )["uniprot"]
    )
    ohm_proteins = {stem.rsplit("_", 1)[0] for stem in ohm_stems}
    if chain_proteins != ohm_proteins:
        errors.append(
            "Ohm mapped-residue table does not cover the same proteins as "
            "the source outputs"
        )
    return errors


def main() -> None:
    benchmark = load_benchmark()
    failures = []
    methods = sorted(EXPECTED_SCORE_COUNTS)
    for method in methods:
        try:
            for filename in ("residue_scores.json", "per_protein_auroc.json"):
                if not (CACHED / method / filename).is_file():
                    raise FileNotFoundError(f"missing {method}/{filename}")
            if method == "Ohm":
                n, auroc_difference, score_difference = _validate_ohm(benchmark)
                print(
                    f"{method:22s} n={n:3d}  max AUROC difference="
                    f"{auroc_difference:.3g}  max score difference={score_difference:.3g}"
                )
                if auroc_difference > 1e-12 or score_difference > 1e-12:
                    failures.append(method)
            else:
                n, difference = _validate_standard(method, benchmark)
                print(
                    f"{method:22s} n={n:3d}  max AUROC difference={difference:.3g}"
                )
                if difference > 1e-12:
                    failures.append(method)
        except Exception as error:
            print(f"{method:22s} ERROR: {error}")
            failures.append(method)

    layer = pd.read_csv(CACHED / "PerLayer" / "per_layer_auroc.csv", index_col=0)
    head = pd.read_csv(CACHED / "PerLayer" / "per_head_auroc.csv", index_col=0)
    print(f"{'PerLayer':22s} layer table={layer.shape}  head table={head.shape}")
    if layer.shape != (109, 33) or head.shape != (109, 660):
        failures.append("PerLayer")

    failures.extend(_check_raw_inputs(benchmark))
    if failures:
        print("\nscore validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nAll released score caches reproduce their released AUROCs.")


if __name__ == "__main__":
    main()
