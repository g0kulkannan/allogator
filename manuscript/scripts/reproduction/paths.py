"""Central path config for manuscript reproduction scripts.

The directory root is taken from the ``ALLOGATOR_ROOT`` environment
variable when set (``manuscript/run_all.sh`` exports it), otherwise inferred
from this file's location.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("ALLOGATOR_ROOT",
                            Path(__file__).resolve().parents[3]))

MANUSCRIPT            = ROOT / "manuscript"
DATA                  = MANUSCRIPT / "data"
BENCHMARK             = DATA / "benchmark"
STRUCTURES            = DATA / "structures"
STRUCTURES_MULTICHAIN = DATA / "structures_multichain"
PREDICTIONS_RAW       = DATA / "predictions_raw"
CACHED                = Path(os.environ.get(
                            "ALLOGATOR_CACHE_ROOT", DATA / "cached_results"))
PYMOL_SESSIONS        = DATA / "pymol_sessions"
FIGURES               = MANUSCRIPT / "figures"

# Benchmark inputs
ACTIVE_PICKLE   = BENCHMARK / "active_residues.pickle"
ALLO_PICKLE     = BENCHMARK / "allo_residues.pickle"
SEQUENCES       = BENCHMARK / "sequences.pickle"
MANIFEST_CSV    = BENCHMARK / "manifest.csv"

# Method-specific cached output locations
def auroc_json(method: str) -> Path:
    return CACHED / method / f"per_protein_auroc.json"

def residue_scores_json(method: str) -> Path:
    return CACHED / method / f"residue_scores.json"

FIGURES.mkdir(parents=True, exist_ok=True)
