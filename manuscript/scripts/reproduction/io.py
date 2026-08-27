"""Benchmark and prediction I/O helpers."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd

from .paths import (ACTIVE_PICKLE, ALLO_PICKLE, SEQUENCES, MANIFEST_CSV,
                     CACHED, auroc_json, residue_scores_json)


def load_benchmark() -> Dict[str, Dict[str, Any]]:
    """Return a {uniprot: {"sequence", "active", "allo"}} dictionary.

    Sequences are amino-acid strings (1-indexed positions in `active`
    and `allo` correspond to characters at index pos-1).
    Active and allosteric residue sets are sets of 1-based residue
    numbers in UniProt coordinates.
    """
    with open(ACTIVE_PICKLE, "rb") as f: active = pickle.load(f)
    with open(ALLO_PICKLE, "rb")   as f: allo   = pickle.load(f)
    with open(SEQUENCES, "rb")     as f: seqs   = pickle.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for u in sorted(set(active) & set(allo) & set(seqs)):
        out[u] = {"sequence": seqs[u],
                  "active":   {int(r) for r in active[u]},
                  "allo":     {int(r) for r in allo[u]}}
    return out


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_CSV)


def load_residue_scores(method: str) -> Dict[str, Dict[int, float]]:
    """Load per-residue scores for `method` from the cached results
    directory. The on-disk JSON may be either:

      {uniprot: {position_str: score}}        # dict-of-dict format
      {uniprot: [[position, score], ...]}     # list-of-pairs format

    The returned mapping always has integer position keys and float
    values."""
    p = residue_scores_json(method)
    if not p.exists():
        raise FileNotFoundError(f"no cached residue scores for method '{method}': {p}")
    raw = json.loads(p.read_text())
    out: Dict[str, Dict[int, float]] = {}
    for u, d in raw.items():
        if isinstance(d, dict):
            out[u] = {int(k): float(v) for k, v in d.items()}
        elif isinstance(d, list):
            out[u] = {int(k): float(v) for k, v in d}
        else:
            raise TypeError(f"unrecognised residue-score entry for {u}: {type(d)}")
    return out


def load_auroc_metrics(method: str) -> Dict[str, float]:
    """Load per-protein AUROCs for `method`."""
    p = auroc_json(method)
    if not p.exists():
        raise FileNotFoundError(f"no cached AUROCs for method '{method}': {p}")
    raw = json.loads(p.read_text())
    out: Dict[str, float] = {}
    for u, v in raw.items():
        if isinstance(v, dict) and "AUROC" in v:
            val = v["AUROC"]
            if isinstance(val, list): val = val[0]
            if val is None: continue
            out[u] = float(val)
        elif v is not None:
            out[u] = float(v)
    return out


def write_auroc_metrics(method: str, aurocs: Dict[str, float]) -> Path:
    p = auroc_json(method); p.parent.mkdir(parents=True, exist_ok=True)
    payload = {u: {"AUROC": [v]} for u, v in aurocs.items()}
    p.write_text(json.dumps(payload, indent=2))
    return p


def write_residue_scores(method: str,
                          scores: Dict[str, Dict[int, float]]) -> Path:
    p = residue_scores_json(method); p.parent.mkdir(parents=True, exist_ok=True)
    payload = {u: {str(k): float(v) for k, v in d.items()}
               for u, d in scores.items()}
    p.write_text(json.dumps(payload, indent=2))
    return p
