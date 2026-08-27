"""Multi-chain structure parsing for the 3D-distance baseline and
interface analysis.

The 109-protein benchmark uses biological assemblies renumbered to
UniProt coordinates with PDBrenum. We iterate every protein chain that
matches the UniProt accession in question and return a list of Cα
coordinates per residue number, so a residue with multiple protomer
copies gets multiple coordinate entries.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import gzip
import numpy as np


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_cif_ca(cif_path: Path) -> Dict[int, List[np.ndarray]]:
    """Parse a PDBrenum-renumbered CIF and return, for every protein
    residue, a list of Cα coordinate arrays (one per chain copy).

    Hetero atoms and non-Cα atoms are ignored. Residue numbers come from
    the auth_seq_id column (which PDBrenum sets to the UniProt number).
    """
    out: Dict[int, List[np.ndarray]] = {}
    in_atom = False; cols: List[str] = []; idx: Dict[str, int] = {}
    with _open(cif_path) as f:
        for line in f:
            if line.startswith("loop_"):
                in_atom = False; cols = []; idx = {}; continue
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".")[-1]); continue
            if cols and not in_atom and line.startswith(("ATOM", "HETATM")):
                idx = {c: i for i, c in enumerate(cols)}
                in_atom = True
            if not in_atom:
                continue
            if line.startswith("#") or line.strip() == "":
                in_atom = False; continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            tok = line.split()
            if len(tok) <= max(idx.values()): continue
            if tok[idx["group_PDB"]] != "ATOM": continue
            if tok[idx["label_atom_id"]].strip('"').strip("'") != "CA": continue
            try:
                resn = int(tok[idx["auth_seq_id"]])
                x = float(tok[idx["Cartn_x"]])
                y = float(tok[idx["Cartn_y"]])
                z = float(tok[idx["Cartn_z"]])
            except (KeyError, ValueError):
                continue
            out.setdefault(resn, []).append(np.array([x, y, z]))
    return out


def parse_pdb_ca(pdb_path: Path) -> Dict[int, List[np.ndarray]]:
    """Same as `parse_cif_ca` but for a PDB-format file."""
    out: Dict[int, List[np.ndarray]] = {}
    with _open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                resn = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            out.setdefault(resn, []).append(np.array([x, y, z]))
    return out


def parse_ca(path: Path) -> Dict[int, List[np.ndarray]]:
    """Dispatch on suffix."""
    s = str(path).lower()
    if s.endswith(".cif") or s.endswith(".cif.gz"):
        return parse_cif_ca(path)
    return parse_pdb_ca(path)


def min_ca_distance_to_set(ca: Dict[int, List[np.ndarray]],
                            target_residues: List[int],
                            query_residues: List[int]
                            ) -> Dict[int, float]:
    """For each residue in `query_residues`, return the minimum Cα–Cα
    distance to any residue in `target_residues`, considering all
    protomer copies. Returns NaN when the query residue is unresolved.
    """
    targets = [c for r in target_residues for c in ca.get(int(r), [])]
    if not targets: return {int(q): float("nan") for q in query_residues}
    targ = np.stack(targets, axis=0)
    out: Dict[int, float] = {}
    for q in query_residues:
        cs = ca.get(int(q), [])
        if not cs:
            out[int(q)] = float("nan"); continue
        qarr = np.stack(cs, axis=0)
        d = np.linalg.norm(qarr[:, None, :] - targ[None, :, :], axis=-1)
        out[int(q)] = float(d.min())
    return out
