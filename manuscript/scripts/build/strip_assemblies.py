"""Strip the chosen reference assemblies down to a single protein chain
with HETATMs (modulators, waters, cofactors) removed, for Ohm scoring.

Input:  manuscript/data/structures_multichain/<PDB>.cif  (renumbered, multi-chain)
Output: manuscript/data/structures/<UniProt>_<PDB>.pdb   (single-chain, HETATM-free)

Selects the chain whose length is closest to the UniProt canonical
sequence; ties broken alphabetically.

The generated PDB files are the structure inputs for the Ohm scoring
pipeline.
"""
from __future__ import annotations
import gzip
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO, Select

from reproduction.io import load_benchmark, load_manifest
from reproduction.paths import STRUCTURES, STRUCTURES_MULTICHAIN


class _ProteinOnly(Select):
    def __init__(self, chain_id: str):
        super().__init__(); self.chain_id = chain_id
    def accept_chain(self, chain): return chain.id == self.chain_id
    def accept_residue(self, residue): return residue.id[0] == " "


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix.endswith(".gz") else open(path)


def _pick_chain(cif_path: Path, target_len: int) -> str | None:
    parser = MMCIFParser(QUIET=True)
    try:
        struct = parser.get_structure("s", str(cif_path))
    except Exception:
        return None
    candidates = []
    for chain in struct[0]:
        n_res = sum(1 for r in chain if r.id[0] == " ")
        if n_res >= 10:
            candidates.append((chain.id, n_res))
    if not candidates:
        return None
    return min(candidates,
               key=lambda c: (abs(c[1] - target_len), c[0]))[0]


def main() -> None:
    benchmark = load_benchmark()
    manifest  = load_manifest()
    uni_to_pdb = dict(zip(manifest.uniprot, manifest.pdb))
    STRUCTURES.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = 0
    for u, entry in benchmark.items():
        pdb = uni_to_pdb.get(u)
        if not pdb: continue
        cif_path = STRUCTURES_MULTICHAIN / f"{pdb.upper()}.cif"
        if not cif_path.exists():
            cif_path = STRUCTURES_MULTICHAIN / f"{pdb.upper()}.cif.gz"
            if not cif_path.exists():
                n_skip += 1; continue
        out = STRUCTURES / f"{u}_{pdb}.pdb"
        if out.exists():
            n_done += 1; continue
        chain_id = _pick_chain(cif_path, len(entry["sequence"]))
        if chain_id is None:
            n_skip += 1; continue
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure("s", str(cif_path))
        io = PDBIO(); io.set_structure(struct)
        io.save(str(out), _ProteinOnly(chain_id))
        n_done += 1
    print(f"wrote {n_done} stripped PDBs to {STRUCTURES}/  (skipped {n_skip})")


if __name__ == "__main__":
    main()
