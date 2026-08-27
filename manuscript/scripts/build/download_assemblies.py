"""Download the multi-chain biological assemblies needed by the 3D-
distance baseline and the interface analysis.

Reads ``manuscript/data/benchmark/manifest.csv`` to get the list of PDB codes, then
fetches the assembly mmCIF from RCSB. The downloaded files have RCSB
author numbering — they must be renumbered to UniProt coordinates with
``PDBrenum`` before they can be used by the scoring scripts:

    pip install pdbrenum
    pdbrenum -mmCIF -i <UniProt>.txt -o manuscript/data/structures_multichain/

(Or run PDBrenum from a checked-out clone of
https://github.com/Faezov/PDBrenum.)

The benchmark we publish was built against PDBe v2024-Q4; structures
authoritatively curated since then may have residue-number conflicts.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd

from reproduction.paths import BENCHMARK, STRUCTURES_MULTICHAIN

URL_TEMPLATE = "https://files.rcsb.org/download/{pdb}-assembly1.cif"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a local copy exists")
    args = ap.parse_args()

    manifest = pd.read_csv(BENCHMARK / "manifest.csv")
    STRUCTURES_MULTICHAIN.mkdir(parents=True, exist_ok=True)
    n_have = n_ok = n_fail = 0
    for pdb in sorted(manifest.pdb.unique()):
        out = STRUCTURES_MULTICHAIN / f"{pdb.upper()}.cif"
        if out.exists() and not args.force:
            n_have += 1
            continue
        url = URL_TEMPLATE.format(pdb=pdb.lower())
        try:
            with urlopen(Request(url, headers={"User-Agent": "allogator/1.0"})) as r:
                data = r.read()
            out.write_bytes(data)
            n_ok += 1
        except HTTPError as e:
            print(f"  HTTP {e.code} for {pdb}", file=sys.stderr)
            n_fail += 1
        time.sleep(0.2)
        if n_ok % 20 == 0:
            print(f"  downloaded {n_ok}")
    print(f"already on disk: {n_have}; downloaded: {n_ok}; failed: {n_fail}")
    print("\nNext step: renumber to UniProt with PDBrenum:")
    print("    pip install pdbrenum")
    print("    pdbrenum -mmCIF -i manuscript/data/benchmark/manifest.csv -o manuscript/data/structures_multichain/")


if __name__ == "__main__":
    main()
