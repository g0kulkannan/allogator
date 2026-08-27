"""Slim and compress raw EVcouplings per-protein coupling tables.

Reads each benchmark protein's full EVcouplings output CSV (with all
11 or so plmc columns: i, A_i, j, A_j, fn, cn, segment_i, segment_j,
mad_score, probability, score), keeps only ``i``, ``j``, ``cn`` —
which is what ``run_evcouplings.py`` consumes — and writes the result
gzipped to ``manuscript/data/predictions_raw/evcouplings/<UniProt>.csv.gz``.

For the 109-protein benchmark this reduces the input size from about 1.2 GB
to about 90 MB without changing the values used for EVcouplings scoring.

Source location is read from the required ``EVCOUPLINGS_RAW`` environment
variable.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from reproduction.paths import BENCHMARK, PREDICTIONS_RAW

DST = PREDICTIONS_RAW / "evcouplings"


def main() -> None:
    raw_source = os.environ.get("EVCOUPLINGS_RAW")
    if not raw_source:
        raise SystemExit(
            "Set EVCOUPLINGS_RAW to the directory of full plmc-output CSVs."
        )
    source = Path(raw_source).expanduser()
    if not source.exists():
        raise SystemExit(
            f"raw EVcouplings dir not found: {source}\n"
            f"Set the EVCOUPLINGS_RAW environment variable to point at "
            f"the directory of full plmc-output CSVs.")
    DST.mkdir(parents=True, exist_ok=True)
    benchmark = pickle.load(open(BENCHMARK / "sequences.pickle", "rb"))
    total_in = total_out = 0
    missing = []
    for u in tqdm(sorted(benchmark), desc="slim+gz"):
        src = source / f"{u}.csv"
        if not src.exists():
            missing.append(u); continue
        df = pd.read_csv(src, usecols=["i", "j", "cn"])
        out = DST / f"{u}.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        total_in  += src.stat().st_size
        total_out += out.stat().st_size
    print(f"in  total: {total_in/1024/1024:>7.1f} MB")
    print(f"out total: {total_out/1024/1024:>7.1f} MB  "
          f"({100*total_out/max(1,total_in):.1f}% of full)")
    if missing:
        print(f"missing CSVs for {len(missing)} proteins: {missing[:5]}...")


if __name__ == "__main__":
    main()
