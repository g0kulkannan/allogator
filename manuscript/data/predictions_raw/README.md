# External-method outputs

This directory contains the original score outputs needed to inspect and
reprocess the external comparator methods.

- `ohm/`: one ACI value per residue in the paired PDB's residue order. The
  paired structures are in `manuscript/data/structures/`.
  `manuscript/scripts/scoring/run_ohm.py` maps the values to residue numbers
  and chains, filters to chains matching the benchmark sequence, applies the
  candidate-residue rule, and calculates the released scores and AUROCs.
- `evcouplings/`: one compressed `i`, `j`, `cn` pair-score table per benchmark
  protein. `manuscript/scripts/scoring/run_evcouplings.py` expands the
  symmetric score table, handles absent pairs as described in the method,
  sums coupling scores to the active-site positions, applies the
  candidate-residue rule, and calculates the released scores and AUROCs.
