# Released score files

Each method directory contains the score layer read by the figure scripts:

- `residue_scores.json`: per-protein, per-residue method scores after applying
  the common candidate-residue rule (active-site positions and their immediate
  sequence neighbors are excluded).
- `per_protein_auroc.json`: AUROC calculated from those scores and the
  benchmark allosteric-site labels.

Most residue-score files use a mapping from residue-number strings to scores.
Some use a list of `[residue_number, score]` pairs. The common loader in
`manuscript/scripts/reproduction/io.py` accepts both forms.

Ohm additionally includes `chain_residue_scores.csv.gz`. It retains separate
chain observations and indicates positions imputed because they were not
resolved in the submitted PDB. The Ohm AUROC is calculated from this
chain-resolved table; `residue_scores.json` takes the maximum score at each
UniProt position for residue-level comparisons.

`PerLayer/` contains the 109 × 33 layer table and 109 × 660 layer/head table
used by Figure S4.

Run the following from the repository root to verify the released
score-to-AUROC transformation without loading any language model:

```bash
PYTHONPATH=src:manuscript/scripts python manuscript/scripts/scoring/validate_cached_scores.py
```
