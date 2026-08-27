# Benchmark data sources

The benchmark contains 109 proteins with experimentally annotated active and
allosteric sites. The files in this directory provide the protein set,
sequences, residue annotations, and representative-structure identifiers used
by the scoring and figure scripts.

## Files

- `manifest.csv`: UniProt accession, sequence length, representative PDB
  assembly, and structure-selection metadata for each protein.
- `sequences.pickle`: UniProt amino-acid sequences.
- `active_residues.pickle`: one-based catalytic-site residue positions.
- `allo_residues.pickle`: one-based allosteric-site residue positions.
- `neff_colabfold_uniref.csv`: effective-sequence-count values used in Fig. S7.

## Sources

- **Allosteric-site annotations:** AlloBench, Maity and Qiao (2025), DOI
  [10.1021/acsomega.5c01263](https://doi.org/10.1021/acsomega.5c01263).
  Source code and data are available from
  [djmaity/allobench](https://github.com/djmaity/allobench) under the MIT
  License.
- **Catalytic-site annotations:** M-CSA catalytic residues combined with
  UniProtKB/Swiss-Prot `Active site` features. Substrate-binding annotations
  are not treated as catalytic sites.
- **Sequences:** UniProtKB/Swiss-Prot sequences for the accessions listed in
  `manifest.csv`.
- **Structures and coordinate mappings:** PDB biological assemblies and the
  PDBe-KB/SIFTS 2024-Q4 mapping, renumbered to UniProt coordinates with
  PDBrenum.
- **Ligand annotations:** PDBe-KB and ChEMBL.

The complete residue lists for each protein are also reported in Table S3.

## External-method score inputs

The benchmark includes the analysis-complete inputs used for the two external
scoring methods:

- `manuscript/data/predictions_raw/evcouplings/` contains 109 compressed
  pair-score tables, one for every benchmark protein. Each table retains the
  `i`, `j`, and `cn` columns consumed by
  `manuscript/scripts/scoring/run_evcouplings.py`.
- `manuscript/data/predictions_raw/ohm/` contains 100 original Ohm ACI output
  files. Each file is paired by filename with the submitted PDB in
  `manuscript/data/structures/` and is consumed by
  `manuscript/scripts/scoring/run_ohm.py`.

Ohm output was not available for `P52731`, `P53704`, `Q01217`, `Q13976`,
`Q8CG03`, `Q8F3Q1`, `Q8I719`, `Q922S4`, or `Q9Y223`; these proteins are not
part of the Ohm comparison. The other 100 files reproduce the released Ohm
residue scores after retaining chains that map to the benchmark UniProt
sequence.
