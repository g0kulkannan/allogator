# AlloGator

AlloGator ranks candidate allosteric residues from a protein sequence and a
defined active site using ESM-1b attention. It requires no structure or
multiple-sequence alignment.

The required inputs are:

- one amino-acid sequence, up to 1,022 residues; and
- the 1-indexed positions of one or more active-site residues.

## Score a new protein

### Install

For new-protein prediction, create a Python 3.10+ environment and install the
command-line tool:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The supplied conda environment includes the additional dependencies needed to
reproduce all publication analyses:

```bash
conda env create -f manuscript/environment.yml
conda activate allogator
pip install -e . --no-deps
```

The first prediction downloads the approximately 7.3 GB ESM-1b checkpoint.
Subsequent runs use the local model cache. CUDA users may need to install a
PyTorch build appropriate for their CUDA setup before installing AlloGator.

### One protein

Score a single-record FASTA file:

```bash
allogator score \
  --fasta examples/example_protein.fasta \
  --active-residues 5,12 \
  --output predictions/example_alpha.csv
```

The sequence can also be supplied directly:

```bash
allogator score \
  --sequence MKTAYIAKQRQISFVKSHFSRQLE \
  --active-residues 5,12 \
  --protein-id example_alpha \
  --output predictions/example_alpha.csv
```

From a checkout, the same interface is available without installing the
console command:

```bash
python predict.py score \
  --fasta examples/example_protein.fasta \
  --active-residues 5,12
```

### A CSV batch

Use three required columns. Active-site positions are separated by semicolons
inside the CSV:

```csv
protein_id,sequence,active_residues
example_alpha,MKTAYIAKQRQISFVKSHFSRQLE,5;12
example_beta,GAVLIPFYWSTCMNQDEKRH,4;11
```

Run the included template with:

```bash
allogator batch \
  --input-csv examples/batch_input.csv \
  --output predictions/batch_scores.csv
```

The entire CSV is validated before inference. ESM-1b is then loaded once and
the proteins are processed sequentially in input order.

### CPU, CUDA, and Apple Silicon

The default `--device auto` selects MPS, then CUDA, then CPU according to
availability. A backend can be selected explicitly:

```bash
allogator score --fasta examples/example_protein.fasta --active-residues 5,12 --device cpu
allogator score --fasta examples/example_protein.fasta --active-residues 5,12 --device cuda
allogator score --fasta examples/example_protein.fasta --active-residues 5,12 --device mps
```

CPU inference is supported but substantially slower;
`--cpu-threads N` controls the number of PyTorch CPU threads.

### Prediction output

The output is one CSV containing the proteins in input order and residues in
sequence order:

| Column | Meaning |
|---|---|
| `protein_id` | User-supplied, FASTA, or batch identifier |
| `residue_number` | 1-indexed sequence position |
| `amino_acid` | Residue at that position |
| `attention_score` | Mean layer/head attention from the active-site rows, summed over active residues |
| `rank` | Descending rank among candidate residues; 1 is highest |
| `rank_percentile` | Within-protein candidate percentile; higher values indicate higher scores |
| `is_active_site` | Whether the row is an input active-site residue |
| `is_sequence_adjacent` | Whether the row is sequence-adjacent to an active-site residue |
| `is_candidate` | Whether the residue is included in the ranking |

Active-site residues and positions immediately adjacent in sequence (±1) are
scored for transparency but excluded from the candidate ranking. Their rank
fields are blank. Add `--candidate-only` to write only ranked candidates.
Tied scores receive their average rank and percentile. A sole candidate has a
percentile of 1. Raw attention scores are not rounded.
Ranks and percentiles are defined within each protein. Raw scores are not
calibrated for comparisons between proteins or between differently sized
active sites.

## Prediction method

For each protein, AlloGator:

1. obtains the 33-layer × 20-head ESM-1b attention tensor;
2. averages attention over all layers and heads;
3. selects the directional query rows corresponding to the supplied active
   site and sums those rows; and
4. ranks the remaining candidate positions after excluding the active site
   and sequence-adjacent residues.

## Reproduce the publication analyses

All files specific to the analyses in *Single-Sequence, Structure-Free
Allosteric Residue Prediction with Protein Language Models* (Kannan et al.)
are grouped under `manuscript/`:

- `manuscript/data/` contains benchmark inputs, raw comparator outputs,
  released score caches, and structures;
- `manuscript/scripts/` contains build, scoring, and figure scripts;
- `manuscript/figures/` contains the generated benchmark figures together
  with the processed tables used directly by Figures S5 and S6; and
- `manuscript/case_studies/` contains the B2AR and androgen-receptor inputs,
  outputs, scripts, and figures.

Install the reproduction dependencies with the supplied conda environment or:

```bash
pip install -e ".[replication]"
```

Generate the computational figures and supplementary tables:

```bash
./manuscript/run_all.sh
```

This writes PNG figures and supplementary tables under
`manuscript/figures/` and case-study figures under
`manuscript/case_studies/figures/`. The released scores and source-output
coverage can be checked without loading a language model:

```bash
./manuscript/run_all.sh check
```

The B2AR and androgen-receptor analyses can also be run directly:

```bash
python manuscript/case_studies/scripts/build_case_study_outputs.py
python manuscript/case_studies/scripts/plot_case_study_figures.py
```

The figure command uses the final per-residue scores and per-protein metrics in
`manuscript/data/cached_results/`. Scripts in
`manuscript/scripts/scoring/` recompute individual methods. Sequence-model
calculations may require a GPU.

The external-method score inputs used for the benchmark are also included:

- `manuscript/data/predictions_raw/evcouplings/` contains one compressed `i`,
  `j`, `cn` pair-score table for each of the 109 benchmark proteins.
- `manuscript/data/predictions_raw/ohm/` contains the original ACI output for
  each of the 100 proteins evaluated with Ohm.
- `manuscript/data/structures/` contains the selected PDB inputs used to map
  the Ohm outputs to UniProt residue positions and chains.

These inputs reproduce the released EVcouplings and target-chain-filtered Ohm
scores and AUROCs:

```bash
PYTHONPATH=src:manuscript/scripts python manuscript/scripts/scoring/run_evcouplings.py
PYTHONPATH=src:manuscript/scripts python manuscript/scripts/scoring/run_ohm.py
```

Generating Ohm or EVcouplings scores for different proteins still requires
the corresponding external method.

The 3D-distance and interface analyses require renumbered multichain PDB
biological assemblies. Required PDB entries are listed in
`manuscript/data/benchmark/manifest.csv`; assemblies can be downloaded with:

```bash
PYTHONPATH=src:manuscript/scripts python manuscript/scripts/build/download_assemblies.py
```

The PDB files in `manuscript/data/structures/` are the inputs paired with the
Ohm ACI outputs; they are not substitutes for the multichain assemblies used
by the distance and interface analyses.

Run an individual scoring script from the repository root, for example:

```bash
PYTHONPATH=src:manuscript/scripts python manuscript/scripts/scoring/run_esm1b.py
```

#### Figure and table scripts

| Manuscript element | Script |
|---|---|
| Fig. 2B | `manuscript/scripts/figures/fig2b_overlap_3way.py` |
| Fig. 2C | `manuscript/scripts/figures/fig2c_auroc_scatter.py` |
| Fig. S1 | `manuscript/scripts/figures/figS1_filter_flowchart.py` |
| Fig. S2A | `manuscript/scripts/figures/figS2a_overlap_compact.py` |
| Fig. S2B | `manuscript/scripts/figures/figS2b_auroc_with_esmpp.py` |
| Fig. S2C | `manuscript/scripts/figures/figS2c_esm1b_vs_prott5.py` |
| Fig. S2D | `manuscript/scripts/figures/figS2d_attn_gamma_grid.py` |
| Fig. S3 | `manuscript/scripts/figures/figS3_esm2_scale.py` |
| Fig. S4 | `manuscript/scripts/figures/figS4_per_layer.py` |
| Fig. S5 | `manuscript/scripts/figures/figS5_distance_vs_attention.py` |
| Fig. S6 | `manuscript/scripts/figures/figS6_interface_bias.py` |
| Fig. S7 | `manuscript/scripts/figures/figS7_neff_per_L.py` |
| Figs. 3 and S8 (B2AR) | `manuscript/case_studies/scripts/build_case_study_outputs.py`; `manuscript/case_studies/scripts/plot_case_study_figures.py` |
| Fig. S9 (androgen receptor) | `manuscript/case_studies/scripts/build_case_study_outputs.py`; `manuscript/case_studies/scripts/plot_case_study_figures.py` |
| Table S1 | `manuscript/scripts/figures/tableS1_per_protein_auroc.py` |
| Table S2 | `manuscript/case_studies/scripts/build_case_study_outputs.py` |
| Table S3 | `manuscript/scripts/figures/tableS3_residues.py` |

The structural views were assembled in PyMOL. The optional script
`manuscript/scripts/figures/pymol_sessions.py` generates sessions from
renumbered multichain structures.

## Data sources

- Allosteric-site annotations: AlloBench (Maity and Qiao, 2025), DOI
  [10.1021/acsomega.5c01263](https://doi.org/10.1021/acsomega.5c01263).
- Catalytic-site annotations: M-CSA and UniProtKB/Swiss-Prot active-site
  features.
- Protein sequences: UniProtKB/Swiss-Prot.
- Structures and residue mappings: PDB biological assemblies and PDBe-KB/SIFTS.
- Ohm allosteric-coupling scores: Wang et al., *Nature Communications*
  (2020).
- EVcouplings evolutionary-coupling scores: Hopf et al., *Bioinformatics*
  (2019).
- B2AR pharmacology: Heydenreich et al., *Science* (2023), DOI
  [10.1126/science.adh1859](https://doi.org/10.1126/science.adh1859), retrieved
  January 7, 2024. The included converted and rounded CSV is distributed under
  CC BY 4.0.
- AndrogenDB: Gottlieb et al., *Human Mutation* (2012), DOI
  [10.1002/humu.22046](https://doi.org/10.1002/humu.22046), retrieved May 9,
  2024.

Additional benchmark-source details are in
[`manuscript/data/benchmark/SOURCES.md`](manuscript/data/benchmark/SOURCES.md).
No DPP4 or ACE2 experimental data are included in this repository.

## Citation

Please cite:

> Kannan GR, Zheng C, Hie BL, Kim PS. Single-Sequence, Structure-Free
> Allosteric Residue Prediction with Protein Language Models. *Cell Systems*.
> 2026. DOI forthcoming.

## License

The code is available under the [MIT License](LICENSE). Third-party datasets
remain subject to their original terms and should be cited as described above.
