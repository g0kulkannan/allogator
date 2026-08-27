#!/usr/bin/env bash
# Entry point for generating the computational figures and tables.
#
# Usage:
#   ./manuscript/run_all.sh
#   ./manuscript/run_all.sh figures
#   ./manuscript/run_all.sh check

set -euo pipefail

MANUSCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$MANUSCRIPT_ROOT/.." && pwd)"
export ALLOGATOR_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$MANUSCRIPT_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

target="${1:-figures}"

preflight_figure_inputs () {
  local missing=0
  local required=(
    "data/benchmark/manifest.csv"
    "data/benchmark/sequences.pickle"
    "data/benchmark/active_residues.pickle"
    "data/benchmark/allo_residues.pickle"
    "data/benchmark/neff_colabfold_uniref.csv"
    "data/cached_results/Ohm/per_protein_auroc.json"
    "data/cached_results/Ohm/residue_scores.json"
    "data/cached_results/Ohm/chain_residue_scores.csv.gz"
    "data/cached_results/EVcouplings/per_protein_auroc.json"
    "data/cached_results/EVcouplings/residue_scores.json"
    "data/cached_results/ESM1b/per_protein_auroc.json"
    "data/cached_results/ESM1b/residue_scores.json"
    "data/cached_results/ESMpp/per_protein_auroc.json"
    "data/cached_results/ESMpp/residue_scores.json"
    "data/cached_results/ESM2_650M/per_protein_auroc.json"
    "data/cached_results/ESM2_650M/residue_scores.json"
    "data/cached_results/ProtT5/per_protein_auroc.json"
    "data/cached_results/ProtT5/residue_scores.json"
    "data/cached_results/ESM1b_contacts/per_protein_auroc.json"
    "data/cached_results/ESM1b_contacts/residue_scores.json"
    "data/cached_results/Distance/per_protein_auroc.json"
    "data/cached_results/Distance/residue_scores.json"
    "data/cached_results/Random/per_protein_auroc.json"
    "data/cached_results/Random/residue_scores.json"
    "data/cached_results/ESM1b_mean_max/per_protein_auroc.json"
    "data/cached_results/ESM1b_mean_max/residue_scores.json"
    "data/cached_results/ESM1b_max_sum/per_protein_auroc.json"
    "data/cached_results/ESM1b_max_sum/residue_scores.json"
    "data/cached_results/ESM1b_max_max/per_protein_auroc.json"
    "data/cached_results/ESM1b_max_max/residue_scores.json"
    "data/cached_results/ESM1b_min_sum/per_protein_auroc.json"
    "data/cached_results/ESM1b_min_sum/residue_scores.json"
    "data/cached_results/ESM1b_min_max/per_protein_auroc.json"
    "data/cached_results/ESM1b_min_max/residue_scores.json"
    "data/cached_results/ESM1b_median_sum/per_protein_auroc.json"
    "data/cached_results/ESM1b_median_sum/residue_scores.json"
    "data/cached_results/ESM1b_median_max/per_protein_auroc.json"
    "data/cached_results/ESM1b_median_max/residue_scores.json"
    "data/cached_results/ESM1b_max_sym_sum/per_protein_auroc.json"
    "data/cached_results/ESM1b_max_sym_sum/residue_scores.json"
    "data/cached_results/ESM2_8M/per_protein_auroc.json"
    "data/cached_results/ESM2_8M/residue_scores.json"
    "data/cached_results/ESM2_35M/per_protein_auroc.json"
    "data/cached_results/ESM2_35M/residue_scores.json"
    "data/cached_results/ESM2_150M/per_protein_auroc.json"
    "data/cached_results/ESM2_150M/residue_scores.json"
    "data/cached_results/ESM2_3B/per_protein_auroc.json"
    "data/cached_results/ESM2_3B/residue_scores.json"
    "data/cached_results/ESM2_15B/per_protein_auroc.json"
    "data/cached_results/ESM2_15B/residue_scores.json"
    "data/cached_results/PerLayer/per_layer_auroc.csv"
    "data/cached_results/PerLayer/per_head_auroc.csv"
    "figures/figS5_distance_vs_attention/attention_rank_vs_distance.csv"
    "figures/figS6_interface_bias/interface_per_protein.csv"
    "case_studies/scripts/build_case_study_outputs.py"
    "case_studies/scripts/plot_case_study_figures.py"
    "case_studies/data/b2ar/configuration.json"
    "case_studies/data/b2ar/esm1b_mean_attention.npz"
    "case_studies/data/b2ar/evcouplings_pairs.csv.gz"
    "case_studies/data/b2ar/mutational_pharmacology.csv"
    "case_studies/data/b2ar/ohm_4ldo_bfactor.pdb"
    "case_studies/data/b2ar/residue_classifications.csv"
    "case_studies/data/b2ar/sequence_and_gpcrdb.csv"
    "case_studies/data/androgen_receptor/configuration.json"
    "case_studies/data/androgen_receptor/esm1b_mean_attention.npz"
    "case_studies/data/androgen_receptor/androgen_db_export.csv"
    "case_studies/data/androgen_receptor/pdb_coverage_alignment.clustal"
    "case_studies/data/androgen_receptor/AF3_dimer.cif"
  )
  local rel
  for rel in "${required[@]}"; do
    if [[ ! -s "$MANUSCRIPT_ROOT/$rel" ]]; then
      echo "missing required input: $rel" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "figure-input check failed; see README.md" >&2
    return 1
  fi
}

run_figures () {
  echo "=== rendering figures ==="
  python "$MANUSCRIPT_ROOT/scripts/figures/fig2b_overlap_3way.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/fig2c_auroc_scatter.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS1_filter_flowchart.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS2a_overlap_compact.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS2b_auroc_with_esmpp.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS2c_esm1b_vs_prott5.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS2d_attn_gamma_grid.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS3_esm2_scale.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS4_per_layer.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS5_distance_vs_attention.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS6_interface_bias.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/figS7_neff_per_L.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/tableS1_per_protein_auroc.py"
  python "$MANUSCRIPT_ROOT/scripts/figures/tableS3_residues.py"
  echo "=== building B2AR and androgen-receptor panels ==="
  python "$MANUSCRIPT_ROOT/case_studies/scripts/build_case_study_outputs.py"
  python "$MANUSCRIPT_ROOT/case_studies/scripts/plot_case_study_figures.py"
}

run_check () {
  echo "=== checking released scores and source-output coverage ==="
  python "$MANUSCRIPT_ROOT/scripts/scoring/validate_cached_scores.py"
}

case "$target" in
  figures) preflight_figure_inputs; run_figures ;;
  check) preflight_figure_inputs; run_check ;;
  -h|--help)
    echo "usage: ./manuscript/run_all.sh [figures|check]"
    exit 0
    ;;
  *)
    echo "usage: ./manuscript/run_all.sh [figures|check]" >&2
    exit 2
    ;;
esac

echo "=== done: $target ==="
