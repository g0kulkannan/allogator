#!/usr/bin/env python3
"""Build B2AR and androgen-receptor case-study tables and statistics.

All inputs and outputs are resolved relative to the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import AlignIO, PDB
from scipy.stats import hypergeom, mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def stable_statistic(value: float) -> float:
    """Remove insignificant SciPy/NumPy last-bit differences from JSON."""
    return float(f"{float(value):.12g}")


def excluded_positions(active: list[int], radius: int) -> set[int]:
    return {
        residue + offset
        for residue in set(active)
        for offset in range(-radius, radius + 1)
    }


def scalar(npz: np.lib.npyio.NpzFile, key: str) -> Any:
    return npz[key].item()


def correlation_summary(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    columns = [score_column, "normalised_logEC50", "normalised_amplitude"]
    clean = frame.dropna(subset=columns)
    potency = spearmanr(clean[score_column], clean["normalised_logEC50"])
    efficacy = spearmanr(clean[score_column], clean["normalised_amplitude"])
    return {
        "n": int(len(clean)),
        "normalised_logEC50": {
            "spearman_rho": stable_statistic(potency.statistic),
            "two_sided_p": stable_statistic(potency.pvalue),
        },
        "normalised_amplitude": {
            "spearman_rho": stable_statistic(efficacy.statistic),
            "two_sided_p": stable_statistic(efficacy.pvalue),
        },
    }


def top50_quadrant_enrichment(
    frame: pd.DataFrame,
    score_column: str,
    background_total: int,
    background_non_wt_like: int,
    potency_cutoff: float,
    amplitude_cutoff: float,
) -> dict[str, Any]:
    """Summarize the Fig. S8 top-50 quadrant analysis."""
    top = frame.dropna(subset=[score_column]).nlargest(50, score_column)
    if len(top) != 50:
        raise ValueError(f"{score_column} has fewer than 50 scoreable variants")
    wt_like = top["normalised_amplitude"].ge(amplitude_cutoff) & top[
        "normalised_logEC50"
    ].lt(potency_cutoff)
    non_wt_like = int((~wt_like).sum())
    fold_enrichment = (non_wt_like / len(top)) / (
        background_non_wt_like / background_total
    )
    p_value = hypergeom.sf(
        non_wt_like - 1,
        background_total,
        background_non_wt_like,
        len(top),
    )
    return {
        "top_n": int(len(top)),
        "wt_like_n": int(wt_like.sum()),
        "non_wt_like_n": non_wt_like,
        "fold_enrichment": stable_statistic(fold_enrichment),
        "one_sided_hypergeometric_p": stable_statistic(p_value),
    }


def read_b2ar_mapping(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        raise ValueError("B2AR mapping must contain region, GPCRdb, and sequence rows")

    # The first and last columns are row labels/trailing delimiters. The
    # remaining 413 columns map one-to-one to P07550.
    regions = raw.iloc[0, 1:-1].ffill()
    gpcrdb = raw.iloc[1, 1:-1]
    amino_acids = raw.iloc[2, 1:-1].astype(str).str[0]
    return pd.DataFrame(
        {
            "residue_number": np.arange(1, len(amino_acids) + 1),
            "amino_acid": amino_acids.to_numpy(),
            "structural_region": regions.to_numpy(),
            "GPCRdb_number": gpcrdb.to_numpy(),
        }
    )


def read_ohm_bfactors(path: Path, chain_id: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21] != chain_id:
                continue
            residue_number = int(line[22:26].strip())
            if residue_number >= 5000:
                continue
            beta_factor = float(line[60:66].strip())
            existing_score = scores.setdefault(residue_number, beta_factor)
            if not np.isclose(existing_score, beta_factor):
                raise ValueError(
                    f"Ohm PDB has inconsistent B-factors for residue {residue_number}"
                )
    return scores


def build_evc_scores(
    path: Path,
    active: list[int],
    residue_start: int,
    residue_end: int,
    exclusion_radius: int,
) -> dict[int, float]:
    couplings = pd.read_csv(path)
    table = couplings.pivot(index="j", columns="i", values="score")
    symmetric = (table.fillna(0) + table.T.fillna(0)) / 2
    excluded = excluded_positions(active, exclusion_radius)
    scores: dict[int, float] = {}

    for residue in range(residue_start, residue_end + 1):
        if residue in excluded or residue not in symmetric.columns:
            continue
        values = []
        for active_residue in active:
            if (
                abs(residue - active_residue) > exclusion_radius
                and active_residue in symmetric.index
                and residue in symmetric.columns
            ):
                values.append(float(symmetric.loc[active_residue, residue]))
        if values:
            scores[residue] = float(np.sum(values))
    return scores


def build_b2ar() -> dict[str, Any]:
    case_dir = DATA / "b2ar"
    config = load_json(case_dir / "configuration.json")
    active = [int(x) for x in config["active_site_residues_1_based"]]
    radius = int(config["candidate_exclusion_radius_in_sequence"])
    excluded = excluded_positions(active, radius)

    matrix_archive = np.load(case_dir / "esm1b_mean_attention.npz", allow_pickle=False)
    matrix = matrix_archive["mean_attention"]
    sequence = str(scalar(matrix_archive, "sequence"))
    accession = str(scalar(matrix_archive, "accession"))
    if accession != config["accession"]:
        raise ValueError(f"B2AR accession mismatch: {accession}")
    if matrix.shape != (len(sequence), len(sequence)):
        raise ValueError("B2AR attention matrix and sequence lengths differ")

    mapping = read_b2ar_mapping(case_dir / "sequence_and_gpcrdb.csv")
    mapped_sequence = "".join(mapping["amino_acid"])
    if mapped_sequence != sequence:
        raise ValueError("B2AR sequence mapping does not match the attention archive")

    attention_scores = matrix[np.asarray(active) - 1, :].sum(axis=0)
    attention_map = {
        residue: float(attention_scores[residue - 1])
        for residue in range(1, len(sequence) + 1)
        if residue not in excluded
    }

    ohm_map_all = read_ohm_bfactors(
        case_dir / "ohm_4ldo_bfactor.pdb", str(config["ohm_pdb_chain"])
    )
    ohm_map = {
        residue: score
        for residue, score in ohm_map_all.items()
        if residue not in excluded
    }

    evc_start, evc_end = [
        int(x) for x in config["evcouplings_residue_range_inclusive"]
    ]
    evc_map = build_evc_scores(
        case_dir / "evcouplings_pairs.csv.gz",
        active,
        evc_start,
        evc_end,
        radius,
    )

    classifications = pd.read_csv(case_dir / "residue_classifications.csv")
    classifications["amino_acid"] = pd.to_numeric(
        classifications["amino_acid"], errors="coerce"
    ).astype("Int64")
    residue_table = mapping.merge(
        classifications.rename(columns={"amino_acid": "residue_number"}),
        how="left",
        on="residue_number",
    )
    residue_table["is_active_site"] = residue_table["residue_number"].isin(active)
    residue_table["excluded_active_or_sequence_adjacent"] = residue_table[
        "residue_number"
    ].isin(excluded)
    residue_table["attention_score"] = residue_table["residue_number"].map(
        attention_map
    )
    residue_table["evcouplings_score"] = residue_table["residue_number"].map(evc_map)
    residue_table["ohm_ACI"] = residue_table["residue_number"].map(ohm_map)
    residue_table["attention_rank_descending"] = residue_table[
        "attention_score"
    ].rank(ascending=False, method="min")
    residue_table.to_csv(
        OUTPUTS / "b2ar_residue_scores.csv", index=False, float_format="%.12g"
    )

    pharmacology = pd.read_csv(case_dir / "mutational_pharmacology.csv")
    pharmacology["amino_acid"] = pd.to_numeric(
        pharmacology["amino_acid"], errors="coerce"
    ).astype("Int64")
    pharmacology["attention"] = pharmacology["amino_acid"].map(attention_map)
    pharmacology["evc_score"] = pharmacology["amino_acid"].map(evc_map)
    pharmacology["ohm_ACI"] = pharmacology["amino_acid"].map(ohm_map)
    pharmacology["attention_rank"] = pharmacology["attention"].rank(ascending=True)
    pharmacology["evc_score_rank"] = pharmacology["evc_score"].rank(ascending=True)
    pharmacology["ohm_ACI_rank"] = pharmacology["ohm_ACI"].rank(ascending=True)
    pharmacology.to_csv(
        OUTPUTS / "b2ar_plot_data.csv", index=False, float_format="%.12g"
    )

    # Select the ESM-1b top 20 before considering coverage by the comparison
    # methods. This retains expression-low variants such as W286A, P211A, and
    # C106A and leaves unavailable comparison-method ranks as N/A.
    table = pharmacology.dropna(subset=["attention_rank"]).copy()
    table["flipped_attention_rank"] = (
        pharmacology["attention_rank"].max() + 1 - table["attention_rank"]
    )
    table["flipped_evc_score_rank"] = (
        pharmacology["evc_score_rank"].max() + 1 - table["evc_score_rank"]
    )
    table["flipped_ohm_ACI_rank"] = (
        pharmacology["ohm_ACI_rank"].max() + 1 - table["ohm_ACI_rank"]
    )
    table = table.sort_values("flipped_attention_rank").head(20)
    for rank_column in [
        "flipped_attention_rank",
        "flipped_evc_score_rank",
        "flipped_ohm_ACI_rank",
    ]:
        table[rank_column] = table[rank_column].map(
            lambda value: int(value) if pd.notna(value) else "N/A"
        )
    table = table[
        [
            "mutation",
            "GPCRdb",
            "motif",
            "potency",
            "efficacy",
            "flipped_attention_rank",
            "flipped_evc_score_rank",
            "flipped_ohm_ACI_rank",
        ]
    ].rename(
        columns={
            "mutation": "Mutation",
            "GPCRdb": "GPCRdb numbering",
            "motif": "Motif",
            "potency": "Potency",
            "efficacy": "Efficacy",
            "flipped_attention_rank": "ESM1b Attention Rank",
            "flipped_evc_score_rank": "EVcouplings Rank",
            "flipped_ohm_ACI_rank": "Ohm Rank",
        }
    )
    table = table.fillna("N/A")
    table.to_csv(OUTPUTS / "tableS2.csv", index=False)

    # Figure 3B excludes variants explicitly annotated as having low
    # expression. Figure S8 correlations use every method-scoreable variant.
    signaling_test = pharmacology.loc[
        pharmacology["expression_low"].eq(False)
    ].copy()
    group1_mask = (
        signaling_test["potency"].eq("WT-like")
        & signaling_test["efficacy"].eq("WT-like")
    )
    group2_mask = (
        (
            signaling_test["potency"].ne("WT-like")
            | signaling_test["efficacy"].ne("WT-like")
        )
        & signaling_test["potency"].notna()
        & signaling_test["efficacy"].notna()
    )
    group1 = signaling_test.loc[group1_mask, "attention"].dropna()
    group2 = signaling_test.loc[group2_mask, "attention"].dropna()
    mw = mannwhitneyu(group1, group2, alternative="two-sided")

    # The dotted lines in Figure S8 define WT-like signaling as amplitude
    # >= 0.75 and
    # normalized logEC50 < 0.87. The background is all 412 alanine variants;
    # the separate WT control row is not part of that population.
    potency_cutoff = 0.87
    amplitude_cutoff = 0.75
    background = pharmacology.loc[pharmacology["amino_acid"].notna()].copy()
    background_wt_like = background["normalised_amplitude"].ge(
        amplitude_cutoff
    ) & background["normalised_logEC50"].lt(potency_cutoff)
    background_total = int(len(background))
    background_non_wt_like = int((~background_wt_like).sum())
    top50 = {
        method: top50_quadrant_enrichment(
            pharmacology,
            score_column,
            background_total,
            background_non_wt_like,
            potency_cutoff,
            amplitude_cutoff,
        )
        for method, score_column in {
            "esm1b_attention": "attention",
            "ohm": "ohm_ACI",
            "evcouplings": "evc_score",
        }.items()
    }
    # The separate Figure 3 top-20 enrichment defines a positive residue as
    # pharmacologically important or part of a named allosteric motif.
    top20_population = pharmacology.dropna(subset=["attention"]).copy()
    top20_population["annotated_allosteric"] = (
        top20_population["pharma_important"].eq(1)
        | top20_population["motif"].notna()
    )
    top20 = top20_population.nlargest(20, "attention")
    top20_background_total = int(len(top20_population))
    top20_background_positive = int(top20_population["annotated_allosteric"].sum())
    top20_positive = int(top20["annotated_allosteric"].sum())
    top20_fold = (top20_positive / len(top20)) / (
        top20_background_positive / top20_background_total
    )
    top20_p = hypergeom.sf(
        top20_positive - 1,
        top20_background_total,
        top20_background_positive,
        len(top20),
    )

    stats = {
        "accession": accession,
        "active_site_residues_1_based": active,
        "attention": correlation_summary(pharmacology, "attention"),
        "evcouplings": correlation_summary(pharmacology, "evc_score"),
        "ohm": correlation_summary(pharmacology, "ohm_ACI"),
        "signaling_group_test": {
            "test": "two-sided Mann-Whitney U",
            "wt_like_n": int(len(group1)),
            "altered_n": int(len(group2)),
            "statistic": stable_statistic(mw.statistic),
            "p": stable_statistic(mw.pvalue),
        },
        "top50_quadrant_enrichment": {
            "definition": {
                "wt_like": "normalised_amplitude >= 0.75 and normalised_logEC50 < 0.87",
                "non_wt_like": "all other quadrants",
                "background_total": background_total,
                "background_non_wt_like": background_non_wt_like,
                "test": "one-sided hypergeometric enrichment",
            },
            "methods": top50,
        },
        "top20_annotated_allosteric_enrichment": {
            "definition": "pharma_important == 1 OR named allosteric motif",
            "background_total": top20_background_total,
            "background_positive": top20_background_positive,
            "top_n": int(len(top20)),
            "top_positive": top20_positive,
            "fold_enrichment": stable_statistic(top20_fold),
            "test": "one-sided hypergeometric enrichment",
            "one_sided_hypergeometric_p": stable_statistic(top20_p),
        },
        "top_attention_residues": [
            int(x)
            for x in residue_table.sort_values("attention_score", ascending=False)[
                "residue_number"
            ].dropna().head(20)
        ],
    }

    write_json(OUTPUTS / "b2ar_statistics.json", stats)
    return stats


def clean_androgen_db(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["Position"]).copy()
    position_text = frame["Position"].astype(str)
    frame = frame.loc[~position_text.str.contains("-", regex=False)].copy()
    frame["Position"] = pd.to_numeric(
        frame["Position"].astype(str).str.replace("\n", "", regex=False),
        errors="raise",
    ).astype(int)

    status_by_position: dict[int, str] = {}
    for position, group in frame.groupby("Position", sort=False):
        phenotypes = group["Phenotype"]
        status_by_position[int(position)] = (
            "clinical_variant"
            if any(phenotype != "Normal" for phenotype in phenotypes)
            else "normal"
        )
    return pd.DataFrame(
        sorted(status_by_position.items()), columns=["residue_number", "clinical_status"]
    )


def build_pdb_coverage(path: Path, target_accession: str) -> pd.DataFrame:
    alignment = AlignIO.read(path, "clustal")
    array = np.asarray([list(str(record.seq)) for record in alignment])
    target_rows = [
        index
        for index, record in enumerate(alignment)
        if record.id == target_accession
    ]
    if len(target_rows) != 1:
        raise ValueError(
            f"Expected one {target_accession} row in the PDB-coverage alignment"
        )
    target_index = target_rows[0]
    target_columns = array[target_index] != "-"
    structure_rows = np.delete(array, target_index, axis=0)[:, target_columns]
    coverage = np.mean(structure_rows != "-", axis=0)
    return pd.DataFrame(
        {
            "residue_number": np.arange(1, len(coverage) + 1),
            "coverage_fraction": coverage,
            "present_in_any_structure": np.any(structure_rows != "-", axis=0),
        }
    )


def build_af3_plddt(path: Path, sequence: str) -> pd.DataFrame:
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("AR_AF3_dimer", path)
    scores: list[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                first_atom = next(iter(residue), None)
                if first_atom is not None:
                    scores.append(float(first_atom.get_bfactor()))

    # The AlphaFold3 model is a homodimer; use the first chain for the track.
    first_half = scores[: len(scores) // 2]
    if len(first_half) != len(sequence):
        raise ValueError(
            f"AR AF3 first-chain length {len(first_half)} != sequence length {len(sequence)}"
        )
    return pd.DataFrame(
        {
            "residue_number": np.arange(1, len(sequence) + 1),
            "amino_acid": list(sequence),
            "pLDDT": first_half,
        }
    )


def build_androgen_receptor() -> dict[str, Any]:
    case_dir = DATA / "androgen_receptor"
    config = load_json(case_dir / "configuration.json")
    active_weights = {
        int(residue): int(weight)
        for residue, weight in config["active_site_residue_weights"].items()
    }
    if not active_weights or any(weight < 1 for weight in active_weights.values()):
        raise ValueError("AR active-site weights must be positive integers")
    active = sorted(active_weights)
    radius = int(config["candidate_exclusion_radius_in_sequence"])
    excluded = excluded_positions(active, radius)

    matrix_archive = np.load(case_dir / "esm1b_mean_attention.npz", allow_pickle=False)
    matrix = matrix_archive["mean_attention"]
    sequence = str(scalar(matrix_archive, "sequence"))
    accession = str(scalar(matrix_archive, "accession"))
    if accession != config["accession"]:
        raise ValueError(f"AR accession mismatch: {accession}")
    if matrix.shape != (len(sequence), len(sequence)):
        raise ValueError("AR attention matrix and sequence lengths differ")

    weighted_rows = [
        matrix[residue - 1, :] * weight
        for residue, weight in active_weights.items()
    ]
    attention_scores = np.sum(weighted_rows, axis=0)
    candidate_positions = sorted(set(range(1, len(sequence) + 1)) - excluded)
    residues = pd.DataFrame({"residue_number": candidate_positions})
    residues["attention_score"] = residues["residue_number"].map(
        lambda residue: float(attention_scores[int(residue) - 1])
    )

    variants = clean_androgen_db(case_dir / "androgen_db_export.csv")
    variants.to_csv(OUTPUTS / "ar_clinical_variant_positions.csv", index=False)
    clinical_map = dict(zip(variants["residue_number"], variants["clinical_status"]))
    residues["amino_acid"] = residues["residue_number"].map(
        lambda residue: sequence[int(residue) - 1]
    )
    residues["clinical_status"] = residues["residue_number"].map(clinical_map).fillna(
        "unknown"
    )
    residues["plot_color_class"] = np.where(
        residues["clinical_status"].eq("clinical_variant"), "purple", "gray"
    )

    percentile = float(config["attention_percentile_cutoff"])
    cutoff = float(np.percentile(residues["attention_score"], percentile))
    residues["above_attention_percentile_cutoff"] = residues["attention_score"].ge(
        cutoff
    )
    dbd_start, dbd_end = [
        int(x) for x in config["dna_binding_domain_residue_range_inclusive"]
    ]
    residues["in_DNA_binding_domain"] = residues["residue_number"].between(
        dbd_start, dbd_end
    )

    residues = residues.sort_values("residue_number")
    residues.to_csv(
        OUTPUTS / "ar_residue_scores.csv", index=False, float_format="%.12g"
    )

    display_excluded = {
        int(position)
        for position in config["figureS9_display_excluded_positions_1_based"]
    }
    if not display_excluded.issubset(set(candidate_positions)):
        raise ValueError("Figure S9 display exclusions are outside the candidate set")
    figure_points = residues.loc[
        ~residues["residue_number"].isin(display_excluded)
    ].copy()
    expected_figure_counts = {"unknown": 553, "clinical_variant": 347}
    observed_figure_counts = figure_points["clinical_status"].value_counts().to_dict()
    if len(figure_points) != 900 or observed_figure_counts != expected_figure_counts:
        raise ValueError(
            f"Figure S9 expected 900 positions with {expected_figure_counts}, "
            f"found {len(figure_points)} with {observed_figure_counts}"
        )
    figure_points.to_csv(
        OUTPUTS / "ar_figureS9_plot_data.csv",
        index=False,
        float_format="%.12g",
    )

    dbd_hits = residues.loc[
        residues["in_DNA_binding_domain"]
        & residues["above_attention_percentile_cutoff"]
    ].copy()
    dbd_hits.to_csv(
        OUTPUTS / "ar_high_attention_dbd.csv", index=False, float_format="%.12g"
    )

    coverage = build_pdb_coverage(
        case_dir / "pdb_coverage_alignment.clustal", accession
    )
    coverage.to_csv(OUTPUTS / "ar_pdb_coverage.csv", index=False, float_format="%.12g")
    plddt = build_af3_plddt(case_dir / "AF3_dimer.cif", sequence)
    plddt.to_csv(OUTPUTS / "ar_af3_plddt.csv", index=False, float_format="%.12g")

    figure_status_counts = {
        key: int(value)
        for key, value in figure_points["clinical_status"].value_counts().items()
    }

    highlighted = [
        {
            "residue": f"{row.amino_acid}{int(row.residue_number)}",
            "attention_score": float(row.attention_score),
        }
        for row in dbd_hits.itertuples()
    ]

    stats = {
        "accession": accession,
        "active_site_residue_weights": {
            str(residue): int(weight)
            for residue, weight in active_weights.items()
        },
        "candidate_score_count": int(len(residues)),
        "attention_percentile": percentile,
        "attention_cutoff": cutoff,
        "high_attention_DBD_residues": highlighted,
        "candidate_clinical_status_counts": {
            key: int(value)
            for key, value in residues["clinical_status"].value_counts().items()
        },
        "figureS9_population": {
            "rows": int(len(figure_points)),
            "excluded_positions": sorted(display_excluded),
            "clinical_status_counts": figure_status_counts,
            "color_counts": {
                key: int(value)
                for key, value in figure_points["plot_color_class"].value_counts().items()
            },
        },
        "androgen_db": {
            "unique_positions": int(len(variants)),
            "clinical_variant_positions": int(
                variants["clinical_status"].eq("clinical_variant").sum()
            ),
            "normal_positions": int(variants["clinical_status"].eq("normal").sum()),
        },
        "pdb_coverage": {
            "residue_positions": int(len(coverage)),
        },
        "af3_plddt_rows": int(len(plddt)),
    }
    write_json(OUTPUTS / "ar_statistics.json", stats)
    return stats


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    b2ar = build_b2ar()
    ar = build_androgen_receptor()
    print(
        "Built B2AR/AR outputs successfully: "
        f"B2AR n={b2ar['attention']['n']}; "
        f"AR candidates={ar['candidate_score_count']}."
    )


if __name__ == "__main__":
    main()
