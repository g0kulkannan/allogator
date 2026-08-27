"""Compute the ESM-1b contact-removal and single-layer/head analyses.

The ESM-1b controls in Figures S2 and S4 share a single forward pass between:

* contact-removed mean-attention residue scores and per-protein AUROC;
* per-layer AUROC (mean over the 20 heads in one layer); and
* per-head AUROC for all 33 x 20 individual attention heads.

An ESM-1b contact probability greater than the within-protein mean plus one
standard deviation is removed from the mean attention map. Active-site
residues and residues one sequence position away are excluded from the
candidate set.

By default, outputs are written to ``manuscript/data/cached_results``. Use
``--output-root`` to write them elsewhere.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from allobench.attention import load_model
from reproduction.io import load_benchmark
from reproduction.paths import CACHED
from reproduction.scoring import candidate_mask


N_LAYERS = 33
N_HEADS = 20
FLUSH_EVERY = 1
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CACHED,
        help="directory containing ESM1b_contacts/ and PerLayer/ outputs",
    )
    parser.add_argument(
        "--with-per-layer",
        action="store_true",
        help="also emit the 109x33 per-layer and 109x660 per-head tables",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N pending proteins",
    )
    return parser.parse_args()


def _attention_and_contacts(
    model: Any,
    alphabet: Any,
    device: torch.device,
    sequence: str,
    n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full attentions and the ESM-1b contact-probability matrix."""
    batch_converter = alphabet.get_batch_converter()
    with torch.no_grad():
        _, _, tokens = batch_converter([("seq", sequence)])
        tokens = tokens.to(device)
        result = model(tokens, repr_layers=[n_layers], return_contacts=True)
        attention = result["attentions"][0].detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        contacts = result["contacts"][0, : len(sequence), : len(sequence)]
        contacts = contacts.detach().cpu().numpy().astype(np.float32, copy=False)
        del result, tokens
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return attention, contacts


def _labels_and_indices(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence = entry["sequence"]
    active = sorted(int(x) for x in entry["active"])
    allosteric = {int(x) for x in entry["allo"]}
    length = len(sequence)
    if not active or not allosteric:
        raise ValueError("benchmark entry has an empty active or allosteric site")
    if max(active) > length or max(allosteric) > length:
        raise ValueError("benchmark residue is outside the canonical sequence")
    candidate_indices = np.where(candidate_mask(length, active))[0]
    active_indices = np.asarray([x - 1 for x in active], dtype=np.int64)
    labels = np.asarray(
        [1 if int(i + 1) in allosteric else 0 for i in candidate_indices],
        dtype=np.int8,
    )
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("candidate set does not contain both AUROC classes")
    return active_indices, candidate_indices, labels


def _score_contact_removed(
    attention: np.ndarray,
    contacts: np.ndarray,
    active_indices: np.ndarray,
    candidate_indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, float], float]:
    """Apply the contact threshold and score candidate residues."""
    residue_attention = attention.mean(axis=(0, 1))[1:-1, 1:-1]
    contact_cutoff = float(contacts.mean() + contacts.std())
    keep = contacts <= contact_cutoff

    # Diagonal and sequence-adjacent cells cannot enter the score because those
    # positions are excluded by candidate_mask.
    scores = (residue_attention[active_indices, :] * keep[active_indices, :]).sum(
        axis=0
    )
    candidate_scores = scores[candidate_indices]
    auroc = float(roc_auc_score(labels, candidate_scores))
    score_map = {
        str(int(position + 1)): float(score)
        for position, score in zip(candidate_indices, candidate_scores, strict=True)
    }
    return auroc, score_map, contact_cutoff


def _score_layers_and_heads(
    attention: np.ndarray,
    active_indices: np.ndarray,
    candidate_indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 33 layer AUROCs and 33x20 single-head AUROCs."""
    residue_attention = attention[:, :, 1:-1, 1:-1]
    selected = residue_attention[
        :, :, active_indices[:, None], candidate_indices[None, :]
    ]
    head_scores = selected.sum(axis=2)  # (33, 20, number of candidates)

    per_head = np.empty((N_LAYERS, N_HEADS), dtype=np.float64)
    per_layer = np.empty(N_LAYERS, dtype=np.float64)
    for layer_index in range(N_LAYERS):
        for head_index in range(N_HEADS):
            per_head[layer_index, head_index] = roc_auc_score(
                labels, head_scores[layer_index, head_index]
            )
        per_layer[layer_index] = roc_auc_score(
            labels, head_scores[layer_index].mean(axis=0)
        )
    return per_layer, per_head.reshape(N_LAYERS * N_HEADS)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_table(path: Path, accessions: list[str], n_columns: int) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path, index_col=0)
        frame.columns = [int(x) for x in frame.columns]
        return frame.reindex(index=accessions, columns=range(n_columns))
    return pd.DataFrame(index=accessions, columns=range(n_columns), dtype=float)


def _attach_permutation_p_values(
    benchmark: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    residue_scores: dict[str, Any],
) -> None:
    """Attach deterministic one-sided permutation p-values for Fig. S2A.

    The RNG and in-place label shuffling match ``figS2a_overlap_compact.py``.
    Rank-sum AUROC is used inside the permutation loop; it is algebraically
    identical to repeatedly calling ``sklearn.metrics.roc_auc_score``.
    """
    if all(
        isinstance(metrics.get(accession, {}).get("AUROC"), list)
        and len(metrics[accession]["AUROC"]) >= 2
        for accession in benchmark
    ):
        return
    rng = np.random.default_rng(PERMUTATION_SEED)
    for accession in tqdm(sorted(benchmark), desc="contact permutations"):
        entry = benchmark[accession]
        _, candidate_indices, labels = _labels_and_indices(entry)
        score_map = residue_scores[accession]
        values = np.asarray(
            [float(score_map[str(int(i + 1))]) for i in candidate_indices],
            dtype=np.float64,
        )
        ranks = rankdata(values, method="average")
        n_positive = int(labels.sum())
        n_negative = int(labels.size - n_positive)
        offset = n_positive * (n_positive + 1) / 2
        denominator = n_positive * n_negative
        observed = float(metrics[accession]["AUROC"][0])
        shuffled = labels.copy()
        at_least_observed = 0
        for _ in range(N_PERMUTATIONS):
            rng.shuffle(shuffled)
            permuted = (float(ranks[shuffled == 1].sum()) - offset) / denominator
            at_least_observed += int(permuted >= observed)
        metrics[accession] = {
            "AUROC": [observed, at_least_observed / N_PERMUTATIONS]
        }


def main() -> None:
    args = _parse_args()
    benchmark = load_benchmark()
    accessions = sorted(benchmark)
    if len(accessions) != 109:
        raise RuntimeError(f"expected 109 benchmark proteins, found {len(accessions)}")

    contact_dir = args.output_root / "ESM1b_contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = contact_dir / "per_protein_auroc.json"
    scores_path = contact_dir / "residue_scores.json"
    metadata_path = contact_dir / "method_metadata.json"
    metrics = _read_json(metrics_path)
    residue_scores = _read_json(scores_path)
    contact_cutoffs: dict[str, float] = {}
    if metadata_path.exists():
        contact_cutoffs = _read_json(metadata_path).get("contact_cutoffs", {})

    layer_frame: pd.DataFrame | None = None
    head_frame: pd.DataFrame | None = None
    layer_path: Path | None = None
    head_path: Path | None = None
    if args.with_per_layer:
        per_layer_dir = args.output_root / "PerLayer"
        per_layer_dir.mkdir(parents=True, exist_ok=True)
        layer_path = per_layer_dir / "per_layer_auroc.csv"
        head_path = per_layer_dir / "per_head_auroc.csv"
        layer_frame = _read_table(layer_path, accessions, N_LAYERS)
        head_frame = _read_table(head_path, accessions, N_LAYERS * N_HEADS)

    def complete(accession: str) -> bool:
        contact_done = accession in metrics and accession in residue_scores
        if not args.with_per_layer:
            return contact_done
        assert layer_frame is not None and head_frame is not None
        return (
            contact_done
            and layer_frame.loc[accession].notna().all()
            and head_frame.loc[accession].notna().all()
        )

    pending = [accession for accession in accessions if not complete(accession)]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"[esm1b-supplementary] benchmark={len(accessions)} "
        f"complete={len(accessions) - len([u for u in accessions if not complete(u)])} "
        f"running={len(pending)} output={args.output_root}"
    )
    if not pending:
        if len(metrics) == len(accessions) and len(residue_scores) == len(accessions):
            _attach_permutation_p_values(benchmark, metrics, residue_scores)
            _write_json(metrics_path, metrics)
        return

    model, alphabet, n_layers, device = load_model("ESM1b")
    if n_layers != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} ESM-1b layers, found {n_layers}")
    print(f"[esm1b-supplementary] model on {device}")

    started = time.time()
    since_flush = 0
    for accession in tqdm(pending, desc="ESM-1b supplementary"):
        entry = benchmark[accession]
        active_indices, candidate_indices, labels = _labels_and_indices(entry)
        attention, contacts = _attention_and_contacts(
            model, alphabet, device, entry["sequence"], n_layers
        )
        if attention.shape[:2] != (N_LAYERS, N_HEADS):
            raise RuntimeError(
                f"{accession}: unexpected attention shape {attention.shape}"
            )
        expected_length = len(entry["sequence"])
        if contacts.shape != (expected_length, expected_length):
            raise RuntimeError(
                f"{accession}: unexpected contacts shape {contacts.shape}"
            )

        auroc, scores, cutoff = _score_contact_removed(
            attention, contacts, active_indices, candidate_indices, labels
        )
        metrics[accession] = {"AUROC": [auroc]}
        residue_scores[accession] = scores
        contact_cutoffs[accession] = cutoff

        if args.with_per_layer:
            assert layer_frame is not None and head_frame is not None
            layer_values, head_values = _score_layers_and_heads(
                attention, active_indices, candidate_indices, labels
            )
            layer_frame.loc[accession] = layer_values
            head_frame.loc[accession] = head_values

        del attention, contacts
        gc.collect()
        since_flush += 1
        if since_flush >= FLUSH_EVERY:
            _write_json(metrics_path, metrics)
            _write_json(scores_path, residue_scores)
            if args.with_per_layer:
                assert layer_frame is not None and head_frame is not None
                assert layer_path is not None and head_path is not None
                layer_frame.to_csv(layer_path)
                head_frame.to_csv(head_path)
            since_flush = 0

    if len(metrics) == len(accessions) and len(residue_scores) == len(accessions):
        _attach_permutation_p_values(benchmark, metrics, residue_scores)
    _write_json(metrics_path, metrics)
    _write_json(scores_path, residue_scores)
    if args.with_per_layer:
        assert layer_frame is not None and head_frame is not None
        assert layer_path is not None and head_path is not None
        layer_frame.to_csv(layer_path)
        head_frame.to_csv(head_path)

    complete_accessions = [u for u in accessions if complete(u)]
    metadata = {
        "model": "esm1b_t33_650M_UR50S",
        "benchmark_proteins": len(accessions),
        "complete_proteins": len(complete_accessions),
        "contact_rule": "remove contact_probability > within-protein mean + 1 SD",
        "candidate_rule": "exclude active-site residues and sequence-adjacent positions (+/-1)",
        "attention_rule": "mean across all layers and heads; sum active-site rows",
        "per_layer_column_indexing": "0-based columns 0..32; Figure S4 displays layers 0..32",
        "per_head_column_indexing": "column = zero_based_layer * 20 + zero_based_head",
        "contact_cutoffs": {u: contact_cutoffs[u] for u in sorted(contact_cutoffs)},
    }
    _write_json(metadata_path, metadata)
    elapsed = (time.time() - started) / 60
    print(
        f"[esm1b-supplementary] complete={len(complete_accessions)}/109 "
        f"elapsed={elapsed:.1f} min"
    )


if __name__ == "__main__":
    main()
