"""Core input validation, prediction, ranking, and output for AlloGator."""
from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import rankdata

from .attention import active_site_attention_scores, free_model, load_model
from .candidates import candidate_mask


MAX_SEQUENCE_LENGTH = 1022
SUPPORTED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")
OUTPUT_COLUMNS = [
    "protein_id",
    "residue_number",
    "amino_acid",
    "attention_score",
    "rank",
    "rank_percentile",
    "is_active_site",
    "is_sequence_adjacent",
    "is_candidate",
]


@dataclass(frozen=True)
class ProteinInput:
    protein_id: str
    sequence: str
    active_residues: tuple[int, ...]


def normalize_sequence(raw_sequence: Any) -> str:
    """Normalize a protein sequence and validate the ESM-1b alphabet/length."""
    if raw_sequence is None:
        raise ValueError("sequence is missing")
    sequence = re.sub(r"\s+", "", str(raw_sequence)).upper()
    if not sequence:
        raise ValueError("sequence is empty")
    invalid = sorted(set(sequence) - SUPPORTED_RESIDUES)
    if invalid:
        symbols = ", ".join(repr(symbol) for symbol in invalid)
        raise ValueError(f"sequence contains unsupported symbol(s): {symbols}")
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence has {len(sequence)} residues; ESM-1b supports at most "
            f"{MAX_SEQUENCE_LENGTH}"
        )
    return sequence


def parse_active_residues(raw_positions: Any, sequence_length: int) -> tuple[int, ...]:
    """Parse comma-, semicolon-, or whitespace-separated one-based positions."""
    if raw_positions is None:
        raise ValueError("active_residues is missing")
    text = str(raw_positions).strip()
    if not text:
        raise ValueError("at least one active-site residue is required")
    parts = [part for part in re.split(r"[\s,;]+", text) if part]
    try:
        positions = sorted({int(part) for part in parts})
    except ValueError as exc:
        raise ValueError(
            "active_residues must contain one-based integers separated by "
            "commas, semicolons, or spaces"
        ) from exc
    if not positions:
        raise ValueError("at least one active-site residue is required")
    outside = [position for position in positions if not 1 <= position <= sequence_length]
    if outside:
        raise ValueError(
            f"active-site position(s) {outside} fall outside the sequence range "
            f"1..{sequence_length}"
        )
    return tuple(positions)


def make_protein_input(
    protein_id: Any,
    raw_sequence: Any,
    raw_active_residues: Any,
) -> ProteinInput:
    """Validate and construct one prediction input."""
    identifier = "" if protein_id is None else str(protein_id).strip()
    if not identifier:
        raise ValueError("protein_id is empty")
    sequence = normalize_sequence(raw_sequence)
    active = parse_active_residues(raw_active_residues, len(sequence))
    if not candidate_mask(len(sequence), active).any():
        raise ValueError(
            "active-site and sequence-adjacent exclusions leave no candidate residues"
        )
    return ProteinInput(identifier, sequence, active)


def read_single_fasta(path: Path) -> tuple[str, str]:
    """Read exactly one FASTA record and return ``(identifier, sequence)``."""
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_lines)))
            header = line[1:].strip()
            sequence_lines = []
            if not header:
                raise ValueError(f"{path}: FASTA header on line {line_number} is empty")
        elif header is None:
            raise ValueError(f"{path}: sequence appears before the first FASTA header")
        else:
            sequence_lines.append(line)
    if header is not None:
        records.append((header, "".join(sequence_lines)))
    if len(records) != 1:
        raise ValueError(
            f"{path}: expected exactly one FASTA record, found {len(records)}; "
            "use batch mode for multiple proteins"
        )
    header, sequence = records[0]
    identifier = header.split()[0]
    return identifier, sequence


def read_batch_csv(path: Path) -> list[ProteinInput]:
    """Read and prevalidate a batch CSV before the model is loaded."""
    if not path.is_file():
        raise FileNotFoundError(f"batch CSV not found: {path}")
    required = {"protein_id", "sequence", "active_residues"}
    records: list[ProteinInput] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        duplicates = sorted(
            {
                name
                for name in fieldnames
                if fieldnames.count(name) > 1
            },
            key=lambda name: "" if name is None else name,
        )
        if duplicates:
            labels = ", ".join(repr(name) for name in duplicates)
            raise ValueError(f"{path}: duplicate CSV column name(s): {labels}")
        fields = set(fieldnames)
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(missing)}"
            )
        for row in reader:
            if None in row:
                raise ValueError(
                    f"{path}: row {reader.line_num}: too many CSV fields; "
                    "separate active-site positions with semicolons or quote "
                    "a comma-separated list"
                )
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                record = make_protein_input(
                    row.get("protein_id"),
                    row.get("sequence"),
                    row.get("active_residues"),
                )
            except ValueError as exc:
                raise ValueError(f"{path}: row {reader.line_num}: {exc}") from exc
            if record.protein_id in seen:
                raise ValueError(
                    f"{path}: row {reader.line_num}: duplicate protein_id "
                    f"{record.protein_id!r}"
                )
            seen.add(record.protein_id)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: batch CSV contains no protein records")
    return records


def build_prediction_rows(
    record: ProteinInput,
    residue_scores: Sequence[float],
    candidate_only: bool = False,
) -> list[dict[str, Any]]:
    """Build transparent per-residue output and rank candidate residues."""
    scores = np.asarray(residue_scores, dtype=np.float64)
    length = len(record.sequence)
    if scores.shape != (length,):
        raise ValueError(
            f"{record.protein_id}: expected {length} residue scores, "
            f"found shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{record.protein_id}: residue scores contain non-finite values")

    mask = candidate_mask(length, record.active_residues)
    candidate_indices = np.flatnonzero(mask)
    candidate_scores = scores[candidate_indices]
    descending_ranks = rankdata(-candidate_scores, method="average")
    ascending_ranks = rankdata(candidate_scores, method="average")
    if len(candidate_indices) == 1:
        percentiles = np.ones(1, dtype=np.float64)
    else:
        percentiles = (ascending_ranks - 1) / (len(candidate_indices) - 1)
    rank_by_index = {
        int(index): (float(rank), float(percentile))
        for index, rank, percentile in zip(
            candidate_indices, descending_ranks, percentiles, strict=True
        )
    }
    active_set = set(record.active_residues)
    rows: list[dict[str, Any]] = []
    for index, (amino_acid, score) in enumerate(
        zip(record.sequence, scores, strict=True)
    ):
        position = index + 1
        is_active = position in active_set
        is_adjacent = not is_active and any(
            abs(position - active_position) == 1
            for active_position in record.active_residues
        )
        is_candidate = bool(mask[index])
        if candidate_only and not is_candidate:
            continue
        rank, percentile = rank_by_index.get(index, (None, None))
        rows.append(
            {
                "protein_id": record.protein_id,
                "residue_number": position,
                "amino_acid": amino_acid,
                "attention_score": float(score),
                "rank": rank,
                "rank_percentile": percentile,
                "is_active_site": is_active,
                "is_sequence_adjacent": is_adjacent,
                "is_candidate": is_candidate,
            }
        )
    return rows


def predict_records(
    records: Sequence[ProteinInput],
    device: str = "auto",
    candidate_only: bool = False,
    model_loader: Callable[..., tuple[Any, Any, int, Any]] | None = None,
    sequence_scorer: Callable[..., np.ndarray] | None = None,
    model_free: Callable[[Any], None] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load ESM-1b once and score validated proteins sequentially."""
    if not records:
        raise ValueError("no proteins were provided")
    loader = model_loader or load_model
    scorer = sequence_scorer or active_site_attention_scores
    release = model_free or free_model
    model, alphabet, n_layers, selected_device = loader("ESM1b", device=device)
    rows: list[dict[str, Any]] = []
    try:
        print(f"Loaded ESM-1b on {selected_device}")
        for index, record in enumerate(records, 1):
            print(
                f"[{index}/{len(records)}] scoring {record.protein_id} "
                f"({len(record.sequence)} residues)"
            )
            try:
                scores = scorer(
                    model,
                    alphabet,
                    selected_device,
                    record.sequence,
                    record.active_residues,
                    n_layers,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{record.protein_id}: ESM-1b inference failed on "
                    f"{selected_device}: {exc}. Try --device cpu or a shorter sequence "
                    "if this is an out-of-memory error."
                ) from exc
            rows.extend(build_prediction_rows(record, scores, candidate_only))
    finally:
        release(model)
    return rows, str(selected_device)


def prepare_output_path(output_path: Path, input_path: Path | None = None) -> Path:
    """Reject input collisions and verify output writability before inference."""
    output = output_path.expanduser()
    resolved_output = output.resolve()
    if input_path is not None and resolved_output == input_path.expanduser().resolve():
        raise ValueError("output path must differ from the input FASTA or batch CSV")
    if output.exists() and output.is_dir():
        raise ValueError(f"output path is a directory: {output}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent):
            pass
    except OSError as exc:
        raise OSError(f"cannot write to output directory {output.parent}: {exc}") from exc
    return output


def write_prediction_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Atomically write the combined prediction table."""
    rows = list(rows)
    if not rows:
        raise ValueError("prediction output contains no rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
