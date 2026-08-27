"""Command-line interface for scoring new proteins with AlloGator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from .predict import (
    make_protein_input,
    prepare_output_path,
    predict_records,
    read_batch_csv,
    read_single_fasta,
    write_prediction_csv,
)


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="inference backend (default: auto selects MPS, then CUDA, then CPU)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=_positive_integer,
        default=None,
        help="number of PyTorch CPU threads (default: PyTorch setting)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("allogator_scores.csv"),
        help="combined residue-score CSV (default: allogator_scores.csv)",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="write only ranked candidates, omitting active-site and adjacent rows",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allogator",
        description=(
            "Rank candidate allosteric residues from a protein sequence and "
            "one-based active-site positions."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("score", help="score one protein")
    source = single.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="amino-acid sequence")
    source.add_argument("--fasta", type=Path, help="single-record FASTA file")
    single.add_argument(
        "--active-residues",
        required=True,
        help="one-based active-site positions, e.g. 42,67,105",
    )
    single.add_argument(
        "--protein-id",
        default=None,
        help="output identifier (default: FASTA ID or 'protein')",
    )
    _add_runtime_options(single)

    batch = subparsers.add_parser("batch", help="score proteins from a CSV")
    batch.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with protein_id, sequence, and active_residues columns",
    )
    _add_runtime_options(batch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_path: Path | None = None
        if args.command == "score":
            if args.fasta is not None:
                input_path = args.fasta
                fasta_id, raw_sequence = read_single_fasta(args.fasta)
                protein_id = args.protein_id or fasta_id
            else:
                raw_sequence = args.sequence
                protein_id = args.protein_id or "protein"
            records = [
                make_protein_input(
                    protein_id, raw_sequence, args.active_residues
                )
            ]
        else:
            input_path = args.input_csv
            records = read_batch_csv(args.input_csv)

        output_path = prepare_output_path(args.output, input_path)
        if args.cpu_threads is not None:
            torch.set_num_threads(args.cpu_threads)
        rows, selected_device = predict_records(
            records,
            device=args.device,
            candidate_only=args.candidate_only,
        )
        write_prediction_csv(rows, output_path)
        candidate_count = sum(bool(row["is_candidate"]) for row in rows)
        print(
            f"Saved {len(rows)} rows ({candidate_count} ranked candidates) for "
            f"{len(records)} protein(s) to {output_path} using {selected_device}"
        )
        return 0
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"allogator: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
