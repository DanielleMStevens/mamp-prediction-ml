#!/usr/bin/env python3
"""
Scramble peptide and receptor sequences for a composition-preserving control.

Reads the immuno-stratify training CSV, shuffles the amino-acid order of each
peptide (`Sequence`) and receptor (`Receptor Sequence`) independently (length
and composition are unchanged), and writes a new training CSV. The test set is
left unaltered. Per-residue chemical features are permuted with the same
residue order as their matching sequence so they stay aligned after scrambling.
"""

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple


PEPTIDE_SEQUENCE_COLUMN = "Sequence"
RECEPTOR_SEQUENCE_COLUMN = "Receptor Sequence"
PEPTIDE_CHEMICAL_COLUMNS = (
    "Sequence_Bulkiness",
    "Sequence_Charge",
    "Sequence_Hydrophobicity",
)
RECEPTOR_CHEMICAL_COLUMNS = (
    "Receptor_Bulkiness",
    "Receptor_Charge",
    "Receptor_Hydrophobicity",
)


def scramble_sequence(
    sequence: str,
    rng: random.Random,
    max_attempts: int = 100,
) -> Tuple[str, List[int]]:
    """Shuffle residues in `sequence` and return (scrambled, permutation).

    The permutation maps original index -> new index of that residue. If a
    sequence can be rearranged into a different order, retry until it differs
    from the original (or until max_attempts is reached).
    """
    residues = list(sequence)
    n = len(residues)
    original_order = list(range(n))

    if n <= 1:
        return sequence, original_order

    permutation = original_order[:]
    scrambled = sequence
    for _ in range(max_attempts):
        permutation = original_order[:]
        rng.shuffle(permutation)
        scrambled = "".join(residues[i] for i in permutation)
        if scrambled != sequence:
            break
    return scrambled, permutation


def permute_feature_string(feature_string: Optional[str], permutation: List[int]) -> Optional[str]:
    """Apply the same residue permutation to a comma-separated feature string."""
    if feature_string is None:
        return feature_string
    values = feature_string.split(",")
    if len(values) != len(permutation):
        raise ValueError(
            f"Chemical feature length ({len(values)}) does not match "
            f"sequence length ({len(permutation)})."
        )
    return ",".join(values[i] for i in permutation)


def scramble_fields(
    row: Dict[str, str],
    sequence_column: str,
    chemical_columns: Sequence[str],
    rng: random.Random,
) -> bool:
    """Scramble one sequence column and its aligned chemical features.

    Returns True if the scrambled sequence differs from the original.
    """
    sequence = row.get(sequence_column) or ""
    scrambled, permutation = scramble_sequence(sequence, rng)
    row[sequence_column] = scrambled
    for column in chemical_columns:
        if column in row and row[column] not in (None, ""):
            row[column] = permute_feature_string(row[column], permutation)
    return scrambled != sequence


def scramble_row(row: Dict[str, str], rng: random.Random) -> Tuple[Dict[str, str], bool, bool]:
    peptide_changed = scramble_fields(
        row, PEPTIDE_SEQUENCE_COLUMN, PEPTIDE_CHEMICAL_COLUMNS, rng
    )
    receptor_changed = scramble_fields(
        row, RECEPTOR_SEQUENCE_COLUMN, RECEPTOR_CHEMICAL_COLUMNS, rng
    )
    return row, peptide_changed, receptor_changed


def scramble_csv(input_path: str, output_path: str, rng: random.Random) -> int:
    print(f"Reading {input_path}")
    with open(input_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in {input_path}")
        for column in (PEPTIDE_SEQUENCE_COLUMN, RECEPTOR_SEQUENCE_COLUMN):
            if column not in fieldnames:
                raise ValueError(f"Column '{column}' not found in {input_path}")

        rows = []
        n_peptide_changed = 0
        n_receptor_changed = 0
        for row in reader:
            scrambled_row, peptide_changed, receptor_changed = scramble_row(row, rng)
            rows.append(scrambled_row)
            n_peptide_changed += int(peptide_changed)
            n_receptor_changed += int(receptor_changed)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"  Peptide sequences changed: {n_peptide_changed}/{len(rows)}")
    print(f"  Receptor sequences changed: {n_receptor_changed}/{len(rows)}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scramble peptide and receptor Sequence columns in the training CSV only."
        )
    )
    parser.add_argument(
        "--train_csv",
        default="05_datasets/train_data_with_all_train_immuno_stratify.csv",
        help="Input training CSV",
    )
    parser.add_argument(
        "--train_output",
        default="05_datasets/train_data_with_all_train_immuno_scrambled.csv",
        help="Output scrambled training CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffles (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("PEPTIDE + RECEPTOR SEQUENCE SCRAMBLE (TRAIN ONLY)")
    print("=" * 60)
    print(f"Random seed: {args.seed}")

    n_train = scramble_csv(args.train_csv, args.train_output, rng)

    print("=" * 60)
    print(f"Train rows: {n_train}")
    print("Test set is unchanged.")
    print("=" * 60)


if __name__ == "__main__":
    main()
