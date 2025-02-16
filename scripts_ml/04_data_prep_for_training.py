#!/usr/bin/env python3

"""
Data preparation script for MAMP prediction model.

This script processes:
1. MAMP prediction data from Excel files
2. Receptor ectodomain sequences from FASTA files

The script expects the following files in the input directory:
- All_LRR_PRR_ligand_data.xlsx: Main data file
- receptor_ectodomains.fasta: Combined receptor sequences file
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse a FASTA file into a dictionary of sequences."""
    sequences = {}
    current_header = None
    current_sequence = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                
                header = line[1:]
                parts = header.split('|')
                if len(parts) >= 3:
                    species = parts[0]
                    locus_id = parts[1]
                    receptor = parts[2]
                    current_header = f"{species}|{locus_id}|{receptor}"
                else:
                    current_header = header
                
                current_sequence = []
            else:
                current_sequence.append(line)
                
    if current_header:
        sequences[current_header] = ''.join(current_sequence)
    
    return sequences

def load_receptor_sequences(in_data_dir: Path) -> Dict[str, str]:
    """
    Load receptor sequences from the combined FASTA file.
    
    Args:
        in_data_dir (Path): Path to input data directory
        
    Returns:
        dict: Mapping of receptor names to their sequences
    """
    fasta_path = in_data_dir.parent / "out_data" / "lrr_domain_sequences.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"Could not find receptor sequence file at {fasta_path}")
        
    return parse_fasta(fasta_path)

def process_data(in_data_dir: Path, use_legacy_columns: bool = True) -> pd.DataFrame:
    """
    Process the raw MAMP prediction data.
    
    Args:
        in_data_dir (Path): Path to input data directory
        use_legacy_columns (bool): If True, use legacy column names (Epitope, Sequence, Known Outcome)
                                 instead of new names (Ligand, Ligand Sequence, Immunogenicity)
    """
    excel_path = in_data_dir / "All_LRR_PRR_ligand_data.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel data file at {excel_path}")
    
    data_df = pd.read_excel(excel_path)
    
    required_columns = ["Plant species", "Receptor", "Locus ID/Genbank", "Ligand", "Ligand Sequence", "Immunogenicity"]
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Excel file: {missing_columns}")

    receptor_ligand_pairs = data_df[required_columns].drop_duplicates()
    print(f"Unique receptor-ligand pairs: {len(receptor_ligand_pairs)}")
    
    receptor_ligand_pairs['Receptor Name'] = receptor_ligand_pairs.apply(
        lambda x: f"{x['Plant species']}|{x['Locus ID/Genbank']}|{x['Receptor']}", 
        axis=1
    )
    
    receptor_name_to_seq = load_receptor_sequences(in_data_dir)
    receptor_ligand_pairs['Receptor Sequence'] = receptor_ligand_pairs["Receptor Name"].map(receptor_name_to_seq)
    
    before_filter = len(receptor_ligand_pairs)
    receptor_ligand_pairs = receptor_ligand_pairs.dropna(subset=['Receptor Sequence'])
    after_filter = len(receptor_ligand_pairs)
    print(f"Filtered out {before_filter - after_filter} rows with missing receptor sequences")
    
    if use_legacy_columns:
        column_mapping = {
            'Ligand': 'Epitope',
            'Ligand Sequence': 'Sequence',
            'Immunogenicity': 'Known Outcome'
        }
        receptor_ligand_pairs = receptor_ligand_pairs.rename(columns=column_mapping)
    
    return receptor_ligand_pairs

def split_with_rare_handling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split with stratification.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['Known Outcome']
        )
    except ValueError as e:
        print("Warning: Could not stratify by Known Outcome, falling back to random split")
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42
        )
    
    return train_df, test_df

def main():
    """Main execution function."""
    base_dir = Path(__file__).parent.parent
    in_data_dir = base_dir / "in_data"
    out_dir = base_dir / "datasets" / "stratify"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    receptor_ligand_pairs = process_data(in_data_dir, use_legacy_columns=True)
    train_df, test_df = split_with_rare_handling(receptor_ligand_pairs)
    
    train_df.to_csv(out_dir / "train_stratify.csv", index=False)
    test_df.to_csv(out_dir / "test_stratify.csv", index=False)
    
    print("\nDataset statistics:")
    print(f"Total samples: {len(receptor_ligand_pairs)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nClass distribution:")
    print("Training set:")
    print(train_df.groupby(['Known Outcome']).size())
    print("\nTest set:")
    print(test_df.groupby(['Known Outcome']).size())

if __name__ == "__main__":
    main() 