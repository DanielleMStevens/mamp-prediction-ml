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
    """
    Parse a FASTA file into a dictionary of sequences.
    
    Args:
        fasta_path (Path): Path to FASTA file
        
    Returns:
        dict: Mapping of sequence headers to sequences
    """
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
                current_header = line[1:]  # Remove '>' character
                # Remove the final | and number range if present
                if '|' in current_header:
                    current_header = current_header.rsplit('|', 1)[0]
                current_sequence = []
            else:
                current_sequence.append(line)
                
    if current_header:  # Add the last sequence
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
    fasta_path = in_data_dir / "receptor_ectodomains.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError("Could not find receptor sequence file")
        
    return parse_fasta(fasta_path)

def process_data(in_data_dir: Path, use_legacy_columns: bool = True) -> pd.DataFrame:
    """
    Process the raw MAMP prediction data.
    
    Args:
        in_data_dir (Path): Path to input data directory
        use_legacy_columns (bool): If True, use legacy column names (Epitope, Sequence, Known Outcome)
                                 instead of new names (Ligand, Ligand Sequence, Immunogenicity)
    """
    # Load main data
    data_df = pd.read_excel(in_data_dir / "All_LRR_PRR_ligand_data.xlsx")
    
    # Filter out specific literature data
    data_df = data_df[~data_df["Literature_Data"].isin([
        "01_Steinbrenner_2020", 
        "02_Snoeck_2022",
        "22_Kim_2020"
    ])]
    
    # Create receptor-ligand pairs
    receptor_ligand_pairs = data_df[[
        "Plant species", 
        "Receptor", 
        "Ligand",
        "Ligand Sequence", 
        'Immunogenicity'
    ]].drop_duplicates()
    
    # Add receptor name column
    receptor_ligand_pairs['Receptor Name'] = receptor_ligand_pairs.apply(
        lambda x: f"{x['Plant species']}|{x['Receptor']}", 
        axis=1
    )
    
    # Add receptor sequences
    receptor_name_to_seq = load_receptor_sequences(in_data_dir)
    receptor_ligand_pairs['Receptor Sequence'] = receptor_ligand_pairs["Receptor Name"].map(receptor_name_to_seq)
    
    # Filter out rows with missing receptor sequences
    receptor_ligand_pairs = receptor_ligand_pairs.dropna(subset=['Receptor Sequence'])
    
    # Add column renaming if legacy columns option is enabled
    if use_legacy_columns:
        column_mapping = {
            'Ligand': 'Epitope',
            'Ligand Sequence': 'Sequence',
            'Immunogenicity': 'Known Outcome'
        }
        receptor_ligand_pairs = receptor_ligand_pairs.rename(columns=column_mapping)
    
    return receptor_ligand_pairs

def split_with_rare_handling(df: pd.DataFrame, min_samples: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split handling rare cases separately.
    
    Args:
        df (pd.DataFrame): Input dataframe
        min_samples (int): Minimum samples required for a combination to not be considered rare
        
    Returns:
        tuple: (train_df, test_df) DataFrames
    """
    # Identify rare combinations
    combinations = df.groupby(['Sequence', 'Known Outcome']).size()
    rare_combinations = combinations[combinations < min_samples].index
    
    # Split the data into rare and common cases
    rare_mask = df.apply(
        lambda x: (x['Sequence'], x['Known Outcome']) in rare_combinations, 
        axis=1
    )
    rare_cases = df[rare_mask]
    common_cases = df[~rare_mask]
    
    # Split common cases with stratification
    if len(common_cases) > 0:
        stratify_array = [
            (epitope, outcome) 
            for epitope, outcome in zip(
                common_cases['Sequence'], 
                common_cases['Known Outcome']
            )
        ]
        common_train, common_test = train_test_split(
            common_cases, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_array
        )
    else:
        common_train, common_test = pd.DataFrame(), pd.DataFrame()
    
    # Split rare cases without stratification
    if len(rare_cases) > 0:
        rare_train, rare_test = train_test_split(
            rare_cases, 
            test_size=0.2, 
            random_state=42
        )
    else:
        rare_train, rare_test = pd.DataFrame(), pd.DataFrame()
    
    # Combine the results
    train_df = pd.concat([common_train, rare_train])
    test_df = pd.concat([common_test, rare_test])
    
    return train_df, test_df

def main():
    """Main execution function."""
    # Setup paths
    base_dir = Path(__file__).parent.parent
    in_data_dir = base_dir / "in_data"
    out_dir = base_dir / "datasets" / "stratify"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data with legacy column names by default
    receptor_ligand_pairs = process_data(in_data_dir, use_legacy_columns=True)
    
    # Create train/test split with rare case handling
    train_df, test_df = split_with_rare_handling(receptor_ligand_pairs)
    
    # Save processed data
    train_df.to_csv(out_dir / "train_stratify.csv", index=False)
    test_df.to_csv(out_dir / "test_stratify.csv", index=False)
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total samples: {len(receptor_ligand_pairs)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nTraining set distribution:")
    print(train_df.groupby(['Epitope', 'Known Outcome']).size().unstack(fill_value=0))
    print("\nTest set distribution:")
    print(test_df.groupby(['Epitope', 'Known Outcome']).size().unstack(fill_value=0))
    print("\nReceptor sequences:")
    receptor_sequences = load_receptor_sequences(in_data_dir)
    for receptor_name, sequence in receptor_sequences.items():
        print(f"- {receptor_name}: {len(sequence)} amino acids")

if __name__ == "__main__":
    main() 