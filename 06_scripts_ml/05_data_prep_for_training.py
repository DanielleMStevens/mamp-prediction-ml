#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------
"""
Data preparation script for MAMP prediction model.

This script processes and prepares data for training a MAMP (Microbe-Associated Molecular Pattern) 
prediction model by:
1. Loading and processing MAMP prediction data from Excel files
2. Loading and parsing receptor ectodomain sequences from FASTA files
3. Combining the data and creating stratified train/test splits

Input Files Required:
- All_LRR_PRR_ligand_data.xlsx: Contains MAMP-receptor interaction data
- lrr_domain_sequences.fasta: Contains receptor protein sequences in FASTA format

Output:
- Generates train and test CSV files with processed data for model training
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """
    Parse a FASTA format file into a dictionary of sequences.
    
    Args:
        fasta_path (Path): Path to the FASTA file
        
    Returns:
        Dict[str, str]: Dictionary mapping sequence headers to their corresponding sequences
        
    Note:
        Expected FASTA header format: >species|locus_id|receptor
        Headers are parsed to create a standardized key format
    """
    sequences = {}
    current_header = None
    current_sequence = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Handle FASTA header lines (starting with '>')
            if line.startswith('>'):
                # Save the previous sequence if it exists
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                
                # Parse the new header
                header = line[1:]  # Remove '>' character
                parts = header.split('|')
                if len(parts) >= 3:
                    # Standard format: species|locus_id|receptor
                    species = parts[0]
                    locus_id = parts[1]
                    receptor = parts[2]
                    current_header = f"{species}|{locus_id}|{receptor}"
                else:
                    # Fallback for non-standard headers
                    current_header = header
                
                current_sequence = []
            else:
                # Accumulate sequence lines
                current_sequence.append(line)
                
    # Save the last sequence
    if current_header:
        sequences[current_header] = ''.join(current_sequence)
    
    return sequences

def load_receptor_sequences(in_data_dir: Path) -> Dict[str, str]:
    """
    Load receptor protein sequences from the FASTA file.
    
    Args:
        in_data_dir (Path): Base input directory path
        
    Returns:
        Dict[str, str]: Dictionary mapping receptor identifiers to their sequences
        
    Raises:
        FileNotFoundError: If the FASTA file is not found at the expected location
    """
    fasta_path = in_data_dir.parent / "03_out_data" / "lrr_domain_sequences.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"Could not find receptor sequence file at {fasta_path}")
        
    return parse_fasta(fasta_path)

def process_data(in_data_dir: Path, use_legacy_columns: bool = True) -> pd.DataFrame:
    """
    Process and combine MAMP prediction data from Excel and FASTA files.
    
    Args:
        in_data_dir (Path): Directory containing input data files
        use_legacy_columns (bool): If True, use old column names (Epitope, Sequence, Known Outcome)
                                 If False, use new names (Ligand, Ligand Sequence, Immunogenicity)
    
    Returns:
        pd.DataFrame: Processed dataframe containing combined receptor-ligand data
        
    Raises:
        FileNotFoundError: If required Excel file is not found
        ValueError: If required columns are missing from Excel file
    """


    # Load Excel data
    excel_path = in_data_dir / "All_LRR_PRR_ligand_data.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel data file at {excel_path}")
    
    data_df = pd.read_excel(excel_path)
    
    # Validate required columns
    required_columns = ["Plant species", "Receptor", "Locus ID/Genbank", "Ligand", "Ligand Sequence", "Immunogenicity"]
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Excel file: {missing_columns}")

    # Extract unique receptor-ligand pairs
    receptor_ligand_pairs = data_df[required_columns].drop_duplicates()
    print(f"Unique receptor-ligand pairs: {len(receptor_ligand_pairs)}")
    
    # Create standardized receptor identifiers
    receptor_ligand_pairs['Receptor Name'] = receptor_ligand_pairs.apply(
        lambda x: f"{x['Plant species']}|{x['Locus ID/Genbank']}|{x['Receptor']}", 
        axis=1
    )
    
    # Add receptor sequences
    receptor_name_to_seq = load_receptor_sequences(in_data_dir)
    receptor_ligand_pairs['Receptor Sequence'] = receptor_ligand_pairs["Receptor Name"].map(receptor_name_to_seq)
    
    # Log missing sequences
    missing_receptors = receptor_ligand_pairs[receptor_ligand_pairs['Receptor Sequence'].isna()]
    print("\nMissing receptor sequences for:")
    for _, row in missing_receptors.iterrows():
        print(f"- {row['Receptor Name']} ({row['Plant species']}, {row['Receptor']})")
    
    # Remove entries with missing sequences
    before_filter = len(receptor_ligand_pairs)
    receptor_ligand_pairs = receptor_ligand_pairs.dropna(subset=['Receptor Sequence'])
    after_filter = len(receptor_ligand_pairs)
    print(f"\nFiltered out {before_filter - after_filter} rows with missing receptor sequences")
    
    
    # Handle legacy column names if needed
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
    Create a stratified train/test split while handling potential rare classes.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the full dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test dataframes
        
    Note:
        - Attempts stratified split based on 'Known Outcome' column
        - Falls back to random split if stratification fails (e.g., due to rare classes)
        - Uses 80/20 train/test split ratio
    """
    try:
        # Attempt stratified split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['Known Outcome']
        )
    except ValueError as e:
        print("Warning: Could not stratify by Known Outcome, falling back to random split")
        # Fallback to random split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42
        )
    
    return train_df, test_df

def main():
    """
    Main execution function that orchestrates the data preparation process.
    
    Steps:
    1. Set up input/output paths
    2. Process raw data from Excel and FASTA files
    3. Create train/test splits
    4. Save processed datasets
    5. Print dataset statistics
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    in_data_dir = base_dir / "02_in_data"
    out_dir = base_dir / "05_datasets" 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data and create splits
    receptor_ligand_pairs = process_data(in_data_dir, use_legacy_columns=True)
    train_df, test_df = split_with_rare_handling(receptor_ligand_pairs)
    
    # Save processed datasets
    train_df.to_csv(out_dir / "train_stratify.csv", index=False)
    test_df.to_csv(out_dir / "test_stratify.csv", index=False)
    
    # Print statistics
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