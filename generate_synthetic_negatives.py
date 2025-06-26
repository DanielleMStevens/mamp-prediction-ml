#!/usr/bin/env python3
"""
Script to generate synthetic negative training data for MAMP prediction ML.

This script takes the original training data and creates synthetic negative examples
by deliberately mismatching receptors with epitopes that don't naturally pair together.
The synthetic data is labeled as "Non-Immunogenic" and appended to the original data.
"""

import pandas as pd
import numpy as np
import random
from itertools import product
import argparse
import os

def load_training_data(filepath):
    """Load the training data CSV file."""
    print(f"Loading training data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def get_unique_pairs(df):
    """Get unique receptor-epitope pairs from the data."""
    # Get unique combinations of receptor and epitope
    unique_pairs = df[['Receptor', 'Epitope']].drop_duplicates()
    print(f"Found {len(unique_pairs)} unique receptor-epitope pairs")
    return unique_pairs

def generate_synthetic_negatives(df, num_synthetic_per_original=1):
    """
    Generate synthetic negative examples by mismatching receptors and epitopes.
    
    Args:
        df: Original training dataframe
        num_synthetic_per_original: Number of synthetic examples to create per original example
    
    Returns:
        DataFrame with synthetic negative examples
    """
    print(f"Generating synthetic negatives with ratio {num_synthetic_per_original}:1")
    
    # Get all unique receptors and epitopes
    unique_receptors = df['Receptor'].unique()
    unique_epitopes = df['Epitope'].unique()
    
    print(f"Found {len(unique_receptors)} unique receptors: {list(unique_receptors)[:10]}...")
    print(f"Found {len(unique_epitopes)} unique epitopes: {list(unique_epitopes)[:10]}...")
    
    # Get existing receptor-epitope pairs to avoid duplicates
    existing_pairs = set(zip(df['Receptor'], df['Epitope']))
    print(f"Existing pairs: {len(existing_pairs)}")
    
    # Generate all possible receptor-epitope combinations
    all_possible_pairs = set(product(unique_receptors, unique_epitopes))
    print(f"All possible combinations: {len(all_possible_pairs)}")
    
    # Find mismatched pairs (combinations that don't exist in original data)
    mismatched_pairs = list(all_possible_pairs - existing_pairs)
    print(f"Available mismatched pairs: {len(mismatched_pairs)}")
    
    # Calculate how many synthetic examples to create
    target_synthetic_count = len(df) * num_synthetic_per_original
    
    if len(mismatched_pairs) < target_synthetic_count:
        print(f"Warning: Only {len(mismatched_pairs)} mismatched pairs available, but {target_synthetic_count} requested.")
        print("Will use all available mismatched pairs.")
        selected_pairs = mismatched_pairs
    else:
        # Randomly sample the required number of mismatched pairs
        selected_pairs = random.sample(mismatched_pairs, target_synthetic_count)
    
    print(f"Creating {len(selected_pairs)} synthetic negative examples")
    
    # Create synthetic rows
    synthetic_rows = []
    
    for receptor, epitope in selected_pairs:
        # Find a representative row for this receptor to copy other attributes
        receptor_rows = df[df['Receptor'] == receptor]
        epitope_rows = df[df['Epitope'] == epitope]
        
        if not receptor_rows.empty and not epitope_rows.empty:
            # Use the first matching receptor row as template
            template_row = receptor_rows.iloc[0].copy()
            
            # Update the epitope information from a matching epitope row
            epitope_template = epitope_rows.iloc[0]
            
            # Update the relevant columns
            template_row['Epitope'] = epitope
            template_row['Sequence'] = epitope_template['Sequence']
            template_row['Known Outcome'] = 'Non-Immunogenic'  # Label as non-immunogenic
            
            # Update sequence-related features (copy from epitope template)
            sequence_cols = [col for col in df.columns if 'Sequence_' in col]
            for col in sequence_cols:
                if col in epitope_template:
                    template_row[col] = epitope_template[col]
            
            synthetic_rows.append(template_row)
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_rows)
    print(f"Successfully created {len(synthetic_df)} synthetic negative examples")
    
    return synthetic_df

def combine_and_save_data(original_df, synthetic_df, output_filepath):
    """Combine original and synthetic data and save to file."""
    print("Combining original and synthetic data...")
    
    # Combine the datasets
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Known Outcome distribution:")
    print(combined_df['Known Outcome'].value_counts())
    
    # Save to file
    print(f"Saving combined dataset to: {output_filepath}")
    combined_df.to_csv(output_filepath, index=False)
    print("Data saved successfully!")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic negative training data")
    parser.add_argument("--input", "-i", 
                       default="05_datasets/train_data_with_all_train_immuno_stratify.csv",
                       help="Input training data CSV file")
    parser.add_argument("--output", "-o", 
                       default="05_datasets/train_data_with_synthetic_negatives.csv",
                       help="Output CSV file with synthetic negatives")
    parser.add_argument("--ratio", "-r", type=float, default=1.0,
                       help="Ratio of synthetic negatives to original data (default: 1.0)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("SYNTHETIC NEGATIVE DATA GENERATION")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Synthetic ratio: {args.ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Load original data
    original_df = load_training_data(args.input)
    
    # Generate synthetic negatives
    synthetic_df = generate_synthetic_negatives(original_df, args.ratio)
    
    # Combine and save
    combined_df = combine_and_save_data(original_df, synthetic_df, args.output)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original data: {len(original_df)} rows")
    print(f"Synthetic negatives: {len(synthetic_df)} rows")
    print(f"Combined data: {len(combined_df)} rows")
    print(f"Output saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main() 