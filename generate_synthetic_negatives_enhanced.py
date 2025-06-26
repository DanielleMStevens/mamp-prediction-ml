#!/usr/bin/env python3
"""
Enhanced script to generate synthetic negative training data for MAMP prediction ML.

This script creates synthetic negative examples by deliberately mismatching receptors 
with epitopes that don't naturally pair together. It can generate more synthetic data
by allowing duplication of mismatched pairs when needed.
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

def generate_synthetic_negatives_enhanced(df, num_synthetic_per_original=1, allow_duplicates=True):
    """
    Generate synthetic negative examples by mismatching receptors and epitopes.
    
    Args:
        df: Original training dataframe
        num_synthetic_per_original: Number of synthetic examples to create per original example
        allow_duplicates: Whether to allow duplicate mismatched pairs when needed
    
    Returns:
        DataFrame with synthetic negative examples
    """
    print(f"Generating synthetic negatives with ratio {num_synthetic_per_original}:1")
    print(f"Allow duplicates: {allow_duplicates}")
    
    # Get all unique receptors and epitopes
    unique_receptors = df['Receptor'].unique()
    unique_epitopes = df['Epitope'].unique()
    
    print(f"Found {len(unique_receptors)} unique receptors: {list(unique_receptors)}")
    print(f"Found {len(unique_epitopes)} unique epitopes: {list(unique_epitopes)}")
    
    # Get existing receptor-epitope pairs to avoid duplicates
    existing_pairs = set(zip(df['Receptor'], df['Epitope']))
    print(f"Existing receptor-epitope pairs: {len(existing_pairs)}")
    
    # Generate all possible receptor-epitope combinations
    all_possible_pairs = set(product(unique_receptors, unique_epitopes))
    print(f"All possible combinations: {len(all_possible_pairs)}")
    
    # Find mismatched pairs (combinations that don't exist in original data)
    mismatched_pairs = list(all_possible_pairs - existing_pairs)
    print(f"Available mismatched pairs: {len(mismatched_pairs)}")
    
    # Calculate how many synthetic examples to create
    target_synthetic_count = int(len(df) * num_synthetic_per_original)
    print(f"Target synthetic count: {target_synthetic_count}")
    
    # Select pairs for synthetic generation
    if len(mismatched_pairs) >= target_synthetic_count:
        # We have enough unique mismatched pairs
        selected_pairs = random.sample(mismatched_pairs, target_synthetic_count)
        print(f"Using {len(selected_pairs)} unique mismatched pairs")
    else:
        if allow_duplicates:
            # Use all unique mismatched pairs and then sample with replacement
            selected_pairs = mismatched_pairs.copy()
            remaining_needed = target_synthetic_count - len(mismatched_pairs)
            
            # Sample additional pairs with replacement
            additional_pairs = random.choices(mismatched_pairs, k=remaining_needed)
            selected_pairs.extend(additional_pairs)
            
            print(f"Using all {len(mismatched_pairs)} unique pairs + {remaining_needed} duplicates")
            print(f"Total selected pairs: {len(selected_pairs)}")
        else:
            # Only use available unique pairs
            selected_pairs = mismatched_pairs
            print(f"Warning: Only {len(mismatched_pairs)} unique pairs available, using all of them")
    
    # Create synthetic rows
    synthetic_rows = []
    pair_counts = {}  # Track how many times each pair is used
    
    for receptor, epitope in selected_pairs:
        pair_key = (receptor, epitope)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        
        # Find representative rows for this receptor and epitope
        receptor_rows = df[df['Receptor'] == receptor]
        epitope_rows = df[df['Epitope'] == epitope]
        
        if not receptor_rows.empty and not epitope_rows.empty:
            # Randomly select template rows to add variety
            receptor_template = receptor_rows.sample(n=1).iloc[0]
            epitope_template = epitope_rows.sample(n=1).iloc[0]
            
            # Create new synthetic row
            synthetic_row = receptor_template.copy()
            
            # Update epitope-related information
            synthetic_row['Epitope'] = epitope
            synthetic_row['Sequence'] = epitope_template['Sequence']  # Use epitope sequence
            synthetic_row['Known Outcome'] = 'Non-Immunogenic'  # Label as synthetic negative
            
            # Update sequence-related features from epitope template
            sequence_cols = [col for col in df.columns if 'Sequence_' in col]
            for col in sequence_cols:
                if col in epitope_template:
                    synthetic_row[col] = epitope_template[col]
            
            # Add some random variation to continuous features to increase diversity
            if allow_duplicates and pair_counts[pair_key] > 1:
                # Add small random noise to numerical columns for duplicates
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if col in synthetic_row and pd.notna(synthetic_row[col]):
                        # Add small random noise (±5% of the value)
                        noise_factor = random.uniform(0.95, 1.05)
                        synthetic_row[col] = synthetic_row[col] * noise_factor
            
            synthetic_rows.append(synthetic_row)
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_rows)
    
    # Print statistics about pair usage
    print(f"\nPair usage statistics:")
    unique_pairs_used = len(pair_counts)
    max_usage = max(pair_counts.values()) if pair_counts else 0
    print(f"  Unique pairs used: {unique_pairs_used}")
    print(f"  Maximum times a pair was used: {max_usage}")
    
    print(f"Successfully created {len(synthetic_df)} synthetic negative examples")
    
    return synthetic_df

def analyze_data_distribution(df, title="Data Distribution"):
    """Analyze and print data distribution."""
    print(f"\n{title}:")
    print(f"  Total rows: {len(df)}")
    print(f"  Known Outcome distribution:")
    outcome_counts = df['Known Outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    {outcome}: {count} ({percentage:.1f}%)")
    
    print(f"  Receptor distribution:")
    receptor_counts = df['Receptor'].value_counts()
    for receptor, count in receptor_counts.items():
        print(f"    {receptor}: {count}")
    
    print(f"  Epitope distribution:")
    epitope_counts = df['Epitope'].value_counts()
    for epitope, count in epitope_counts.items():
        print(f"    {epitope}: {count}")

def combine_and_save_data(original_df, synthetic_df, output_filepath):
    """Combine original and synthetic data and save to file."""
    print("\nCombining original and synthetic data...")
    
    # Combine the datasets
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    # Analyze distributions
    analyze_data_distribution(original_df, "Original Data Distribution")
    analyze_data_distribution(synthetic_df, "Synthetic Data Distribution")
    analyze_data_distribution(combined_df, "Combined Data Distribution")
    
    # Save to file
    print(f"\nSaving combined dataset to: {output_filepath}")
    combined_df.to_csv(output_filepath, index=False)
    print("Data saved successfully!")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic negative training data (Enhanced)")
    parser.add_argument("--input", "-i", 
                       default="05_datasets/train_data_with_all_train_immuno_stratify.csv",
                       help="Input training data CSV file")
    parser.add_argument("--output", "-o", 
                       default="05_datasets/train_data_with_synthetic_negatives_enhanced.csv",
                       help="Output CSV file with synthetic negatives")
    parser.add_argument("--ratio", "-r", type=float, default=1.0,
                       help="Ratio of synthetic negatives to original data (default: 1.0)")
    parser.add_argument("--no-duplicates", action="store_true",
                       help="Don't allow duplicate mismatched pairs")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("ENHANCED SYNTHETIC NEGATIVE DATA GENERATION")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Synthetic ratio: {args.ratio}")
    print(f"Allow duplicates: {not args.no_duplicates}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Load original data
    original_df = load_training_data(args.input)
    
    # Generate synthetic negatives
    synthetic_df = generate_synthetic_negatives_enhanced(
        original_df, 
        args.ratio, 
        allow_duplicates=not args.no_duplicates
    )
    
    # Combine and save
    combined_df = combine_and_save_data(original_df, synthetic_df, args.output)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original data: {len(original_df)} rows")
    print(f"Synthetic negatives: {len(synthetic_df)} rows")
    print(f"Combined data: {len(combined_df)} rows")
    print(f"Increase in dataset size: {((len(combined_df) - len(original_df)) / len(original_df) * 100):.1f}%")
    print(f"Output saved to: {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main() 