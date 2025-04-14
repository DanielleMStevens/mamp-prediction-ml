"""
This script performs SHAP (SHapley Additive exPlanations) analysis on a trained ESM-based model
for peptide-receptor binding prediction. It provides various analyses including global feature importance,
individual prediction explanations, feature interactions, and subset behavior analysis.

The script uses the SHAP framework to explain model predictions and generate visualizations that help
understand how the model makes its decisions. It supports both peptide and receptor sequence inputs
and can analyze their contributions to the final predictions.

Usage:
    python 07_analyze_model_shap.py --model_path path/to/model --data_path path/to/data.csv 
                                   [--output_dir output/dir] [--num_samples 100] [--device cuda]

Requirements:
    - torch
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - shap
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path, PosixPath
from functools import partial
import shap
import sys
import os
from torch.serialization import add_safe_globals

# Add argparse.Namespace to the safe globals list
add_safe_globals([argparse.Namespace, PosixPath])

# Get the absolute path to the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # 06_scripts_ml
sys.path.append(os.path.join(current_dir, "models"))
from esm_with_receptor_model import ESMWithReceptorModel

# Define a simple dataset class if importing is problematic
class SimpleSeqAffDataset:
    """
    A simple dataset class for handling peptide-receptor sequence pairs.
    
    This class provides a basic interface for accessing peptide sequences,
    receptor sequences, and their corresponding binding outcomes.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sequence and outcome data
        
    Attributes:
        df (pandas.DataFrame): The stored DataFrame
    """
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Replace these with your actual column names from the CSV
        return {
            'peptide_x': row['Sequence'],
            'receptor_x': row['Receptor Sequence'],
            'y': row['Known Outcome'] if 'Known Outcome' in row else 0
        }


def parse_args():
    """
    Parse command line arguments for the SHAP analysis script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - model_path: Path to the saved model checkpoint
            - data_path: Path to the dataset CSV file
            - output_dir: Directory to save analysis outputs
            - num_samples: Number of samples to analyze
            - device: Device to run the model on ('cuda' or 'cpu')
    """
    parser = argparse.ArgumentParser('SHAP Analysis Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default='07_model_analysis', help='Output directory for analysis')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def setup_analysis(args):
    """
    Set up the analysis environment by loading the model and dataset.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        tuple: Contains:
            - model: Loaded ESMWithReceptorModel instance
            - dataset: SimpleSeqAffDataset instance
            - output_dir: Path object for the output directory
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = ESMWithReceptorModel()
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    
    # Load dataset
    dataset = SimpleSeqAffDataset(df=pd.read_csv(args.data_path))
    
    return model, dataset, output_dir

def analyze_global_importance(model, dataset, num_samples, output_dir):
    """
    Analyze global feature importance using SHAP values.
    
    This function calculates and visualizes the global importance of features across
    multiple samples from the dataset. It generates both plots and numerical scores
    for feature importance.
    
    Args:
        model: The trained model instance
        dataset: Dataset containing peptide-receptor pairs
        num_samples (int): Number of samples to analyze
        output_dir (Path): Directory to save analysis outputs
        
    Returns:
        dict: Contains:
            - shap_values: List of SHAP values for analyzed samples
            - feature_importance: Array of global feature importance scores
    """
    print("Analyzing global feature importance...")
    
    # Get global feature importance
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    sampled_data = [dataset[i] for i in indices]
    
    # Aggregate SHAP values
    all_shap_values = []
    for data in sampled_data:
        try:
            # Ensure we're passing strings to the explain_prediction method
            peptide_seq = str(data['peptide_x'])
            receptor_seq = str(data['receptor_x'])
            
            print(f"Analyzing sequence: {peptide_seq[:20]}... with receptor: {receptor_seq[:20]}...")
            
            shap_values = model.explain_prediction(
                peptide_seq,
                receptor_seq
            )
            all_shap_values.append(shap_values)
        except Exception as e:
            print(f"Error analyzing sample: {e}")
            continue
    
    if not all_shap_values:
        print("No valid SHAP values were generated. Check your data and model.")
        return {'shap_values': [], 'feature_importance': np.array([])}
    
    # Calculate feature importance
    feature_importance = np.abs(np.mean([sv['peptide_shap'].values for sv in all_shap_values], axis=0))
    
    # Plot global importance
    plt.figure(figsize=(15, 10))
    model.plot_shap_summary(
        all_shap_values,
        output_path=output_dir / 'global_importance.png'
    )
    
    # Save feature importance scores
    importance_df = pd.DataFrame({
        'feature': range(len(feature_importance)),
        'importance': feature_importance
    })
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return {
        'shap_values': all_shap_values,
        'feature_importance': feature_importance
    }

def analyze_individual_predictions(model, dataset, num_samples, output_dir):
    """
    Generate and visualize SHAP explanations for individual predictions.
    
    Creates detailed visualizations showing how each feature contributes to
    specific predictions for both peptide and receptor sequences.
    
    Args:
        model: The trained model instance
        dataset: Dataset containing peptide-receptor pairs
        num_samples (int): Number of individual predictions to analyze
        output_dir (Path): Directory to save analysis outputs
        
    Returns:
        list: List of dictionaries containing analysis results for each sample
    """
    print("Analyzing individual predictions...")
    
    # Sample some examples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    results = []
    
    for idx in indices:
        data = dataset[idx]
        explanation = model.explain_prediction(
            data['peptide_x'],
            data['receptor_x']
        )
        
        # Create individual explanation plot
        plt.figure(figsize=(15, 10))
        shap.plots.text(explanation['peptide_shap'], show=False)
        plt.savefig(output_dir / f'individual_explanation_{idx}_peptide.png')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        shap.plots.text(explanation['receptor_shap'], show=False)
        plt.savefig(output_dir / f'individual_explanation_{idx}_receptor.png')
        plt.close()
        
        results.append({
            'index': idx,
            'explanation': explanation
        })
    
    return results

def analyze_feature_interactions(model, dataset, num_samples, output_dir):
    """
    Analyze and visualize interactions between different features.
    
    Creates a heatmap showing how different features interact with each other
    in contributing to the model's predictions.
    
    Args:
        model: The trained model instance
        dataset: Dataset containing peptide-receptor pairs
        num_samples (int): Number of samples to use for interaction analysis
        output_dir (Path): Directory to save analysis outputs
        
    Returns:
        dict: Contains interaction analysis results
    """
    print("Analyzing feature interactions...")
    
    # Get interaction values
    interaction_data = model.analyze_feature_interactions(dataset, num_samples)
    
    # Plot interaction matrix
    interaction_values = np.mean([v['peptide_shap'].values for v in interaction_data['interaction_values']], axis=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_values, cmap='RdBu', center=0)
    plt.title('Feature Interaction Strength')
    plt.savefig(output_dir / 'feature_interactions.png')
    plt.close()
    
    return interaction_data

def analyze_subset_behavior(model, dataset, output_dir):
    """
    Analyze model behavior across different confidence-based subsets.
    
    Segments predictions into high, medium, and low confidence groups and
    analyzes how the model behaves differently across these subsets.
    
    Args:
        model: The trained model instance
        dataset: Dataset containing peptide-receptor pairs
        output_dir (Path): Directory to save analysis outputs
        
    Returns:
        dict: Analysis results for each confidence-based subset
    """
    print("Analyzing subset behavior...")
    
    # Define different subsets to analyze
    subsets = {
        'high_confidence': lambda x: max(model.get_pr(model(model.collate_fn([x])['x'])).detach().cpu().numpy()[0]) > 0.9,
        'medium_confidence': lambda x: 0.5 < max(model.get_pr(model(model.collate_fn([x])['x'])).detach().cpu().numpy()[0]) <= 0.9,
        'low_confidence': lambda x: max(model.get_pr(model(model.collate_fn([x])['x'])).detach().cpu().numpy()[0]) <= 0.5
    }
    
    results = {}
    for subset_name, condition in subsets.items():
        print(f"Analyzing {subset_name} subset...")
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(exist_ok=True)
        
        analysis = model.analyze_subset_behavior(dataset, condition)
        if 'error' not in analysis:
            # Plot subset-specific SHAP values
            plt.figure(figsize=(12, 8))
            model.plot_shap_summary(
                analysis['shap_values'],
                output_path=subset_dir / 'shap_summary.png'
            )
            
            # Save subset statistics
            pd.DataFrame({
                'prediction_mean': np.mean(analysis['predictions'], axis=0),
                'prediction_std': np.std(analysis['predictions'], axis=0)
            }).to_csv(subset_dir / 'statistics.csv')
            
        results[subset_name] = analysis
    
    return results

def main():
    """
    Main execution function that runs all SHAP analyses.
    
    This function orchestrates the complete analysis pipeline:
    1. Parses command line arguments
    2. Sets up the analysis environment
    3. Runs global importance analysis
    4. Analyzes individual predictions
    5. Analyzes feature interactions
    6. Performs subset behavior analysis
    """
    args = parse_args()
    model, dataset, output_dir = setup_analysis(args)
    
    # Run all analyses
    global_importance = analyze_global_importance(model, dataset, args.num_samples, output_dir)
    individual_predictions = analyze_individual_predictions(model, dataset, args.num_samples, output_dir)
    feature_interactions = analyze_feature_interactions(model, dataset, args.num_samples, output_dir)
    subset_analysis = analyze_subset_behavior(model, dataset, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 