"""
This script performs SHAP (SHapley Additive exPlanations) analysis on a trained ESM-based model
for peptide-receptor binding prediction. It provides various analyses including global feature importance,
individual prediction explanations, feature interactions, and subset behavior analysis.

The script uses the SHAP framework to explain model predictions and generate visualizations that help
understand how the model makes its decisions. It supports both peptide and receptor sequence inputs
and can analyze their contributions to the final predictions.

Usage:
    python 07_analyze_model_shap.py --model_path models/checkpoint-50.pth \
                                   --data_path 05_datasets/stratify/test.csv \
                                   --output_dir 07_model_results/shap_analysis \
                                   --num_samples 100 \
                                   --device cpu



python 07_model_results/01_analyze_model_shap.py --model_path 07_model_results/00_mamp_ml/checkpoint-19.pth \
                           --data_path 05_datasets/test_data_with_all_test_immuno_stratify.csv \
                           --output_dir 07_model_results/shap_analysis \
                           --num_samples 100 \
                           --device cpu \
                           --model esm_bfactor_weighted

python 07_model_results/01_analyze_model_shap.py --model_path 07_model_results/02_immuno_stratify_esm2_with_receptor/checkpoint-19.pth \
   --data_path 05_datasets/test_data_with_all_test_immuno_stratify.csv \
   --output_dir 07_model_results/shap_analysis \
   --num_samples 100 \
   --device cpu \
   --model esm_with_receptor

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
# current_dir = os.path.dirname(os.path.abspath(__file__))  # 06_scripts_ml # This was incorrect for the current script location
# sys.path.append(os.path.join(current_dir, "models")) # This was looking for models inside 07_model_results

# Corrected sys.path modification:
script_dir = os.path.dirname(os.path.abspath(__file__))  # This is .../07_model_results
project_root_containing_script_dir = os.path.dirname(script_dir) # This is .../mamp_prediction_ml

# Path to the directory containing the 'models' package (i.e., 06_scripts_ml)
models_base_path = os.path.join(project_root_containing_script_dir, "06_scripts_ml")
sys.path.append(models_base_path)

# esm models - current models
from models.esm_with_receptor_model import ESMWithReceptorModel
from models.esm_all_chemical_features import ESMallChemicalFeatures
from models.esm_positon_weighted import BFactorWeightGenerator
from models.esm_positon_weighted import ESMBfactorWeightedFeatures

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
            - model: Type of model to use (e.g., 'esm_bfactor_weighted')
    """
    parser = argparse.ArgumentParser('SHAP Analysis Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default='07_model_analysis', help='Output directory for analysis')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, default='esm_with_receptor', 
                        choices=['esm_with_receptor', 'esm_bfactor_weighted'],
                        help='Type of model to use for analysis')
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
    if args.model == 'esm_bfactor_weighted':
        model = ESMBfactorWeightedFeatures(args=None, num_classes=3)
    elif args.model == 'esm_with_receptor':
        model = ESMWithReceptorModel()
    else:
        raise ValueError(f"Model type {args.model} not recognized")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    
    # Load dataset
    dataset = SimpleSeqAffDataset(df=pd.read_csv(args.data_path))
    
    return model, dataset, output_dir

def analyze_global_importance(model, dataset, num_samples, output_dir):
    """Analyze global feature importance using SHAP values."""
    print("Analyzing global feature importance...")
    
    # Check if model supports built-in SHAP explanations
    if hasattr(model, 'explain_prediction') and callable(getattr(model, 'explain_prediction')):
        # This is the original path if the model has its own SHAP logic.
        # The user's current model (ESMWithReceptorModel) does not, so this path is skipped.
        print("Model has 'explain_prediction'. Attempting to use it (not fully implemented in this generic script yet).")
        # ... placeholder for original logic that would call model.explain_prediction and aggregate ...
        # For now, let's return an empty result if this path were taken and not implemented.
        return {'shap_values': [], 'feature_importance': np.array([])}
    
    print("Using generic KernelExplainer SHAP approach for global importance...")
    try:
        # 1. Prepare indices for background data and data to explain
        all_dataset_indices = np.arange(len(dataset))
        
        # Use a subset for background data (KernelExplainer can be slow)
        num_background_samples = min(num_samples, len(dataset), 50) # Cap at 50 for speed
        background_indices_1d = np.random.choice(all_dataset_indices, num_background_samples, replace=False)
        # Reshape to 2D for KernelExplainer: (n_samples, n_features=1)
        background_data_for_explainer = background_indices_1d.reshape(-1, 1)

        # 2. Define the prediction function for SHAP's KernelExplainer
        # This function will take a 2D NumPy array `X_input_indices_2d` (shape: num_perturbations, 1 feature)
        # where the feature is the sample index. It must return model predictions.
        def shap_prediction_function(X_input_indices_2d):
            predictions = []
            # Determine the model's device from its parameters
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                # Fallback if model has no parameters, though unlikely for a PyTorch model
                print("Warning: Model has no parameters. Defaulting to CPU for SHAP inputs.")
                model_device = torch.device("cpu")


            for i in range(X_input_indices_2d.shape[0]):
                # Extract the actual dataset index (it's the first and only feature)
                data_idx = int(X_input_indices_2d[i, 0])
                
                # Get the raw sample data
                sample = dataset[data_idx]
                peptide = sample['peptide_x']
                receptor = sample['receptor_x']

                tokenized_input = None
                # Tokenize the sample (ensure model has one of these methods)
                if hasattr(model, 'tokenize_sequences') and callable(getattr(model, 'tokenize_sequences')):
                    tokenized_input = model.tokenize_sequences(peptide, receptor)
                elif hasattr(model, 'tokenizer') and callable(getattr(model, 'tokenizer')):
                    # Assuming a Hugging Face-like tokenizer
                    tokenized_input = model.tokenizer(text=peptide, text_pair=receptor, return_tensors="pt", padding='max_length', truncation=True, max_length=512) # Added max_length
                else:
                    raise AttributeError("Model requires 'tokenize_sequences' or 'tokenizer' method for SHAP analysis.")

                # Prepare input for the model and move to device
                input_on_device = {k: v.to(model_device) for k, v in tokenized_input.items()}
                
                # Get model prediction
                with torch.no_grad():
                    output = model(input_ids=input_on_device['input_ids'], 
                                   attention_mask=input_on_device['attention_mask'])
                    if isinstance(output, tuple): # e.g., (logits, other_outputs)
                        output = output[0]
                    # SHAP expects a NumPy array, typically (num_outputs,) or (1, num_outputs) for one sample
                    predictions.append(output.cpu().numpy().squeeze()) # Squeeze removes batch dim if it was 1

            np_predictions = np.array(predictions)
            # Ensure output is 2D: (num_perturbations, num_model_outputs)
            if np_predictions.ndim == 1:
                np_predictions = np_predictions.reshape(-1, 1)
            return np_predictions

        # 3. Create SHAP KernelExplainer
        # We pass the prediction function and the 2D background data (indices)
        explainer = shap.KernelExplainer(shap_prediction_function, background_data_for_explainer)
        
        # 4. Select samples to explain
        # Let's explain a small number of samples. These also need to be 2D.
        num_explain_samples = min(10, num_background_samples)
        explain_indices_1d = background_indices_1d[:num_explain_samples]
        explain_data_for_shap_values = explain_indices_1d.reshape(-1, 1)
        
        # 5. Calculate SHAP values
        # `nsamples` here is the number of perturbations KernelExplainer runs for each sample explained.
        # Can be computationally intensive.
        shap_values_result = explainer.shap_values(explain_data_for_shap_values, nsamples=100) 
        
        # `shap_values_result` will be an array (if model has 1 output) or list of arrays (if multi-output).
        # Shape: (num_explained_samples, num_shap_features=1) because our "feature" is the index.
        # This SHAP value represents the impact of using that specific sample (index) vs. others.

        # 6. Plotting
        # `summary_plot` expects SHAP values and the feature data used for explanation.
        shap_values_for_plot = shap_values_result
        if isinstance(shap_values_result, list): # If multi-output model
            shap_values_for_plot = shap_values_result[0] # Plot for the first output class

        if shap_values_for_plot is not None and explain_data_for_shap_values is not None:
            plt.figure()
            try:
                shap.summary_plot(shap_values_for_plot, features=explain_data_for_shap_values, 
                                  feature_names=['sample_index_value'], show=False)
                plt.tight_layout()
                plt.savefig(output_dir / 'global_importance_summary.png')
            except Exception as plot_e:
                print(f"Error creating SHAP summary_plot: {plot_e}")
            finally:
                plt.close()
        else:
            print("SHAP values or data for plotting is None, skipping summary_plot.")

        return {'shap_values': shap_values_result, 'explained_data_indices': explain_indices_1d}
    
    except Exception as e:
        print(f"Error in generic SHAP analysis for global importance: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'shap_values': [], 'feature_importance': np.array([])}

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
    
    # Check if model supports SHAP explanations
    if not hasattr(model, 'explain_prediction'):
        print("Warning: The selected model doesn't support SHAP explanations for individual predictions")
        print("Skipping individual prediction analysis")
        return []
    
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
    
    # Check if model supports feature interaction analysis
    if not hasattr(model, 'analyze_feature_interactions'):
        print("Warning: The selected model doesn't support feature interaction analysis")
        print("Skipping feature interaction analysis")
        return {}
    
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
    
    # Check if model supports subset behavior analysis
    if not hasattr(model, 'analyze_subset_behavior'):
        print("Warning: The selected model doesn't support subset behavior analysis")
        print("Skipping subset behavior analysis")
        return {}
    
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