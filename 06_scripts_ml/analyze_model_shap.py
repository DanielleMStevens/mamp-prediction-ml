import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from models.esm_with_receptor_model import ESMWithReceptorModel
from dataset import SeqAffDataset
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser('SHAP Analysis Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default='shap_analysis', help='Output directory for analysis')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def setup_analysis(args):
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
    dataset = SeqAffDataset(df=pd.read_csv(args.data_path))
    
    return model, dataset, output_dir

def analyze_global_importance(model, dataset, num_samples, output_dir):
    """Analyze global feature importance"""
    print("Analyzing global feature importance...")
    
    # Get global feature importance
    importance_data = model.global_feature_importance(dataset, num_samples)
    
    # Plot global importance
    plt.figure(figsize=(15, 10))
    model.plot_shap_summary(
        importance_data['shap_values'],
        output_path=output_dir / 'global_importance.png'
    )
    
    # Save feature importance scores
    importance_df = pd.DataFrame({
        'feature': range(len(importance_data['feature_importance'])),
        'importance': importance_data['feature_importance']
    })
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return importance_data

def analyze_individual_predictions(model, dataset, num_samples, output_dir):
    """Analyze individual predictions"""
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
    """Analyze feature interactions"""
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
    """Analyze model behavior across different subsets"""
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