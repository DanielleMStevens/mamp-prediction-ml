"""
This script performs sequence-level feature analysis on a trained ESM-based model
for peptide-receptor binding prediction. It analyzes sequence composition, chemical properties,
and complementarity features to understand model behavior.

Usage:
    python 07_model_results/02_analyze_model_shap.py --model_path models/checkpoint-50.pth \
                                   --data_path 05_datasets/stratify/test.csv \
                                   --output_dir 07_model_results/shap_analysis \
                                   --num_samples 100 \
                                   --device cpu

Requirements:
    - torch
    - numpy
    - pandas
    - matplotlib
    - seaborn
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path, PosixPath
import sys
import os
import json
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
        # Map outcome labels to indices
        self.category_to_index = {
            "Immunogenic": 0, 
            "Non-Immunogenic": 1, 
            "Weakly Immunogenic": 2
        }
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse chemical features from comma-separated strings to lists of floats
        def parse_features(feature_str):
            if pd.isna(feature_str) or feature_str == '':
                return []
            return [float(x.strip()) for x in str(feature_str).split(',')]
        
        # Map outcome to index
        outcome = row['Known Outcome'] if 'Known Outcome' in row else 'Non-Immunogenic'
        y = self.category_to_index.get(outcome, 1)  # Default to Non-Immunogenic
        
        return {
            'peptide_x': row['Sequence'],
            'receptor_x': row['Receptor Sequence'],
            'y': y,
            'receptor_id': row.get('Receptor', ''),  # Use Receptor column as ID
            'sequence_bulkiness': parse_features(row.get('Sequence_Bulkiness', '')),
            'sequence_charge': parse_features(row.get('Sequence_Charge', '')),
            'sequence_hydrophobicity': parse_features(row.get('Sequence_Hydrophobicity', '')),
            'receptor_bulkiness': parse_features(row.get('Receptor_Bulkiness', '')),
            'receptor_charge': parse_features(row.get('Receptor_Charge', '')),
            'receptor_hydrophobicity': parse_features(row.get('Receptor_Hydrophobicity', ''))
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


def analyze_sequence_level_features(all_results, class_names, output_dir, dataset):
    """
    Analyze higher-level sequence and chemical features beyond individual amino acids.
    
    Args:
        all_results (list): Results from character-level analysis
        class_names (list): Names of prediction classes
        output_dir (Path): Directory to save analysis outputs
        dataset: Dataset containing the samples
    """
    print("Analyzing sequence-level and chemical features...")
    
    # Calculate higher-level features for each sample
    sequence_features = []
    
    for result in all_results:
        sample_idx = result['sample_index']
        sample = dataset[sample_idx]
        
        peptide_seq = sample['peptide_x']
        receptor_seq = sample['receptor_x']
        
        # Extract baseline prediction for this sample
        baseline_pred = np.array(result['baseline_prediction'])
        
                 # Get actual ground truth label
        actual_label = sample.get('y', 1)  # Default to Non-Immunogenic if not found
        actual_class_name = class_names[actual_label] if actual_label < len(class_names) else 'Unknown'
        
        # Calculate sequence-level features
        features = {
            'sample_index': sample_idx,
            'baseline_prediction': baseline_pred,
            'predicted_class': class_names[baseline_pred.argmax()],
            'actual_class': actual_class_name,
            'confidence': baseline_pred.max(),
            
            # Length features
            'peptide_length': len(peptide_seq),
            'receptor_length': len(receptor_seq),
            'length_ratio': len(peptide_seq) / len(receptor_seq) if len(receptor_seq) > 0 else 0,
            
            # Amino acid composition features
            'peptide_hydrophobic_pct': calculate_hydrophobic_percentage(peptide_seq),
            'receptor_hydrophobic_pct': calculate_hydrophobic_percentage(receptor_seq),
            'peptide_charged_pct': calculate_charged_percentage(peptide_seq),
            'receptor_charged_pct': calculate_charged_percentage(receptor_seq),
            'peptide_polar_pct': calculate_polar_percentage(peptide_seq),
            'receptor_polar_pct': calculate_polar_percentage(receptor_seq),
            
            # Chemical property aggregates (from existing features)
            'peptide_avg_hydrophobicity': np.mean(sample['sequence_hydrophobicity']) if sample['sequence_hydrophobicity'] else 0,
            'receptor_avg_hydrophobicity': np.mean(sample['receptor_hydrophobicity']) if sample['receptor_hydrophobicity'] else 0,
            'peptide_avg_charge': np.mean(sample['sequence_charge']) if sample['sequence_charge'] else 0,
            'receptor_avg_charge': np.mean(sample['receptor_charge']) if sample['receptor_charge'] else 0,
            'peptide_avg_bulkiness': np.mean(sample['sequence_bulkiness']) if sample['sequence_bulkiness'] else 0,
            'receptor_avg_bulkiness': np.mean(sample['receptor_bulkiness']) if sample['receptor_bulkiness'] else 0,
            
            # Complementarity features
            'hydrophobicity_match': calculate_chemical_complementarity(
                sample['sequence_hydrophobicity'], sample['receptor_hydrophobicity']),
            'charge_complementarity': calculate_charge_complementarity(
                sample['sequence_charge'], sample['receptor_charge']),
            'bulkiness_match': calculate_chemical_complementarity(
                sample['sequence_bulkiness'], sample['receptor_bulkiness']),
            
            # Sequence complexity
            'peptide_entropy': calculate_sequence_entropy(peptide_seq),
            'receptor_entropy': calculate_sequence_entropy(receptor_seq),
        }
        
        sequence_features.append(features)
    
    # Create visualizations for sequence-level features
    create_sequence_feature_plots(sequence_features, class_names, output_dir)
    
    # Analyze feature correlations with predictions
    analyze_feature_correlations(sequence_features, class_names, output_dir)
    
    # Save sequence-level feature data
    with open(output_dir / 'sequence_level_features.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_features = []
        for feat in sequence_features:
            json_feat = feat.copy()
            json_feat['baseline_prediction'] = json_feat['baseline_prediction'].tolist()
            json_features.append(json_feat)
        
        json.dump({
            'sequence_features': json_features,
            'feature_descriptions': {
                'peptide_length': 'Number of amino acids in peptide',
                'receptor_length': 'Number of amino acids in receptor',
                'length_ratio': 'Peptide length / Receptor length',
                'hydrophobic_pct': 'Percentage of hydrophobic amino acids',
                'charged_pct': 'Percentage of charged amino acids (+ and -)',
                'polar_pct': 'Percentage of polar amino acids',
                'avg_hydrophobicity': 'Average hydrophobicity score',
                'avg_charge': 'Average charge score',
                'avg_bulkiness': 'Average bulkiness score',
                'hydrophobicity_match': 'Correlation between peptide and receptor hydrophobicity',
                'charge_complementarity': 'Negative correlation for charge complementarity',
                'bulkiness_match': 'Correlation between peptide and receptor bulkiness',
                'entropy': 'Sequence diversity/complexity measure'
            }
        }, f, indent=2)
    
    print("Sequence-level feature analysis completed!")
    return sequence_features

def calculate_hydrophobic_percentage(sequence):
    """Calculate percentage of hydrophobic amino acids."""
    hydrophobic = set('AILVMFWYC')
    if not sequence:
        return 0
    return sum(1 for aa in sequence.upper() if aa in hydrophobic) / len(sequence) * 100

def calculate_charged_percentage(sequence):
    """Calculate percentage of charged amino acids."""
    charged = set('DEKR')
    if not sequence:
        return 0
    return sum(1 for aa in sequence.upper() if aa in charged) / len(sequence) * 100

def calculate_polar_percentage(sequence):
    """Calculate percentage of polar amino acids."""
    polar = set('STNQH')
    if not sequence:
        return 0
    return sum(1 for aa in sequence.upper() if aa in polar) / len(sequence) * 100

def calculate_chemical_complementarity(seq1_values, seq2_values):
    """Calculate correlation between two sequences' chemical properties."""
    if not seq1_values or not seq2_values or len(seq1_values) != len(seq2_values):
        return 0
    
    try:
        correlation = np.corrcoef(seq1_values, seq2_values)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    except:
        return 0

def calculate_charge_complementarity(seq1_charge, seq2_charge):
    """Calculate charge complementarity (negative correlation for opposite charges)."""
    if not seq1_charge or not seq2_charge or len(seq1_charge) != len(seq2_charge):
        return 0
    
    try:
        # For charge, we want negative correlation (opposite charges attract)
        correlation = np.corrcoef(seq1_charge, seq2_charge)[0, 1]
        return -correlation if not np.isnan(correlation) else 0
    except:
        return 0

def calculate_sequence_entropy(sequence):
    """Calculate Shannon entropy of amino acid sequence."""
    if not sequence:
        return 0
    
    # Count amino acid frequencies
    aa_counts = {}
    for aa in sequence.upper():
        if aa.isalpha():
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    if not aa_counts:
        return 0
    
    # Calculate entropy
    total = sum(aa_counts.values())
    entropy = 0
    for count in aa_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def create_sequence_feature_plots(sequence_features, class_names, output_dir):
    """Create plots for sequence-level features."""
    print("Creating sequence-level feature plots...")
    
    # Convert to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(sequence_features)
    
    # Debug: Print class distribution
    print("\nDEBUG: Sample class distribution:")
    print("Predicted classes:", df['predicted_class'].value_counts().to_dict())
    print("Actual classes:", df['actual_class'].value_counts().to_dict())
    print(f"Total samples: {len(df)}")
    print()
    
    # Define feature groups for plotting
    feature_groups = {
        'Length Features': ['peptide_length', 'receptor_length', 'length_ratio'],
        'Composition Features': ['peptide_hydrophobic_pct', 'receptor_hydrophobic_pct', 
                               'peptide_charged_pct', 'receptor_charged_pct',
                               'peptide_polar_pct', 'receptor_polar_pct'],
        'Chemical Properties': ['peptide_avg_hydrophobicity', 'receptor_avg_hydrophobicity',
                              'peptide_avg_charge', 'receptor_avg_charge',
                              'peptide_avg_bulkiness', 'receptor_avg_bulkiness'],
        'Complementarity Features': ['hydrophobicity_match', 'charge_complementarity', 'bulkiness_match'],
        'Complexity Features': ['peptide_entropy', 'receptor_entropy']
    }
    
    # Create plots for each feature group
    for group_name, features in feature_groups.items():
        plt.figure(figsize=(16, 12))
        
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        for i, feature in enumerate(features):
            plt.subplot(n_rows, n_cols, i + 1)
            
                         # Create violin plot for each class (using ACTUAL labels, not predicted)
            class_data = []
            class_labels = []
            
            for class_name in class_names:
                class_values = df[df['actual_class'] == class_name][feature].values
                if len(class_values) > 0:
                    class_data.append(class_values)
                    class_labels.append(class_name)
                    print(f"  {class_name}: {len(class_values)} samples for {feature}")
                else:
                    print(f"  {class_name}: 0 samples for {feature}")
            
            if class_data:
                positions = range(len(class_data))
                violins = plt.violinplot(class_data, positions=positions, showmeans=True, showmedians=True)
                
                # Color violins by class
                colors = ['lightblue', 'lightcoral', 'lightgreen']
                for pc, color in zip(violins['bodies'], colors[:len(class_data)]):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                plt.xticks(positions, class_labels, rotation=45, ha='right')
                plt.ylabel(feature.replace('_', ' ').title())
                plt.title(f'{feature.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{feature.replace("_", " ").title()}')
        
        plt.suptitle(f'{group_name} by Actual Class (Ground Truth)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        safe_group_name = group_name.replace(' ', '_').lower()
        plt.savefig(output_dir / f'sequence_features_{safe_group_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {group_name} plot")

def analyze_feature_correlations(sequence_features, class_names, output_dir):
    """Analyze feature distributions and correlations using violin plots."""
    print("Analyzing feature correlations with violin plots...")
    
    import pandas as pd
    df = pd.DataFrame(sequence_features)
    
    # Select numerical features for analysis
    numerical_features = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
                         and col not in ['sample_index', 'confidence']]
    
    # Calculate correlations with prediction confidence for ranking
    correlations = []
    for feature in numerical_features:
        corr_with_confidence = df[feature].corr(df['confidence'])
        if not np.isnan(corr_with_confidence):
            correlations.append((feature, corr_with_confidence))
    
    # Sort features by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Select top features for detailed violin plot analysis
    top_features = [feat for feat, corr in correlations[:12]]  # Top 12 most correlated features
    
    if not top_features:
        print("No significant correlations found for violin plot analysis")
        return
    
    # Create comprehensive violin plot showing feature distributions by class
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Prepare data for violin plot - one violin per class
        class_data = []
        class_labels = []
        class_colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        for class_name in class_names:
            class_values = df[df['actual_class'] == class_name][feature].values
            if len(class_values) > 0:
                class_data.append(class_values)
                class_labels.append(class_name)
        
        if class_data and len(class_data) > 1:  # Need multiple classes for comparison
            positions = range(len(class_data))
            violins = ax.violinplot(class_data, positions=positions, 
                                   showmeans=True, showmedians=True, showextrema=True)
            
            # Color violins by class
            for pc, color in zip(violins['bodies'], class_colors[:len(class_data)]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('navy')
            
            # Customize violin plot elements
            violins['cmeans'].set_color('red')
            violins['cmedians'].set_color('orange') 
            violins['cmaxes'].set_color('black')
            violins['cmins'].set_color('black')
            violins['cbars'].set_color('black')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.set_ylabel('Feature Value')
            
            # Get correlation for this feature
            correlation = next((corr for feat, corr in correlations if feat == feature), 0)
            correlation_text = f"r={correlation:.3f}"
            color_text = "green" if correlation > 0 else "red"
            
            ax.set_title(f'{feature.replace("_", " ").title()}\n{correlation_text}', 
                        fontsize=11, color=color_text)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for pos, (class_name, data) in enumerate(zip(class_labels, class_data)):
                mean_val = np.mean(data)
                ax.text(pos, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.2f}', 
                       ha='center', va='top', fontsize=9, fontweight='bold')
        
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11)
    
    # Hide empty subplots
    for i in range(len(top_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Feature Distributions Across Classes (Violin Plots)\n'
                 'Red=Mean, Orange=Median, r=Correlation with Confidence', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations_with_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature correlation violin plots")
    
    # Create a second plot showing correlation rankings
    plt.figure(figsize=(14, 10))
    
    if correlations:
        features, corr_values = zip(*correlations)
        y_pos = np.arange(len(features))
        
        colors = ['forestgreen' if val > 0 else 'crimson' for val in corr_values]
        bars = plt.barh(y_pos, corr_values, color=colors, alpha=0.7)
        
        # Add correlation values as text
        for i, (bar, corr_val) in enumerate(zip(bars, corr_values)):
            x_pos = bar.get_width() + (0.02 if corr_val > 0 else -0.02)
            plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{corr_val:.3f}',
                    ha='left' if corr_val > 0 else 'right', va='center', fontsize=10)
        
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Correlation with Prediction Confidence')
        plt.title('Feature Correlation Rankings\n'
                 'Green=Positive, Red=Negative Correlation', fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3, axis='x')
    else:
        plt.text(0.5, 0.5, 'No correlations to display', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
        plt.title('Feature Correlations with Prediction Confidence')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlation_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature correlation rankings plot")
    
    # Create a summary statistics comparison across classes
    summary_stats = {}
    for class_name in class_names:
        class_df = df[df['actual_class'] == class_name]
        if len(class_df) > 0:
            summary_stats[class_name] = {}
            for feature in top_features:
                if feature in class_df.columns:
                    values = class_df[feature].values
                    summary_stats[class_name][feature] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_samples': len(values)
                    }
    
    # Save enhanced correlation data
    correlation_data = {
        'correlations_with_confidence': correlations,
        'top_features_analyzed': top_features,
        'class_summary_statistics': summary_stats,
        'summary': {
            'strongest_positive_correlation': max(correlations, key=lambda x: x[1]) if correlations else None,
            'strongest_negative_correlation': min(correlations, key=lambda x: x[1]) if correlations else None,
            'most_important_feature': max(correlations, key=lambda x: abs(x[1])) if correlations else None,
            'total_samples_analyzed': len(df),
            'features_with_significant_correlation': len([c for c in correlations if abs(c[1]) > 0.1])
        }
    }
    
    with open(output_dir / 'sequence_feature_correlations.json', 'w') as f:
        json.dump(correlation_data, f, indent=2)

def analyze_global_importance(model, dataset, num_samples, output_dir):
    """Analyze sequence-level features only (skip expensive character analysis)."""
    print("Analyzing global feature importance (sequence-level only)...")
    
    try:
        # Get model device
        model_device = next(model.parameters()).device
        
        # Define class names
        class_names = ['Immunogenic', 'Non-Immunogenic', 'Weakly Immunogenic']
        
        # Select samples for analysis - try to get balanced representation across classes
        all_indices = list(range(len(dataset)))
        
        # Try to get a balanced sample across different classes
        class_samples = {0: [], 1: [], 2: []}  # Immunogenic, Non-Immunogenic, Weakly Immunogenic
        
        for idx in all_indices:
            sample = dataset[idx]
            class_label = sample.get('y', 1)
            if class_label in class_samples:
                class_samples[class_label].append(idx)
        
        # Select samples trying to balance across classes
        selected_indices = []
        samples_per_class = max(1, num_samples // 3)
        
        for class_label, indices in class_samples.items():
            if indices:
                n_to_select = min(samples_per_class, len(indices))
                selected_from_class = np.random.choice(indices, n_to_select, replace=False)
                selected_indices.extend(selected_from_class)
                print(f"Selected {n_to_select} samples from {class_names[class_label]} class")
        
        # If we need more samples, randomly select from remaining
        remaining_needed = num_samples - len(selected_indices)
        if remaining_needed > 0:
            remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
            if remaining_indices:
                additional = np.random.choice(remaining_indices, 
                                            min(remaining_needed, len(remaining_indices)), 
                                            replace=False)
                selected_indices.extend(additional)
        
        sample_indices = selected_indices[:num_samples]
        print(f"Total samples selected: {len(sample_indices)}")
        
        all_results = []
        
        # Simple prediction function
        def simple_prediction_function(sample):
            """Get prediction for a single sample."""
            try:
                batch_input = model.collate_fn([sample])
                device_batch = {}
                for key, value in batch_input.items():
                    if isinstance(value, dict):
                        device_batch[key] = {k: v.to(model_device) if torch.is_tensor(v) else v 
                                           for k, v in value.items()}
                    elif torch.is_tensor(value):
                        device_batch[key] = value.to(model_device)
                    else:
                        device_batch[key] = value
                
                with torch.no_grad():
                    output = model(device_batch)
                    probs = torch.softmax(output, dim=-1)
                    return probs.cpu().numpy()[0]
            except Exception as e:
                print(f"Error in prediction: {e}")
                return np.array([0.33, 0.33, 0.34])
        
        # Process each sample
        for sample_idx, idx in enumerate(sample_indices):
            print(f"Analyzing sample {idx}...")
            sample = dataset[idx]
            
            # Get baseline prediction
            baseline_pred = simple_prediction_function(sample)
            print(f"Processed sample {sample_idx} - baseline prediction: {class_names[baseline_pred.argmax()]} (conf: {baseline_pred.max():.3f})")
            
            # Store minimal results
            all_results.append({
                'sample_index': int(idx),
                'original_sequence': f"{sample['peptide_x']} <eos> {sample['receptor_x']}",
                'baseline_prediction': baseline_pred.tolist(),
                'character_contributions': []  # Empty - we're not doing character analysis
            })
        
        # Analyze higher-level features
        analyze_sequence_level_features(all_results, class_names, output_dir, dataset)
        
        # Save results as JSON
        with open(output_dir / 'character_contributions.json', 'w') as f:
            json.dump({
                'samples': all_results,
                'class_names': class_names,
                'description': 'Sequence-level analysis only (character analysis skipped for performance)'
            }, f, indent=2)
        
        print(f"Sequence-level analysis completed. Files saved to {output_dir}")
        return {'results': all_results, 'class_names': class_names}
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def main():
    """
    Main execution function that runs sequence-level analysis.
    
    This function orchestrates the analysis pipeline:
    1. Parses command line arguments
    2. Sets up the analysis environment
    3. Runs sequence-level feature analysis
    """
    args = parse_args()
    model, dataset, output_dir = setup_analysis(args)
    
    # Run sequence-level analysis
    global_importance = analyze_global_importance(model, dataset, args.num_samples, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 