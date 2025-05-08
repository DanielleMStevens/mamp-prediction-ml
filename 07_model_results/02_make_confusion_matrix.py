#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: Generate confusion matrix and report misclassification details
# Inputs: Predictions, receptor/ligand data
# Outputs: Confusion matrices and misclassification report
#-----------------------------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from pathlib import Path
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser('Confusion Matrix Generation Script', add_help=False)
    parser.add_argument('--predictions_path', default='test_preds.pth', type=str, 
                        help='Path to predictions file')
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--data_info_path', type=str, required=True,
                        help='Path to Test CSV/TSV file containing receptor and ligand information')
    args = parser.parse_args()
    return args

def analyze_classifications(gt, preds, data_info):
    """
    Identify and categorize all samples into false positives (custom def),
    false negatives (custom def), or correct classifications based on the
    3-class model output (0=Immuno, 1=NonImmuno, 2=WeaklyImmuno).
    Reports the original 3-class known and predicted outcome labels.
    """
    false_positives = []
    false_negatives = []
    correct_classifications = []

    # Mapping from numerical label to string label
    label_map = {
        0: "Immunogenic",
        1: "Non-Immunogenic",
        2: "Weakly Immunogenic"
    }

    for idx, (true_label, pred_label) in enumerate(zip(gt, preds)):

        # Get the string representation for known and predicted outcomes
        known_outcome_str = label_map.get(true_label, f"Unknown Label ({true_label})")
        predicted_outcome_str = label_map.get(pred_label, f"Unknown Label ({pred_label})")

        # Basic row info shared by all
        row_info_base = {
            'plant_species': data_info.iloc[idx]['Plant species'],
            'receptor': data_info.iloc[idx]['Receptor'],
            'locus_id': data_info.iloc[idx]['Locus ID/Genbank'],
            'epitope': data_info.iloc[idx]['Epitope'],
            'sequence': data_info.iloc[idx]['Sequence'],
            'receptor_sequence': data_info.iloc[idx]['Receptor Sequence'],
            'known_outcome': known_outcome_str,
            'predicted_outcome': predicted_outcome_str,
        }

        # Check for Correct Prediction First
        if true_label == pred_label:
            correct_classifications.append(row_info_base)
            continue # Move to next sample

        # --- Handle Misclassifications (Your Custom Definitions) ---

        # Determine if it's a False Negative based on user definition:
        # FN: (0->1), (0->2), (1->2)
        is_fn = (true_label == 0 and (pred_label == 1 or pred_label == 2)) or \
                (true_label == 1 and pred_label == 2)

        # Determine if it's a False Positive based on user definition:
        # FP: (1->0), (2->1), (2->0)
        is_fp = (true_label == 1 and pred_label == 0) or \
                (true_label == 2 and (pred_label == 0 or pred_label == 1))

        # Assign misclassification type string
        misclassification_type = "False Negative" if is_fn else "False Positive"
        row_info = row_info_base.copy() # Start with base info
        row_info['misclassification_type'] = misclassification_type

        if is_fn:
            false_negatives.append(row_info)
        elif is_fp:
            false_positives.append(row_info)
        # else: # Should not happen if definitions cover all misclassifications

    print(f"Found {len(correct_classifications)} correct classifications.")
    print(f"Found {len(false_positives)} false positives and {len(false_negatives)} false negatives for the report, based on custom definitions.")
    return correct_classifications, false_positives, false_negatives

def write_misclassification_report(false_positives, false_negatives, output_path):
    """
    Write detailed misclassification report to a tab-delimited file
    """
    with open(output_path, 'w') as f:
        # Write header (add Plant_species)
        headers = ['Misclassification_type', 'Known_outcome', 'Predicted_outcome', 'Plant_species', 'Ligand', 'Ligand_sequence', 'Receptor', 'Receptor_sequence']
        f.write('\t'.join(headers) + '\n')

        # Write all misclassifications in one list
        all_misclassifications = false_positives + false_negatives
        for entry in all_misclassifications:
            # Ensure all necessary keys exist before accessing
            row = [
                entry.get('misclassification_type', 'N/A'),
                entry.get('known_outcome', 'N/A'),
                entry.get('predicted_outcome', 'N/A'),
                entry.get('plant_species', 'N/A'), # Add plant species
                entry.get('epitope', 'N/A'),
                entry.get('sequence', 'N/A'),
                entry.get('receptor', 'N/A'),
                entry.get('receptor_sequence', 'N/A')
            ]
            f.write('\t'.join(map(str, row)) + '\n') # Ensure strings for join

def write_correct_classification_report(correct_samples, output_path):
    """
    Write detailed correct classification report to a tab-delimited file
    """
    with open(output_path, 'w') as f:
        # Write header (add Plant_species)
        headers = ['Known_outcome', 'Predicted_outcome', 'Plant_species', 'Ligand', 'Ligand_sequence', 'Receptor', 'Receptor_sequence']
        f.write('\t'.join(headers) + '\n')

        # Write all correct classifications
        for entry in correct_samples:
             # Ensure all necessary keys exist before accessing
            row = [
                entry.get('known_outcome', 'N/A'),
                entry.get('predicted_outcome', 'N/A'),
                entry.get('plant_species', 'N/A'), # Add plant species
                entry.get('epitope', 'N/A'),
                entry.get('sequence', 'N/A'),
                entry.get('receptor', 'N/A'),
                entry.get('receptor_sequence', 'N/A')
            ]
            f.write('\t'.join(map(str, row)) + '\n') # Ensure strings for join

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    predictions = torch.load(args.predictions_path)
    
    # Load data information
    data_info = pd.read_csv(args.data_info_path)
    
    # Get ground truth and predictions
    gt = predictions['gt'].detach().cpu().numpy()
    preds = predictions['pr'].detach().cpu().numpy()
    
    # For probability outputs, convert to class predictions
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=1)
    
    # Create confusion matrices
    cm = confusion_matrix(gt, preds)
    cm_percentage = confusion_matrix(gt, preds, normalize='true') * 100
    
    # Create raw counts confusion matrix plot
    plt.figure(figsize=(1.8, 1.6), dpi=450)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                annot_kws={"size": 8, "family": "Arial"},
                vmin=0, vmax=np.max(cm), # Set scale based on max count
                cbar_kws={'ticks': np.linspace(0, np.max(cm), 5)}) # Add 5 ticks to colorbar
    plt.title('Confusion Matrix (Counts)', fontsize=7, fontname='Arial')
    plt.ylabel('True Label', fontsize=7, fontname='Arial')
    plt.xlabel('Predicted Label', fontsize=7, fontname='Arial')
    plt.tick_params(axis='both', which='major', labelsize=8)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    legend = plt.gca().get_legend()
    if legend is not None:
        plt.setp(legend.get_texts(), fontname='Arial', fontsize=5)
    # Set colorbar label font size
    plt.gca().collections[0].colorbar.ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.5)
    
    # Save counts plot
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_counts.pdf'))
    plt.close()
    
    # Create percentage confusion matrix plot 
    plt.figure(figsize=(1.8, 1.6), dpi=450)
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Purples',
                annot_kws={"size": 8, "family": "Arial"},
                vmin=0, vmax=100, # Set scale from 0-100%
                cbar_kws={'ticks': np.arange(0, 101, 20)}) # Add ticks every 20%
    plt.title('Confusion Matrix (%)', fontsize=7, fontname='Arial')
    plt.ylabel('True Label', fontsize=7, fontname='Arial')
    plt.xlabel('Predicted Label', fontsize=7, fontname='Arial')
    plt.tick_params(axis='both', which='major', labelsize=8)
    for tick in plt.gca().get_xticklabels():
        tick.set_fontname("Arial")
    for tick in plt.gca().get_yticklabels():
        tick.set_fontname("Arial")
    legend = plt.gca().get_legend()
    if legend is not None:
        plt.setp(legend.get_texts(), fontname='Arial', fontsize=5)
    # Set colorbar label font size
    plt.gca().collections[0].colorbar.ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.5)
    # Save percentage plot
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_percentages.pdf'))
    plt.close()
    
    # Analyze classifications
    correct_samples, false_positives, false_negatives = analyze_classifications(gt, preds, data_info)

    # Write misclassification report in tab-delimited format
    write_misclassification_report(
        false_positives,
        false_negatives,
        os.path.join(args.output_dir, 'misclassification_report.tsv')
    )

    # Write correct classification report in tab-delimited format
    write_correct_classification_report(
        correct_samples,
        os.path.join(args.output_dir, 'correct_classification_report.tsv')
    )

    print("\nClassification Reports (Correct and Misclassified) Exported.")
    report = classification_report(gt, preds)
    
    # Save classification report to a text file with better formatting
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Performance Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Class Labels:\n")
        f.write("0: Immunogenic\n")
        f.write("1: Non-Immunogenic\n")
        f.write("2: Weakly Immunogenic\n\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(report)
        f.write("\nMetric Definitions:\n")
        f.write("-" * 30 + "\n")
        f.write("precision: True Positives / (True Positives + False Positives)\n")
        f.write("recall: True Positives / (True Positives + False Negatives)\n")
        f.write("f1-score: 2 * (precision * recall) / (precision + recall)\n")
        f.write("support: Number of samples for each class\n")

if __name__ == '__main__':
    args = parse_args()
    main(args) 