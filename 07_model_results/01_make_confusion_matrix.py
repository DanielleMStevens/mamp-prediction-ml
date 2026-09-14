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

def style_confusion_heatmap(ax, cbar):
    """Apply clean figure aesthetic: thin black box border, short ticks, no clutter."""
    from matplotlib.patches import Rectangle

    # Hide spines so the single Rectangle frame is not doubled
    for spine in ax.spines.values():
        spine.set_visible(False)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    left, right = min(x0, x1), max(x0, x1)
    bottom, top = min(y0, y1), max(y0, y1)
    frame = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor='black',
        linewidth=0.4,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(frame)

    ax.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=1.5,
        width=0.4,
        colors='black',
        labelsize=8,
        pad=1.5,
        top=False,
        right=False,
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname('Arial')
        tick.set_fontsize(8)

    ax.set_facecolor('white')
    ax.grid(False)

    if cbar is not None:
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.4)
        cbar.ax.tick_params(
            labelsize=7,
            length=1.2,
            width=0.4,
            colors='black',
            direction='out',
        )
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Arial')


def plot_confusion_matrix(cm, output_path, title, fmt, vmin, vmax, cbar_ticks):
    """Save a clean confusion-matrix heatmap PDF."""
    fig, ax = plt.subplots(figsize=(1.7, 1.5), dpi=300)
    fig.patch.set_facecolor('white')

    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Purples',
        annot_kws={'size': 9, 'family': 'Arial'},
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'ticks': cbar_ticks, 'shrink': 0.85, 'pad': 0.04},
        linewidths=0.3,
        linecolor='white',
        square=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=9, fontname='Arial', pad=4)
    ax.set_ylabel('True Label', fontsize=8, fontname='Arial', labelpad=2)
    ax.set_xlabel('Predicted Label', fontsize=8, fontname='Arial', labelpad=2)

    style_confusion_heatmap(ax, heatmap.collections[0].colorbar)
    fig.tight_layout(pad=0.35)
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


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
    
    plot_confusion_matrix(
        cm,
        os.path.join(args.output_dir, 'confusion_matrix_counts.pdf'),
        title='Confusion Matrix (Counts)',
        fmt='d',
        vmin=0,
        vmax=np.max(cm),
        cbar_ticks=np.linspace(0, np.max(cm), 5),
    )
    plot_confusion_matrix(
        cm_percentage,
        os.path.join(args.output_dir, 'confusion_matrix_percentages.pdf'),
        title='Confusion Matrix (%)',
        fmt='.1f',
        vmin=0,
        vmax=100,
        cbar_ticks=np.arange(0, 101, 20),
    )
    
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