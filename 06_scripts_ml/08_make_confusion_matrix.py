#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Confusion Matrix Generation Script', add_help=False)
    parser.add_argument('--predictions_path', default='test_preds.pth', type=str, 
                        help='Path to predictions file')
    parser.add_argument('--output_dir', default='./results', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    predictions = torch.load(args.predictions_path)
    
    # Get ground truth and predictions
    gt = predictions['gt'].detach().cpu().numpy() #ground truth
    preds = predictions['pr'].detach().cpu().numpy() #predictions
    
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
    
    print("\nClassification Report Exported.")
    report = classification_report(gt, preds)
    
    # Save classification report to a text file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    args = parse_args()
    main(args) 