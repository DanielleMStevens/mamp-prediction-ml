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
    
    # Create confusion matrix
    cm = confusion_matrix(gt, preds)
    
    # Create confusion matrix plot
    plt.figure(figsize=(1.5, 1.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 8})
    plt.title('Confusion Matrix', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.xlabel('Predicted Label', fontsize=7)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout(pad=0.5)  
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.pdf'))
    plt.close()
    
    print("\nClassification Report:")
    report = classification_report(gt, preds)
    print(report)
    
    # Save classification report to a text file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    args = parse_args()
    main(args) 