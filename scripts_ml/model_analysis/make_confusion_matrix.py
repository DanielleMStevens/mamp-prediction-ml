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
    parser.add_argument('--predictions_path', default='test_preds.pth', type=str, help='Path to predictions file')
    parser.add_argument('--output_dir', default='./results', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    predictions = torch.load(args.predictions_path)
    
    # Get ground truth and predictions
    gt = predictions['gt'].detach().cpu().numpy()
    preds = predictions['pr'].detach().cpu().numpy()
    
    # For probability outputs, convert to class predictions
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(gt, preds)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(gt, preds))

if __name__ == '__main__':
    args = parse_args()
    main(args) 