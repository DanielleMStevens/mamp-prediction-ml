"""
A model that uses Facebook's ESM-2 (Evolutionary Scale Modeling) for protein sequence analysis.
This model leverages the pre-trained ESM2 150M parameter model for protein understanding.

The model:
1. Uses a frozen ESM-2 model as the base encoder
2. Takes the CLS token embedding for sequence representation
3. Passes through a 3-layer MLP to predict 3 classes

ESM-2 is a state-of-the-art protein language model trained on evolutionary data.
This implementation keeps the base ESM model frozen and only trains the classification head.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np


class ESMModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # Load pre-trained ESM-2 model and freeze its parameters
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        # Freeze the base model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        E = self.esm_model.config.hidden_size  # Get embedding dimension from model config

        # Classification head: 3-layer MLP with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),
        )
        # Initialize tokenizer for protein sequences
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  
        
        # Specify the loss function(s) to use
        self.losses = ["ce"]  # Cross-entropy loss

    def forward(self, batch_x):
        embeddings = self.esm_model(**batch_x).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        return self.net(embeddings) # (B, S)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = self.tokenizer([example['x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
 
        return inputs

    def batch_decode(self, batch):
        return self.tokenizer.batch_decode(batch['x']['input_ids'], skip_special_tokens=True)

    def get_pr(self, logits):
        """
        Convert logits to probabilities using softmax
        Args:
            logits: Raw model outputs
        Returns:
            torch.Tensor: Probability distributions
        """
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """
        Calculate various evaluation metrics
        Args:
            gt: Ground truth labels
            pr: Model predictions (probabilities)
            train: Boolean indicating if these are training or test metrics
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        prefix = "train" if train else "test"
        pred_labels = pr.argmax(dim=-1)
        
        stats = {
            f"{prefix}_acc": accuracy_score(gt.cpu(), pred_labels.cpu()),
            f"{prefix}_f1_macro": f1_score(gt.cpu(), pred_labels.cpu(), average='macro'),
            f"{prefix}_f1_weighted": f1_score(gt.cpu(), pred_labels.cpu(), average='weighted')
        }
        
        try:
            # Calculate ROC AUC
            stats[f"{prefix}_auroc"] = roc_auc_score(gt.cpu(), pr.cpu(), multi_class='ovr')
            
            # Calculate PR AUC for each class
            gt_onehot = np.eye(3)[gt.cpu()]
            pr_np = pr.cpu().numpy()
            
            for i in range(3):
                precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                stats[f"{prefix}_auprc_class{i}"] = auc(recall, precision)
            
            # Average AUPRC across classes
            stats[f"{prefix}_auprc_macro"] = np.mean([stats[f"{prefix}_auprc_class{i}"] for i in range(3)])
            
        except:
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0
            
        return stats