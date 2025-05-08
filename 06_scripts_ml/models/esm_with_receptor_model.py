#esm_with_receptor_model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

class FiLMWithAttention(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) with attention mechanism.
    This module combines attention and FiLM to modulate features based on context.
    """
    def __init__(self, feature_dim):
        super(FiLMWithAttention, self).__init__()
        # Projection layers for key-query-value attention mechanism
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        # Output projection for FiLM parameters (gamma and beta)
        self.out_proj = nn.Linear(feature_dim, feature_dim * 2)

    def forward(self, x, z, x_mask, z_mask):
        # x: (batch_size, seq_len_x, feature_dim)
        # z: (batch_size, seq_len_z, feature_dim)
        query = self.query_proj(x)  # Shape: (batch_size, seq_len_x, feature_dim)
        keys = self.key_proj(z)  # Shape: (batch_size, seq_len_z, feature_dim)
        values = self.value_proj(z)  # Shape: (batch_size, seq_len_z, feature_dim)

        # Compute attention mask
        attention_mask = x_mask.unsqueeze(2) & z_mask.unsqueeze(1)  # Shape: (batch_size, seq_len_x, seq_len_z)
        attention_mask = attention_mask.float().masked_fill(~attention_mask, float('-inf'))
        feature_dim = x.shape[-1]

        # Compute scaled dot-product attention
        attention_weights = torch.softmax(
            torch.matmul(query, keys.transpose(-2, -1)) / (feature_dim ** 0.5) + attention_mask, dim=-1
        )
        context = torch.matmul(attention_weights, values)  # Shape: (batch_size, seq_len_x, feature_dim)

        # Generate gamma and beta
        gamma_beta = self.out_proj(context)  # Shape: (batch_size, seq_len_x, feature_dim * 2)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # Apply FiLM transformation
        return gamma * x + beta

class FiLM(nn.Module):
    """
    Basic Feature-wise Linear Modulation (FiLM) module.
    Applies a learned affine transformation to features based on conditioning information.
    """
    def __init__(self, feature_dim):
        super(FiLM, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim * 2)  # Generate gamma and beta

    def forward(self, x, z):
        # x: (batch_size, feature_dim)
        # z: (batch_size, feature_dim)
        gamma_beta = self.fc(z)  # Shape: (batch_size, feature_dim * 2)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # Split into gamma and beta
        return gamma * x + beta  # Apply FiLM

class ESMWithReceptorModel(nn.Module):
    """
    Main model class combining ESM embeddings with receptor information for peptide-receptor interaction prediction.
    Uses ESM2 language model for protein sequence embeddings and FiLM for feature modulation.
    """
    def __init__(self, args=None):
        super().__init__()
        # Load pre-trained ESM model and freeze its parameters
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        E = self.esm_model.config.hidden_size

        # Classification head network
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),  # 3 classes output
        )

        self.film = FiLM(E)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  
        self.losses = ["ce"]  # Cross-entropy loss for classification

    def forward(self, batch_x):
        """
        Forward pass of the model
        Args:
            batch_x: Dictionary containing peptide and receptor sequences
        Returns:
            Classification logits for 3 classes
        """
        # Get embeddings for peptide and receptor sequences
        batch_peptide_x = batch_x['peptide_x']
        batch_receptor_x = batch_x['receptor_x']
        peptide_embeddings = self.esm_model(**batch_peptide_x).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        receptor_embeddings = self.esm_model(**batch_receptor_x).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        embeddings_1 = self.film(peptide_embeddings, receptor_embeddings)
        embeddings_2 = self.film(receptor_embeddings, peptide_embeddings)
        
        # Combine modulated embeddings and classify
        return self.net(embeddings_1 + embeddings_2)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}
        
        x_dict = {}

        x_dict['peptide_x'] = self.tokenizer([example['peptide_x'] for example in batch],
                return_tensors='pt', padding=True)
        x_dict['receptor_x'] = self.tokenizer([example['receptor_x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])

        inputs['x'] = x_dict
 
        return inputs

    def batch_decode(self, batch):
        peptide_decoded_ls = self.tokenizer.batch_decode(batch['x']['peptide_x']['input_ids'], skip_special_tokens=True)
        receptor_decoded_ls = self.tokenizer.batch_decode(batch['x']['receptor_x']['input_ids'], skip_special_tokens=True)
        
        return [f"{peptide}:{receptor}" for peptide, receptor in zip(peptide_decoded_ls, receptor_decoded_ls)]

    def get_pr(self, logits):
        """Get predictions from logits"""
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """
        Calculate various evaluation metrics for model performance
        Args:
            gt: Ground truth labels
            pr: Model predictions (probabilities)
            train: Boolean indicating if these are training or test metrics
        Returns:
            Dictionary containing various evaluation metrics:
            - Accuracy
            - Macro and weighted F1 scores
            - ROC AUC (one-vs-rest)
            - PR AUC for each class and macro-averaged
        """
        prefix = "train" if train else "test"
        pred_labels = pr.argmax(dim=-1)
        
        stats = {
            f"{prefix}_acc": accuracy_score(gt.cpu(), pred_labels.cpu()),
            f"{prefix}_f1_macro": f1_score(gt.cpu(), pred_labels.cpu(), average='macro'),
            f"{prefix}_f1_weighted": f1_score(gt.cpu(), pred_labels.cpu(), average='weighted')
        }
        
        try:
            # Calculate ROC AUC and PR AUC metrics
            stats[f"{prefix}_auroc"] = roc_auc_score(gt.cpu(), pr.cpu(), multi_class='ovr')
            
            # Calculate PR AUC for each class
            gt_onehot = np.eye(3)[gt.cpu()]
            pr_np = pr.cpu().numpy()
            
            # Calculate per-class PR AUC
            for i in range(3):
                precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                stats[f"{prefix}_auprc_class{i}"] = auc(recall, precision)
            
            # Average AUPRC across classes
            stats[f"{prefix}_auprc_macro"] = np.mean([stats[f"{prefix}_auprc_class{i}"] for i in range(3)])
            
            # Save predictions to CSV if this is test data
            if not train:
                # Convert predictions and ground truth to numpy arrays
                probs = pr.cpu().numpy()
                labels = gt.cpu().numpy()
                
                # Create DataFrame with probabilities and true labels
                results_df = pd.DataFrame(probs, columns=['prob_class0', 'prob_class1', 'prob_class2'])
                results_df['true_label'] = labels
                results_df['predicted_label'] = pred_labels.cpu().numpy()
                
                # Save to CSV
                results_df.to_csv('test_predictions.csv', index=False)
            
        except:
            # Handle cases where metrics cannot be calculated
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0
            
        return stats