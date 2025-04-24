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
        """
        Initialize the ESM model for protein sequence classification.
        Args:
            args: Optional configuration arguments (currently unused)
        """
        super().__init__()
        # Load pre-trained ESM-2 model and freeze its parameters
        # ESM2_t30_150M_UR50D is a 30-layer transformer model with 150M parameters
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        # Freeze the base model parameters to prevent fine-tuning
        # This preserves the pre-trained protein knowledge
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        # Get embedding dimension from model config (typically 1280 for ESM2-150M)
        E = self.esm_model.config.hidden_size

        # Classification head: 3-layer MLP with ReLU activation
        # Architecture: Input_dim -> Input_dim -> Input_dim/2 -> 3 classes
        # The progressive dimension reduction helps in learning hierarchical features
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),  # Final layer outputs logits for 3 classes
        )
        
        # Initialize tokenizer for protein sequences
        # This tokenizer handles protein-specific tokens and special characters
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  
        
        # Specify the loss function(s) to use
        self.losses = ["ce"]  # Cross-entropy loss for multi-class classification

    def forward(self, batch_x):
        """
        Forward pass of the model.
        Args:
            batch_x: Tokenized input sequences
        Returns:
            Logits for the 3 classes
        """
        # Get CLS token embedding (first token) which represents the entire sequence
        # Shape: (batch_size, embedding_dim)
        embeddings = self.esm_model(**batch_x).last_hidden_state[:, 0, :]
        return self.net(embeddings)  # Pass through classification head
    
    def get_tokenizer(self):
        """Returns the ESM tokenizer instance"""
        return self.tokenizer

    def collate_fn(self, batch):
        """
        Prepares a batch of data for training/inference.
        Args:
            batch: List of dictionaries containing 'x' (sequences) and 'y' (labels)
        Returns:
            dict: Contains tokenized inputs and labels
        """
        inputs = {}
        # Tokenize protein sequences with padding for batch processing
        inputs['x'] = self.tokenizer([example['x'] for example in batch],
                return_tensors='pt', padding=True)
        # Convert labels to tensor
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        return inputs

    def batch_decode(self, batch):
        """
        Decodes tokenized sequences back to amino acid sequences.
        Args:
            batch: Batch of tokenized sequences
        Returns:
            list: Original amino acid sequences
        """
        return self.tokenizer.batch_decode(batch['x']['input_ids'], skip_special_tokens=True)

    def get_pr(self, logits):
        """
        Convert logits to probabilities using softmax
        Args:
            logits: Raw model outputs (batch_size, num_classes)
        Returns:
            torch.Tensor: Probability distributions over classes
        """
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """
        Calculate various evaluation metrics for model performance
        Args:
            gt: Ground truth labels
            pr: Model predictions (probabilities)
            train: Boolean indicating if these are training or test metrics
        Returns:
            dict: Dictionary containing various evaluation metrics:
                - Accuracy
                - Macro and weighted F1 scores
                - ROC AUC (one-vs-rest)
                - PR AUC for each class and macro average
        """
        prefix = "train" if train else "test"
        pred_labels = pr.argmax(dim=-1)  # Convert probabilities to class predictions
        
        # Calculate basic classification metrics
        stats = {
            f"{prefix}_acc": accuracy_score(gt.cpu(), pred_labels.cpu()),
            f"{prefix}_f1_macro": f1_score(gt.cpu(), pred_labels.cpu(), average='macro'),
            f"{prefix}_f1_weighted": f1_score(gt.cpu(), pred_labels.cpu(), average='weighted')
        }
        
        try:
            # Calculate ROC AUC using one-vs-rest approach for multi-class
            stats[f"{prefix}_auroc"] = roc_auc_score(gt.cpu(), pr.cpu(), multi_class='ovr')
            
            # Calculate PR AUC for each class separately
            gt_onehot = np.eye(3)[gt.cpu()]  # Convert to one-hot encoding
            pr_np = pr.cpu().numpy()
            
            # Calculate per-class PR curves and their AUCs
            for i in range(3):
                precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                stats[f"{prefix}_auprc_class{i}"] = auc(recall, precision)
            
            # Calculate macro-average of PR AUCs
            stats[f"{prefix}_auprc_macro"] = np.mean([stats[f"{prefix}_auprc_class{i}"] for i in range(3)])
            
        except:
            # Handle edge cases where metrics cannot be calculated
            # (e.g., missing classes in batch)
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0
            
        return stats