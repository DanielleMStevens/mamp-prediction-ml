import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import math

class FiLMWithConcatenation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) with Concatenation.
    This module combines simple concatenation with FiLM conditioning to process
    sequence and receptor embeddings while incorporating multiple chemical features.
    """
    def __init__(self, feature_dim, num_chemical_features=3):
        """
        Initialize the FiLM with Concatenation module.
        Args:
            feature_dim (int): Dimension of the input features
            num_chemical_features (int): Number of chemical features to process (default: 3)
        """
        super(FiLMWithConcatenation, self).__init__()
        # Separate projections for peptide and receptor
        self.peptide_proj = nn.Linear(feature_dim, feature_dim)
        self.receptor_proj = nn.Linear(feature_dim, feature_dim)
        
        # Chemical feature processing (updated for multiple features)
        self.seq_chem_proj = nn.Sequential(
            nn.Linear(num_chemical_features, 64), # Input dimension changed
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.rec_chem_proj = nn.Sequential(
            nn.Linear(num_chemical_features, 64), # Input dimension changed
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Feature fusion (remains the same, combines pooled peptide/receptor features)
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, z, x_mask, z_mask, seq_chem_features=None, rec_chem_features=None):
        """
        Forward pass of the FiLM with Concatenation module.
        Args:
            x (torch.Tensor): Sequence embeddings
            z (torch.Tensor): Receptor embeddings
            x_mask (torch.Tensor): Mask for sequences (not used in concatenation)
            z_mask (torch.Tensor): Mask for receptors (not used in concatenation)
            seq_chem_features (torch.Tensor, optional): Chemical features for sequences (batch_size, num_features)
            rec_chem_features (torch.Tensor, optional): Chemical features for receptors (batch_size, num_features)
        Returns:
            torch.Tensor: Transformed and pooled features
        """
        batch_size = x.size(0)
        
        # Process sequences
        x = self.layer_norm(x)
        z = self.layer_norm(z)
        
        # Project sequences
        x_proj = self.peptide_proj(x)
        z_proj = self.receptor_proj(z)
        
        # Process chemical features
        if seq_chem_features is not None and rec_chem_features is not None:
            # Project the chemical features for sequences and receptors
            seq_chem_feat = self.seq_chem_proj(seq_chem_features.float()) # Use seq_chem_proj
            rec_chem_feat = self.rec_chem_proj(rec_chem_features.float()) # Use rec_chem_proj
            
            # Add chemical features to sequences (unsqueeze adds the sequence length dimension back)
            x_proj = x_proj + seq_chem_feat.unsqueeze(1)
            z_proj = z_proj + rec_chem_feat.unsqueeze(1)
        
        # Simple concatenation - use masked mean pooling for each sequence
        x_mask_expanded = x_mask.unsqueeze(-1).float()
        z_mask_expanded = z_mask.unsqueeze(-1).float()
        
        # Compute masked mean pooling
        x_pool_max = torch.max(x_proj * x_mask_expanded, dim=1)[0]
        x_pool_mean = (x_proj * x_mask_expanded).sum(dim=1) / (x_mask_expanded.sum(dim=1) + 1e-8)
        z_pool_max = torch.max(z_proj * z_mask_expanded, dim=1)[0]
        z_pool_mean = (z_proj * z_mask_expanded).sum(dim=1) / (z_mask_expanded.sum(dim=1) + 1e-8)
        
        # Concatenate all pooled features
        pooled = torch.cat([x_pool_max, x_pool_mean, z_pool_max, z_pool_mean], dim=-1)
        
        # Final fusion
        output = self.fusion_layer(pooled)
        output = self.dropout(output)
        
        return output

class ESMReceptorChemical(nn.Module):
    """
    Main model class that combines ESM embeddings with FiLM and concatenation mechanisms
    for predicting interactions between peptides and receptors.
    """
    def __init__(self, args, num_classes=3):
        """
        Initialize the ESM-based receptor-chemical interaction model.
        Args:
            args: Configuration arguments containing model backbone information
            num_classes (int): Number of output classes (default: 3)
        """
        super(ESMReceptorChemical, self).__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        # More selective layer freezing strategy
        # Freeze only the embedding and first 20 layers
        modules_to_freeze = [
            self.esm.embeddings,
            *self.esm.encoder.layer[:20]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        self.film = FiLMWithConcatenation(self.esm.config.hidden_size)
        
        # Enhanced classifier with residual connections
        hidden_size = self.esm.config.hidden_size
        self.classifier = nn.Sequential(
            # Layer 1: Dimensionality preservation
            nn.Linear(hidden_size, hidden_size),  # e.g., if hidden_size=1280 -> 1280
            nn.LayerNorm(hidden_size),            # Normalizes the outputs
            nn.ReLU(),                            # Activation function
            nn.Dropout(0.2),                      # Randomly drops 20% of neurons to prevent overfitting

            # Layer 2: Dimensionality reduction
            nn.Linear(hidden_size, hidden_size // 2),    # e.g., 1280 -> 640
            nn.LayerNorm(hidden_size // 2),              # Normalizes the reduced dimension
            nn.ReLU(),                                   # Activation function
            nn.Dropout(0.1),                             # Drops 10% of neurons

            # Output Layer
            nn.Linear(hidden_size // 2, num_classes)     # e.g., 640 -> 3 (for classification)
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.losses = ["ce"]

    def forward(self, batch_x):
        """
        Forward pass of the model.
        Args:
            batch_x (dict): Dictionary containing:
                - peptide_x (dict): Peptide sequence inputs
                - receptor_x (dict): Receptor sequence inputs
                - sequence_bulkiness (tensor): Bulkiness features for sequences
                - receptor_bulkiness (tensor): Bulkiness features for receptors
        Returns:
            torch.Tensor: Logits for classification
        """
        # Get ESM embeddings with gradient checkpointing
        with torch.set_grad_enabled(True):
            sequence_output = self.esm(**batch_x['peptide_x']).last_hidden_state
            receptor_output = self.esm(**batch_x['receptor_x']).last_hidden_state
        
        # Get attention masks
        sequence_mask = batch_x['peptide_x']['attention_mask']
        receptor_mask = batch_x['receptor_x']['attention_mask']
        
        # Process bulkiness features with proper error handling
        seq_bulkiness = batch_x.get('sequence_bulkiness')
        rec_bulkiness = batch_x.get('receptor_bulkiness')
        
        if seq_bulkiness is not None:
            seq_bulkiness = seq_bulkiness.float()
            seq_bulkiness = torch.nan_to_num(seq_bulkiness, nan=0.0)
            # Normalize bulkiness values
            seq_bulkiness = (seq_bulkiness - seq_bulkiness.mean()) / (seq_bulkiness.std() + 1e-8)
            
        if rec_bulkiness is not None:
            rec_bulkiness = rec_bulkiness.float()
            rec_bulkiness = torch.nan_to_num(rec_bulkiness, nan=0.0)
            # Normalize bulkiness values
            rec_bulkiness = (rec_bulkiness - rec_bulkiness.mean()) / (rec_bulkiness.std() + 1e-8)
        
        # Apply FiLM with attention
        combined = self.film(sequence_output, receptor_output, sequence_mask, receptor_mask,
                           seq_bulkiness, rec_bulkiness)
        
        # Get logits
        logits = self.classifier(combined)
        
        return logits

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        Args:
            batch (dict): Batch of data containing sequences, receptors, and labels
            batch_idx (int): Index of the current batch
        """
        logits = self(batch['x'])
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)
        
        loss = self.criterion(logits, batch['y']) + l2_lambda * l2_reg
        return loss

    def get_tokenizer(self):
        """Return the model's tokenizer"""
        return self.tokenizer

    def collate_fn(self, batch):
        """
        Collate function for creating batches from individual examples.
        Args:
            batch (list): List of examples containing peptide and receptor sequences,
                         and their corresponding bulkiness values
        Returns:
            dict: Collated batch with tokenized inputs, bulkiness features, and labels
        """
        inputs = {}
        x_dict = {}

        # Tokenize sequences
        x_dict['peptide_x'] = self.tokenizer([example['peptide_x'] for example in batch],
                return_tensors='pt', padding=True)
        x_dict['receptor_x'] = self.tokenizer([example['receptor_x'] for example in batch],
                return_tensors='pt', padding=True)
        
        # Add bulkiness features if they exist in the batch
        if 'sequence_bulkiness' in batch[0]:
            x_dict['sequence_bulkiness'] = torch.tensor([example['sequence_bulkiness'] for example in batch])
        if 'receptor_bulkiness' in batch[0]:
            x_dict['receptor_bulkiness'] = torch.tensor([example['receptor_bulkiness'] for example in batch])
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        inputs['x'] = x_dict
 
        return inputs

    def batch_decode(self, batch):
        """
        Decode a batch of tokenized sequences back to text.
        Args:
            batch (dict): Batch containing tokenized sequences
        Returns:
            list: Decoded sequences in format "peptide:receptor"
        """
        peptide_decoded_ls = self.tokenizer.batch_decode(batch['x']['peptide_x']['input_ids'], skip_special_tokens=True)
        receptor_decoded_ls = self.tokenizer.batch_decode(batch['x']['receptor_x']['input_ids'], skip_special_tokens=True)
        
        return [f"{peptide}:{receptor}" for peptide, receptor in zip(peptide_decoded_ls, receptor_decoded_ls)]

    def get_pr(self, logits):
        """
        Convert logits to probabilities using softmax.
        Args:
            logits (torch.Tensor): Raw model outputs
        Returns:
            torch.Tensor: Probability distributions
        """
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """
        Calculate various evaluation metrics.
        Args:
            gt (torch.Tensor): Ground truth labels
            pr (torch.Tensor): Predicted probabilities
            train (bool): Whether these are training or test statistics
        Returns:
            dict: Dictionary containing various evaluation metrics including:
                - Accuracy
                - Macro and weighted F1 scores
                - ROC AUC (multi-class)
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