import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import torch.nn.functional as F

class FiLMWithConcatenation(nn.Module):
    """
    FiLM layer that conditions sequence representation with chemical features.
    """
    def __init__(self, feature_dim):
        super().__init__()
        # Process chemical features
        self.chemical_proj = nn.Sequential(
            nn.Linear(3, 64),  # Project 3 chemical features
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # FiLM layer
        self.film_layer = nn.Linear(feature_dim * 2, feature_dim * 2)  # For gamma and beta
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, z, chemical_features=None):
        """
        Args:
            x: Sequence embeddings (batch_size, seq_len, feature_dim)
            z: Pooled context vector (batch_size, feature_dim)
            chemical_features: Combined chemical features (batch_size, seq_len, 6) 
                             [3 for sequence, 3 for receptor]
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Process sequence embeddings
        x = self.layer_norm(x)
        
        # Expand z to match sequence length
        z = z.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Process chemical features if provided
        if chemical_features is not None:
            # Split into sequence and receptor features
            seq_features, rec_features = torch.split(chemical_features, 3, dim=-1)
            # Project each set of features
            seq_chem = self.chemical_proj(seq_features)
            rec_chem = self.chemical_proj(rec_features)
            # Combine with embeddings
            z = z + seq_chem + rec_chem
        
        # Concatenate and generate FiLM parameters
        combined = torch.cat([x, z], dim=-1)
        gamma_beta = self.film_layer(combined)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # Apply FiLM conditioning
        output = gamma * x + beta
        return self.dropout(output)

class ESMallChemicalFeatures(nn.Module):
    """
    ESM model enhanced with chemical features for peptide-receptor interaction prediction.
    """
    def __init__(self, args, num_classes=3):
        super().__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        #esm2_t33_650M_UR50D

        # Freeze early layers, leaving only the last 10 encoder layers trainable
        # (ESM2-t30 has 30 layers total, so layers 20-29 will be trainable)
        # Previous version - training last 10 layers
        # modules_to_freeze = [
        #     self.esm.embeddings,
        #     *self.esm.encoder.layer[:20]  # Freeze first 20 layers, leaving 10 trainable
        # ]
        # for module in modules_to_freeze:
        #     for param in module.parameters():
        #         param.requires_grad = False

        # New version - only train last layer
        modules_to_freeze = [
            self.esm.embeddings,
            *self.esm.encoder.layer[:30]  # Freeze first 29 layers, leaving only last layer trainable
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        # Get feature dimension from ESM model
        self.hidden_size = self.esm.config.hidden_size
        
        # FiLM layer
        self.film = FiLMWithConcatenation(self.hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            # Reduce from 1280 -> 640 dimensions
            nn.Linear(self.hidden_size, self.hidden_size // 2),  
            nn.LayerNorm(self.hidden_size // 2),  # Normalize the 640-dim output
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.2),  # Prevent overfitting
            # Final projection from 640 -> 3 classes (immune response categories)
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Add losses attribute to match other models
        self.losses = ["ce"]  # Cross-entropy loss
        
        # Save hyperparameters
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        """Save hyperparameters."""
        self.hparams = args

    def forward(self, batch_x):
        """
        Forward pass of the model.
        """
        # Unpack the batch data - batch_x is now the inner dictionary
        if isinstance(batch_x, dict) and 'x' in batch_x:
            batch_x = batch_x['x']
        
        # Get ESM embeddings
        combined_tokens = batch_x['combined_tokens']
        # Convert mask to boolean tensor - this mask indicates valid tokens vs padding
        # Used by ESM model to avoid attending to padding tokens during self-attention
        combined_mask = batch_x['combined_mask'].bool()
        
        # Get ESM embeddings, passing the attention mask to properly handle padding
        outputs = self.esm(
            input_ids=combined_tokens,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        sequence_output = outputs.last_hidden_state
        
        # Pool for context vector (max pooling)
        masked_output = sequence_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        pooled_output, _ = torch.max(masked_output, dim=1)
        
        # Combine chemical features
        chemical_features = torch.stack([
            batch_x['seq_bulkiness'],
            batch_x['seq_charge'],
            batch_x['seq_hydrophobicity'],
            batch_x['rec_bulkiness'],
            batch_x['rec_charge'],
            batch_x['rec_hydrophobicity']
        ], dim=-1)
        
        # Apply FiLM conditioning
        # FiLM (Feature-wise Linear Modulation) modulates the sequence features based on:
        # 1. pooled_output: The max-pooled ESM embeddings that capture global sequence context
        # 2. chemical_features: The chemical properties (bulkiness, charge, hydrophobicity) of both peptide and receptor
        # This helps the model learn interactions between sequence and chemical properties
        conditioned_output = self.film(sequence_output, pooled_output, chemical_features)
        
        # Pool conditioned output (using the same boolean mask)
        # 1. Mask out padding tokens by setting them to negative infinity
        # 2. Take max over sequence dimension to get a fixed-size representation
        # This creates a single vector per sequence that captures the most important features
        masked_conditioned = conditioned_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        final_pooled, _ = torch.max(masked_conditioned, dim=1)
        # Classify
        logits = self.classifier(final_pooled)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step with L2 regularization."""
        logits = self(batch['x'])
        labels = batch['y']
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=logits.device, requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)
        
        loss = self.criterion(logits, labels) + l2_lambda * l2_reg
        return loss

    def collate_fn(self, batch):
        """Collate function for batching."""
        tokenizer = self.tokenizer
        
        # Get sequences and labels
        # Extract peptide sequences and convert to strings, handling any non-string inputs
        sequences = [str(item['peptide_x']).strip() for item in batch]
        
        # Extract receptor sequences and convert to strings, handling any non-string inputs  
        receptors = [str(item['receptor_x']).strip() for item in batch]
        
        # Extract labels and convert to tensor, ensuring proper dtype
        labels = torch.tensor([item['y'] for item in batch], dtype=torch.long)
        
        # Combine peptide and receptor sequences with ESM separator token
        # Format: [peptide sequence] [SEP] [receptor sequence]
        # The separator token helps the model distinguish between peptide and receptor
        combined = [
            f"{peptide_seq} {tokenizer.sep_token} {receptor_seq}" 
            for peptide_seq, receptor_seq in zip(sequences, receptors)
        ]
        
        # Tokenize with max_length parameter
        encoded = tokenizer(
            combined,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        
        # Process chemical features
        def process_features(batch, prefix):
            # This function processes chemical features (bulkiness, charge, hydrophobicity) for sequences
            # For each feature type:
            # - If the feature exists in the batch data (e.g. 'sequence_bulkiness'), extract it into a tensor
            # - If the feature doesn't exist, create a zero tensor matching the sequence length
            features = {}
            for feat in ['bulkiness', 'charge', 'hydrophobicity']:
                key = f"{prefix}_{feat}"
                if key in batch[0]:
                    # Extract feature values from batch into tensor
                    features[feat] = torch.tensor([item[key] for item in batch])
                else:
                    # Create zero tensor with shape matching the sequence length
                    features[feat] = torch.zeros(len(batch), encoded['input_ids'].size(1))
            return features
        
        # Process features for both peptide sequences and receptor sequences
        seq_features = process_features(batch, 'sequence')  # Gets chemical features for peptide sequences
        rec_features = process_features(batch, 'receptor')  # Gets chemical features for receptor sequences
        
        # Return with 'y' instead of 'labels' to match expected structure
        return {
            'x': {
                'combined_tokens': encoded['input_ids'],
                'combined_mask': encoded['attention_mask'],
                'seq_bulkiness': seq_features['bulkiness'],
                'seq_charge': seq_features['charge'],
                'seq_hydrophobicity': seq_features['hydrophobicity'],
                'rec_bulkiness': rec_features['bulkiness'],
                'rec_charge': rec_features['charge'],
                'rec_hydrophobicity': rec_features['hydrophobicity'],
            },
            'y': labels
        }

    def get_tokenizer(self):
        return self.tokenizer

    def batch_decode(self, batch):
        """Decode tokenized sequences."""
        # Handle both nested and flat batch structures
        if isinstance(batch, dict) and 'x' in batch:
            tokens = batch['x']['combined_tokens']
        else:
            tokens = batch['combined_tokens']
            
        decoded = self.tokenizer.batch_decode(
            tokens,
            skip_special_tokens=True
        )
        return [seq.split(self.tokenizer.sep_token) for seq in decoded]

    def get_pr(self, logits):
        """Get prediction probabilities."""
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """Calculate evaluation metrics."""
        prefix = "train" if train else "test"
        pred_labels = pr.argmax(dim=-1)
        
        stats = {
            f"{prefix}_acc": accuracy_score(gt.cpu(), pred_labels.cpu()),
            f"{prefix}_f1_macro": f1_score(gt.cpu(), pred_labels.cpu(), average='macro'),
            f"{prefix}_f1_weighted": f1_score(gt.cpu(), pred_labels.cpu(), average='weighted')
        }
        
        try:
            stats[f"{prefix}_auroc"] = roc_auc_score(gt.cpu(), pr.cpu(), multi_class='ovr')
            
            gt_onehot = np.eye(3)[gt.cpu()]
            pr_np = pr.cpu().numpy()
            
            for i in range(3):
                precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                stats[f"{prefix}_auprc_class{i}"] = auc(recall, precision)
            
            stats[f"{prefix}_auprc_macro"] = np.mean([stats[f"{prefix}_auprc_class{i}"] for i in range(3)])
            
        except:
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0
            
        return stats