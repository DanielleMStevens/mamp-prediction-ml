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

class ESM_chainRule_chemical(nn.Module):
    """
    ESM model enhanced with chemical features for peptide-receptor interaction prediction.
    Implements a two-stage ("chain rule") prediction:
    1. Immunogenic vs. Non-immunogenic
    2. Strong vs. Weak Immunogenic (for those classified as Immunogenic)
    """
    def __init__(self, args, num_classes=3): # num_classes is now unused but kept for compatibility
        super().__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        # Freeze early layers
        modules_to_freeze = [
            self.esm.embeddings,
            *self.esm.encoder.layer[:20]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        # Get feature dimension from ESM model
        self.hidden_size = self.esm.config.hidden_size
        
        # FiLM layer
        self.film = FiLMWithConcatenation(self.hidden_size)
        
        # Classification heads for the two stages
        # Stage 1: Non-immunogenic (0) vs Immunogenic (1)
        self.classifier_stage1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, 2)
        )
        # Stage 2: Weak (0) vs Strong (1) - applied only to immunogenic samples
        self.classifier_stage2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, 2)
        )
        
        # Loss with label smoothing, use reduction='none' for masking
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
        
        # Add losses attribute to match other models
        self.losses = ["ce"] # Use standard cross-entropy as the loss function
        
        # Save hyperparameters
        self.save_hyperparameters(args)
        
        # Register our custom loss function in the global loss_dict
        # Import here to avoid circular imports
        from engine_train import loss_dict
        loss_dict["ce"] = self.combined_loss_fn

    def save_hyperparameters(self, args):
        """Save hyperparameters."""
        self.hparams = args

    def forward(self, batch_x):
        """
        Forward pass of the model. Returns logits for both stages.
        """
        # Unpack the batch data - batch_x is now the inner dictionary
        if isinstance(batch_x, dict) and 'x' in batch_x:
            batch_x = batch_x['x']
        
        # Get ESM embeddings
        combined_tokens = batch_x['combined_tokens']
        # Ensure mask is boolean
        combined_mask = batch_x['combined_mask'].bool()
        
        # Get ESM embeddings
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
        conditioned_output = self.film(sequence_output, pooled_output, chemical_features)
        
        # Pool conditioned output (using the same boolean mask)
        masked_conditioned = conditioned_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        final_pooled, _ = torch.max(masked_conditioned, dim=1)
        
        # Classify for both stages
        logits_stage1 = self.classifier_stage1(final_pooled)
        logits_stage2 = self.classifier_stage2(final_pooled)
        
        return logits_stage1, logits_stage2

    def combined_loss_fn(self, output, batch):
        """Custom loss function that combines stage1 and stage2 losses."""
        logits_stage1, logits_stage2 = output
        labels = batch['y']
        
        # Stage 1: Non-immunogenic vs Immunogenic
        labels_stage1 = (labels > 0).long()
        loss_stage1 = self.criterion(logits_stage1, labels_stage1).mean()
        
        # Stage 2: Weak vs Strong, only for Immunogenic samples
        labels_stage2 = torch.zeros_like(labels)
        labels_stage2[labels == 1] = 0  # Weak
        labels_stage2[labels == 2] = 1  # Strong
        
        # Mask to include only immunogenic samples
        mask_stage2 = (labels > 0)
        
        if mask_stage2.sum() > 0:
            loss_stage2_unmasked = self.criterion(logits_stage2, labels_stage2)
            loss_stage2 = (loss_stage2_unmasked * mask_stage2).sum() / mask_stage2.sum()
        else:
            device = logits_stage2.device
            loss_stage2 = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Combine the losses
        combined_loss = loss_stage1 + loss_stage2
        
        # Return a dictionary with a single 'ce' key 
        return {'ce': combined_loss}

    def training_step(self, batch, batch_idx):
        """Training step with L2 regularization and two-stage loss."""
        logits_stage1, logits_stage2 = self(batch['x'])
        output = (logits_stage1, logits_stage2)

        # Get loss from our custom loss function
        loss_dict = self.combined_loss_fn(output, batch)
        combined_loss = loss_dict['ce']
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=logits_stage1.device)
        for name, param in self.named_parameters():
            if param.requires_grad and ('weight' in name or 'bias' not in name) and 'LayerNorm' not in name:
                l2_reg = l2_reg + torch.norm(param)
        
        loss = combined_loss + l2_lambda * l2_reg
        return loss

    def collate_fn(self, batch):
        """Collate function for batching."""
        tokenizer = self.tokenizer
        
        # Get sequences and labels
        sequences = [str(item['peptide_x']) for item in batch]
        receptors = [str(item['receptor_x']) for item in batch]
        labels = torch.tensor([item['y'] for item in batch])
        
        # Combine sequences with separator
        combined = [f"{seq} {tokenizer.sep_token} {rec}" for seq, rec in zip(sequences, receptors)]
        
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
            features = {}
            for feat in ['bulkiness', 'charge', 'hydrophobicity']:
                key = f"{prefix}_{feat}"
                if key in batch[0]:
                    features[feat] = torch.tensor([item[key] for item in batch])
                else:
                    features[feat] = torch.zeros(len(batch), encoded['input_ids'].size(1))
            return features
        
        seq_features = process_features(batch, 'sequence')
        rec_features = process_features(batch, 'receptor')
        
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
        """Get prediction probabilities for the original 3 classes by combining stage probabilities."""
        logits_stage1, logits_stage2 = logits # Unpack the tuple
        
        pr_stage1 = torch.softmax(logits_stage1, dim=-1) # P(Non), P(Immuno)
        pr_stage2 = torch.softmax(logits_stage2, dim=-1) # P(Weak | Immuno), P(Strong | Immuno)
        
        # Combine probabilities:
        # P(Non) = P(Non | Stage 1)
        # P(Weak) = P(Immuno | Stage 1) * P(Weak | Stage 2)
        # P(Strong) = P(Immuno | Stage 1) * P(Strong | Stage 2)
        prob_non = pr_stage1[:, 0]
        prob_weak = pr_stage1[:, 1] * pr_stage2[:, 0]
        prob_strong = pr_stage1[:, 1] * pr_stage2[:, 1]
        
        # Stack to get final probabilities (batch_size, 3)
        final_probs = torch.stack([prob_non, prob_weak, prob_strong], dim=-1)
        return final_probs

    def get_stats(self, gt, pr, train=False):
        """Calculate evaluation metrics for overall, stage 1, and stage 2."""
        # Ensure gt and pr are on CPU
        gt = gt.cpu()
        pr = pr.cpu() # pr are the final 3-class probabilities
        
        prefix = "train" if train else "test"
        pred_labels_final = pr.argmax(dim=-1) # Final 3-class prediction
        
        # --- Overall 3-Class Stats ---
        stats = {
            f"{prefix}_acc": accuracy_score(gt, pred_labels_final),
            f"{prefix}_f1_macro": f1_score(gt, pred_labels_final, average='macro', zero_division=0),
            f"{prefix}_f1_weighted": f1_score(gt, pred_labels_final, average='weighted', zero_division=0)
        }
        try:
            # Check if there are enough samples and classes for AUROC/AUPRC
            if len(np.unique(gt)) > 1 and pr.shape[1] > 1:
                 # Ensure probabilities sum close to 1 for roc_auc_score multi_class='ovr'
                 pr_normalized = pr / pr.sum(dim=1, keepdim=True).clamp(min=1e-9)
                 stats[f"{prefix}_auroc"] = roc_auc_score(gt, pr_normalized, multi_class='ovr')

                 gt_onehot = F.one_hot(gt, num_classes=3).numpy()
                 pr_np = pr.numpy()

                 auprc_scores = []
                 for i in range(3):
                     # Handle cases where a class might not be present in gt
                     if np.sum(gt_onehot[:, i]) > 0:
                         precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                         auprc_scores.append(auc(recall, precision))
                     else:
                         auprc_scores.append(0.0) # Or np.nan? Assign 0 for now.

                 stats[f"{prefix}_auprc_macro"] = np.mean(auprc_scores)
                 for i in range(3):
                     stats[f"{prefix}_auprc_class{i}"] = auprc_scores[i]
            else:
                 raise ValueError("Not enough classes/samples for AUC calculation")

        except Exception as e:
            # print(f"Warning: Could not calculate overall AUC stats ({prefix}): {e}") # Optional warning
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0

        # --- Stage 1 Stats (Non-immunogenic vs Immunogenic) ---
        gt_stage1 = (gt > 0).long()
        # Calculate Stage 1 probabilities: P(Immuno) = P(Weak) + P(Strong)
        pr_stage1_immuno = pr[:, 1] + pr[:, 2]
        pred_labels_stage1 = (pr_stage1_immuno > 0.5).long() # Threshold at 0.5

        stats[f"{prefix}_stage1_acc"] = accuracy_score(gt_stage1, pred_labels_stage1)
        stats[f"{prefix}_stage1_f1"] = f1_score(gt_stage1, pred_labels_stage1, zero_division=0) # Binary F1
        try:
             if len(np.unique(gt_stage1)) > 1:
                 stats[f"{prefix}_stage1_auroc"] = roc_auc_score(gt_stage1, pr_stage1_immuno)
                 precision, recall, _ = precision_recall_curve(gt_stage1, pr_stage1_immuno)
                 stats[f"{prefix}_stage1_auprc"] = auc(recall, precision)
             else:
                  raise ValueError("Not enough classes for Stage 1 AUC")
        except Exception as e:
            # print(f"Warning: Could not calculate Stage 1 AUC stats ({prefix}): {e}")
            stats[f"{prefix}_stage1_auroc"] = 0.0
            stats[f"{prefix}_stage1_auprc"] = 0.0

        # --- Stage 2 Stats (Weak vs Strong, only for immunogenic samples) ---
        mask_stage2 = gt > 0
        stats[f"{prefix}_stage2_acc"] = 0.0
        stats[f"{prefix}_stage2_f1"] = 0.0
        stats[f"{prefix}_stage2_auroc"] = 0.0
        stats[f"{prefix}_stage2_auprc"] = 0.0
        if mask_stage2.sum() > 0:
            gt_stage2_filtered = gt[mask_stage2] - 1 # Map Weak (1) -> 0, Strong (2) -> 1

            # Calculate conditional probabilities P(Strong | Immuno)
            pr_immuno_filtered = pr_stage1_immuno[mask_stage2].clamp(min=1e-9)
            pr_strong_filtered = pr[mask_stage2, 2]
            pr_stage2_strong_cond = pr_strong_filtered / pr_immuno_filtered

            pred_labels_stage2 = (pr_stage2_strong_cond > 0.5).long() # Threshold at 0.5

            if len(gt_stage2_filtered) > 0 : # Ensure there are samples left after filtering
                 stats[f"{prefix}_stage2_acc"] = accuracy_score(gt_stage2_filtered, pred_labels_stage2)
                 stats[f"{prefix}_stage2_f1"] = f1_score(gt_stage2_filtered, pred_labels_stage2, zero_division=0) # Binary F1
                 try:
                      if len(np.unique(gt_stage2_filtered)) > 1:
                           stats[f"{prefix}_stage2_auroc"] = roc_auc_score(gt_stage2_filtered, pr_stage2_strong_cond)
                           precision, recall, _ = precision_recall_curve(gt_stage2_filtered, pr_stage2_strong_cond)
                           stats[f"{prefix}_stage2_auprc"] = auc(recall, precision)
                      else:
                           raise ValueError("Not enough classes for Stage 2 AUC")
                 except Exception as e:
                     # print(f"Warning: Could not calculate Stage 2 AUC stats ({prefix}): {e}")
                     stats[f"{prefix}_stage2_auroc"] = 0.0 # Set default if calculation fails
                     stats[f"{prefix}_stage2_auprc"] = 0.0 # Set default if calculation fails

        return stats