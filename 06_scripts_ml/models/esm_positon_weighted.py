import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os

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

class BFactorWeightGenerator:
    """
    Generates weights based on B-factors from preprocessed data.
    """
    def __init__(self, bfactor_csv_path=None, min_weight=0.1, max_weight=5.0):
        # Make the path parameter optional with a default fallback
        if bfactor_csv_path is None:
            # Try different common paths
            possible_paths = [
                "04_Preprocessing_results/bfactor_winding_lrr_segments.csv"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    bfactor_csv_path = path
                    break
            
            if bfactor_csv_path is None:
                print("Warning: B-factor CSV file not found. Using default weights.")
                self.bfactor_data = {}
                return

        try:
            self.bfactor_data = self._load_bfactor_data(bfactor_csv_path)
        except Exception as e:
            print(f"Warning: Failed to load B-factor data: {e}. Using default weights.")
            self.bfactor_data = {}
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def _load_bfactor_data(self, csv_path):
        """Load and process B-factor data from CSV."""
        df = pd.read_csv(csv_path)
        
        # Group by protein
        protein_data = {}
        for protein_key, group in df.groupby('Protein Key'):
            # Convert protein key format to match training data format
            # Assuming protein_key is in format "Species_LocusID_Receptor"
            parts = protein_key.split('_')
            if len(parts) >= 3:
                species = parts[0].replace('_', ' ')
                locus = parts[1]
                receptor = parts[2]
                training_key = f"{species}|{locus}|{receptor}"
                protein_data[training_key] = {
                    'residue_idx': group['Residue Index'].values,
                    'bfactors': group['Filtered B-Factor'].values
                }
                # Also store with original key as fallback
                protein_data[protein_key] = {
                    'residue_idx': group['Residue Index'].values,
                    'bfactors': group['Filtered B-Factor'].values
                }
        return protein_data
    
    def get_weights(self, protein_key, sequence_length):
        """
        Generate weights for a specific protein sequence.
        """
        # Default weights
        weights = torch.ones(sequence_length) * self.min_weight
        
        # Try both original and converted key formats
        if protein_key in self.bfactor_data:
            data = self.bfactor_data[protein_key]
        else:
            # Try converting the training format to B-factor format
            converted_key = protein_key.replace('|', '_').replace(' ', '_')
            if converted_key in self.bfactor_data:
                data = self.bfactor_data[converted_key]
            else:
                return weights
        
        bfactors = data['bfactors']
        residue_idx = data['residue_idx']
        
        # Convert B-factors to weights
        pos_mask = bfactors > 0
        if pos_mask.any():
            pos_bfactors = bfactors[pos_mask]
            pos_weights = self.min_weight + (self.max_weight - self.min_weight) * (
                pos_bfactors / pos_bfactors.max()
            )
            
            # Assign weights to corresponding positions
            for idx, weight in zip(residue_idx[pos_mask], pos_weights):
                if idx < sequence_length:
                    weights[idx] = weight
        
        return weights

class ESMBfactorWeightedFeatures(nn.Module):
    """
    ESM model with B-factor weighted features for peptide-receptor interaction prediction.
    """
    def __init__(self, args, num_classes=3):
        super().__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        # --- Debug Tokenizer ---
        print(f"DEBUG Tokenizer Info:")
        print(f"  - sep_token: {self.tokenizer.sep_token}, sep_token_id: {self.tokenizer.sep_token_id}")
        print(f"  - eos_token: {self.tokenizer.eos_token}, eos_token_id: {self.tokenizer.eos_token_id}")
        print(f"  - cls_token: {self.tokenizer.cls_token}, cls_token_id: {self.tokenizer.cls_token_id}")
        print(f"  - pad_token: {self.tokenizer.pad_token}, pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"  - unk_token: {self.tokenizer.unk_token}, unk_token_id: {self.tokenizer.unk_token_id}")
        # --- End Debug Tokenizer ---

        # Check if a suitable separator token ID exists
        self.separator_token_id = self.tokenizer.eos_token_id # Default to EOS
        if self.separator_token_id is None:
             # Add fallback logic if EOS is also None, e.g., raise error or try CLS?
             # For now, let's assume EOS exists for ESM2. If not, the debug prints will show it.
             print("WARNING: EOS token ID is None. Check tokenizer configuration.")
             # Potentially raise ValueError("Could not find a suitable separator token ID (tried EOS).")

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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Initialize B-factor weight generator with more robust path handling
        bfactor_path = getattr(args, 'bfactor_csv_path', None)  # Get from args if provided
        self.bfactor_weights = BFactorWeightGenerator(
            bfactor_csv_path=bfactor_path,
            min_weight=0.5,
            max_weight=2.0
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.losses = ["ce"]
        
        # Save hyperparameters
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        """Save hyperparameters."""
        self.hparams = args

    def forward(self, batch_x):
        """Forward pass with B-factor weighted features applied to both embeddings and chemical features."""
        if isinstance(batch_x, dict) and 'x' in batch_x:
            batch_x = batch_x['x']
        
        # Get ESM embeddings
        combined_tokens = batch_x['combined_tokens']
        combined_mask = batch_x['combined_mask'].bool()
        batch_size = combined_tokens.shape[0]
        seq_len = combined_mask.shape[1] # Get sequence length
        device = combined_mask.device # Get device
        
        outputs = self.esm(
            input_ids=combined_tokens,
            attention_mask=combined_mask,
            output_hidden_states=True
        )
        sequence_output = outputs.last_hidden_state
        
        # Get receptor-specific weights for each item in the batch
        all_receptor_weights = []
        receptor_ids = batch_x['receptor_id'] # List of IDs
        for i in range(batch_size):
            single_receptor_id = receptor_ids[i]
            weights = self.bfactor_weights.get_weights(single_receptor_id, seq_len)
            all_receptor_weights.append(weights)
            
        # Stack weights into a single tensor [batch_size, seq_len]
        receptor_weights = torch.stack(all_receptor_weights).to(device)
        
        # Find the separator token position using the chosen token ID
        separator_token_id_to_use = self.separator_token_id # Use the ID determined in __init__
        if separator_token_id_to_use is None:
            raise ValueError("Separator token ID is None during forward pass. Check initialization.")
        
        # Ensure comparison is done correctly
        separator_mask = (combined_tokens == separator_token_id_to_use)
        if not isinstance(separator_mask, torch.Tensor):
             # This shouldn't happen if separator_token_id_to_use is an integer ID
             raise TypeError(f"Comparison resulted in type {type(separator_mask)}, expected torch.Tensor.")
             
        sep_positions = separator_mask.nonzero(as_tuple=True)[1]
        
        # Create a mask for receptor positions [batch_size, seq_len]
        receptor_mask = torch.zeros_like(combined_mask, dtype=torch.bool)
        for i in range(batch_size):
            # Ensure sep_positions has an entry for batch item i
            if i < len(sep_positions) and sep_positions[i] < seq_len - 1:
                receptor_mask[i, sep_positions[i]+1:] = True
            elif i >= len(sep_positions):
                 print(f"Warning: Separator token not found for batch item {i}. Receptor mask will be empty.")
                 # Or handle this case more robustly depending on your data/needs
            # else: sep_positions[i] is the last token, so no receptor part follows.

        # Apply weights to receptor portion of sequence embeddings
        weighted_sequence_output = sequence_output.clone()
        weighted_sequence_output[receptor_mask] = (
            sequence_output[receptor_mask] * 
            receptor_weights[receptor_mask].unsqueeze(-1) 
        )
        
        # Pool for context vector using weighted embeddings
        masked_output = weighted_sequence_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        pooled_output, _ = torch.max(masked_output, dim=1)
        
        # Apply weights to receptor chemical features
        chemical_features = []
        for feat_name in ['bulkiness', 'charge', 'hydrophobicity']:
            seq_feat = batch_x[f'seq_{feat_name}'] 
            rec_feat = batch_x[f'rec_{feat_name}'] 
            weighted_rec_feat = rec_feat * receptor_weights 
            weighted_rec_feat = torch.where(receptor_mask, weighted_rec_feat, rec_feat) 
            chemical_features.extend([seq_feat, weighted_rec_feat])
        
        chemical_features = torch.stack(chemical_features, dim=-1)
        
        # Apply FiLM conditioning with weighted sequence output
        conditioned_output = self.film(weighted_sequence_output, pooled_output, chemical_features)
        
        # Pool conditioned output
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
        # Use the same tokenizer instance as the model
        tokenizer = self.tokenizer 
        separator_token = tokenizer.eos_token # Use EOS token string for combining

        if separator_token is None:
            raise ValueError("EOS token is None in collate_fn. Check tokenizer configuration.")

        # Get sequences and labels
        sequences = [str(item['peptide_x']) for item in batch]
        receptors = [str(item['receptor_x']) for item in batch]
        labels = torch.tensor([item['y'] for item in batch])
        
        # Get receptor IDs for B-factor weights
        receptor_ids = [str(item.get('receptor_id', '')) for item in batch]
        
        # Combine sequences with the chosen separator token (e.g., EOS)
        combined = [f"{seq} {separator_token} {rec}" for seq, rec in zip(sequences, receptors)]
        
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
                feature_length = encoded['input_ids'].size(1) # Get length from tokenized output
                feature_list = []
                for item in batch:
                    if key in item:
                         # Ensure the feature tensor has the correct length (matching tokenized seq len)
                         # This might require padding or truncation if original features differ in length
                         item_feature = torch.tensor(item[key])
                         if len(item_feature) < feature_length:
                              padding = torch.zeros(feature_length - len(item_feature))
                              item_feature = torch.cat([item_feature, padding])
                         elif len(item_feature) > feature_length:
                              item_feature = item_feature[:feature_length]
                         feature_list.append(item_feature)
                    else:
                         feature_list.append(torch.zeros(feature_length))
                features[feat] = torch.stack(feature_list) # Stack features into [batch_size, seq_len]
            return features
        
        seq_features = process_features(batch, 'sequence')
        rec_features = process_features(batch, 'receptor')
        
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
                'receptor_id': receptor_ids,
            },
            'y': labels
        }

    def get_tokenizer(self):
        return self.tokenizer

    def batch_decode(self, batch):
        """Decode tokenized sequences."""
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