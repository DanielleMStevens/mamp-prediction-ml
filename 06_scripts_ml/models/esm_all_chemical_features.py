import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class FiLMWithConcatenation(nn.Module):
    """
    FiLM layer that conditions the sequence representation x
    based on feature vector z and concatenated sequence/receptor bulkiness.
    """
    def __init__(self, feature_dim):
        super().__init__()
        # Adjusted input dimension: feature_dim + 3 (seq features) + 3 (rec features)
        self.film_layer = nn.Linear(feature_dim + 6, 2 * feature_dim) # Gamma and Beta

    def forward(self, x, z, x_mask, z_mask, seq_features=None, rec_features=None):
        """
        Args:
            x: Sequence representation (batch_size, seq_len, feature_dim)
            z: Feature vector (batch_size, feature_dim) - Can be from ESM contact/pooler or other sources
            x_mask: Mask for sequence x (batch_size, seq_len)
            z_mask: Mask for feature vector z (batch_size, 1) - Indicates if z is valid
            seq_features: Combined sequence chemical features (batch_size, seq_len, 3)
            rec_features: Combined receptor chemical features (batch_size, seq_len, 3)

        Returns:
            Conditioned sequence representation.
        """
        batch_size, seq_len, feature_dim = x.shape

        # Expand z to match sequence length if needed
        if z.dim() == 2:
            # Assuming z is (batch_size, feature_dim), expand to (batch_size, seq_len, feature_dim)
            z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        elif z.dim() == 3 and z.shape[1] == 1:
             # Assuming z is (batch_size, 1, feature_dim), expand
            z_expanded = z.expand(-1, seq_len, -1)
        elif z.dim() == 3 and z.shape[1] == seq_len:
            # Assuming z is already (batch_size, seq_len, feature_dim)
            z_expanded = z
        else:
            raise ValueError(f"Unexpected shape for z: {z.shape}")

        # Prepare feature masks if masks are provided
        # z_mask is typically (batch_size,) or (batch_size, 1)
        z_mask_expanded = z_mask.unsqueeze(-1).expand(-1, seq_len, 1) if z_mask is not None else torch.ones(batch_size, seq_len, 1, device=x.device)

        # Apply masks to features before concatenation
        z_expanded = z_expanded * z_mask_expanded

        # Handle optional chemical features
        feature_components = [z_expanded]
        if seq_features is not None:
             # seq_features should be (batch_size, seq_len, 3)
             assert seq_features.dim() == 3 and seq_features.shape[-1] == 3, f"Expected seq_features shape (batch, seq_len, 3), got {seq_features.shape}"
             feature_components.append(seq_features)
        if rec_features is not None:
             # rec_features should be (batch_size, seq_len, 3)
             assert rec_features.dim() == 3 and rec_features.shape[-1] == 3, f"Expected rec_features shape (batch, seq_len, 3), got {rec_features.shape}"
             feature_components.append(rec_features)

        # Concatenate z (expanded) and potentially bulkiness features
        combined_features = torch.cat(feature_components, dim=-1)

        # Generate gamma and beta from combined features
        # The input dimension to film_layer is now feature_dim + number of concatenated features
        gamma_beta = self.film_layer(combined_features)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Apply FiLM conditioning: output = gamma * x + beta
        output = gamma * x + beta

        # Apply mask to the output
        if x_mask is not None:
            output = output * x_mask.unsqueeze(-1)
        
        return output

class ESMallChemicalFeatures(nn.Module):
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
        super(ESMallChemicalFeatures, self).__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

        # Correct way to get embedding dimension from HuggingFace AutoModel
        self.esm_feature_dim = self.esm.config.hidden_size
        # Note: Accessing 'contact_head' might need adjustment depending on how AutoModel loads it.
        # For now, let's assume it's not directly needed or handled differently. If contact predictions are crucial,
        # we might need to load the ESMForProteinInteraction model or check the output structure.
        # self.contact_head = self.esm.contact_head # This attribute might not exist directly

        # More selective layer freezing strategy (from snippet)
        # Freeze only the embedding and first 20 layers
        modules_to_freeze = [
            self.esm.embeddings,
            *self.esm.encoder.layer[:20] # Freeze the first 20 layers
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        # Define the FiLM layer (renamed from film_cond to film)
        film_conditioning_dim = self.esm_feature_dim
        self.film = FiLMWithConcatenation(film_conditioning_dim)

        # Final classification head
        self.classifier = nn.Linear(self.esm_feature_dim, num_classes) # Use the correct feature dim

        self.loss = nn.CrossEntropyLoss() # Or BCEWithLogitsLoss for multi-label

        # Example hyperparameter saving - Check if 'save_hyperparameters' is available
        # self.save_hyperparameters(args) # This is common in PyTorch Lightning, might not be standard nn.Module

    def forward(self, batch_x):
        """
        Process batch data through the model.
        batch_x is expected to be a dictionary from collate_fn.
        """
        # seq_tokens = batch_x['seq_tokens'] # (batch_size, seq_len)
        # rec_tokens = batch_x['rec_tokens'] # (batch_size, rec_len)
        # combined_tokens = batch_x['combined_tokens'] # (batch_size, total_len)
        # seq_mask = batch_x['seq_mask'] # (batch_size, seq_len)
        # rec_mask = batch_x['rec_mask'] # (batch_size, rec_len)
        # combined_mask = batch_x['combined_mask'] # (batch_size, total_len)
        # seq_bulkiness = batch_x['seq_bulkiness'] # (batch_size, seq_len)
        # rec_bulkiness = batch_x['rec_bulkiness'] # (batch_size, rec_len)

        combined_tokens = batch_x['combined_tokens']
        combined_mask = batch_x['combined_mask']
        seq_len = batch_x['seq_tokens'].shape[1] # Original sequence length before padding/combining

        # Extract all chemical features - expected shape [batch, len, 1] from collate_fn
        seq_bulkiness = batch_x['seq_bulkiness']
        seq_charge = batch_x['seq_charge']
        seq_hydrophobicity = batch_x['seq_hydrophobicity']
        rec_bulkiness = batch_x['rec_bulkiness']
        rec_charge = batch_x['rec_charge']
        rec_hydrophobicity = batch_x['rec_hydrophobicity']

        # Combine features: [batch, len, 1] -> [batch, len, 3]
        seq_features = torch.stack([seq_bulkiness, seq_charge, seq_hydrophobicity], dim=-1).float()
        rec_features = torch.stack([rec_bulkiness, rec_charge, rec_hydrophobicity], dim=-1).float()

        # Pad features to match combined_tokens length if necessary
        total_len = combined_tokens.shape[1]
        # Assuming collate_fn pads them correctly to combined length - checking logic
        # Pad seq_features (originally seq_len) to total_len
        padded_seq_features = F.pad(seq_features, (0, 0, 0, total_len - seq_features.shape[1]), "constant", 0)
        # Pad rec_features (originally rec_len) to total_len
        # Receptor starts after sequence separator
        rec_start_pos = seq_len + 1 # Position after separator
        padded_rec_features = F.pad(rec_features, (0, 0, rec_start_pos, total_len - rec_start_pos - rec_features.shape[1] ), "constant", 0) # Adjust padding start


        # ESM processing - Use self.esm
        # Request hidden states; contacts might not be a direct output of AutoModel by default
        # Check the documentation for the specific ESM model on HuggingFace Hub for output structure
        outputs = self.esm(input_ids=combined_tokens, attention_mask=combined_mask, output_hidden_states=True)
        # Typically the last hidden state is used as the representation
        esm_repr = outputs.last_hidden_state # (batch_size, total_len, embed_dim)
        # esm_contacts = outputs.contacts # Check if 'contacts' are available in 'outputs' object


        # Derive conditioning vector 'z' using MAX pooling over combined sequence representation
        masked_repr = esm_repr.masked_fill(~combined_mask.unsqueeze(-1).bool(), -torch.inf)
        z, _ = torch.max(masked_repr, dim=1) # (batch_size, embed_dim)

        # Apply FiLM conditioning
        # The FiLM layer needs the conditioning vector z, potentially its mask (if applicable),
        # and the sequence/receptor features aligned with esm_repr.
        # Ensure z_mask matches z's batch dimension if needed. Here, z is (batch_size, embed_dim), so mask isn't seq-dependent.
        conditioned_repr = self.film(esm_repr, z, combined_mask, None, seq_features=padded_seq_features, rec_features=padded_rec_features)

        # Pool output for classification using MAX pooling
        masked_conditioned_output = conditioned_repr.masked_fill(~combined_mask.unsqueeze(-1).bool(), -torch.inf)
        pooled_output, _ = torch.max(masked_conditioned_output, dim=1) # (batch_size, embed_dim)

        # Classifier
        logits = self.classifier(pooled_output) # (batch_size, num_classes)
        
        return logits

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        Args:
            batch (dict): Batch of data containing sequences, receptors, and labels.
                          Assumes batch is the direct output of collate_fn.
            batch_idx (int): Index of the current batch
        """
        # Assuming 'batch' is the dict from collate_fn, pass it directly
        logits = self(batch)
        labels = batch['labels'] # Extract labels from the batch dict

        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=logits.device, requires_grad=True) # Ensure tensor is on correct device
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)

        loss = self.loss(logits, labels) + l2_lambda * l2_reg # Use extracted labels
        return loss

    def get_tokenizer(self):
        """Return the model's tokenizer"""
        return self.tokenizer

    def collate_fn(self, batch):
        """
        Collates data samples into batches.
        Handles tokenization, padding, and feature extraction/padding.
        """
        sequences = [item['sequence'] for item in batch]
        receptors = [item['receptor'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        # Extract and parse chemical features - Assuming they are comma-separated strings
        def parse_feature(feature_str):
            # return list(map(float, feature_str.split(','))) if feature_str else [] # Handle empty strings if necessary
            # Handle potential NaN or empty values robustly
            values = []
            if feature_str and isinstance(feature_str, str):
                for x in feature_str.split(','):
                    try:
                        values.append(float(x))
                    except ValueError:
                        values.append(0.0) # Replace NaN/invalid with 0 or another strategy
            return values

        # seq_bulkiness_list = [parse_feature(item.get('seq_bulkiness', '')) for item in batch]
        # rec_bulkiness_list = [parse_feature(item.get('rec_bulkiness', '')) for item in batch]
        # NEW features
        seq_bulkiness_list = [parse_feature(item.get('seq_bulkiness', '')) for item in batch]
        seq_charge_list = [parse_feature(item.get('seq_charge', '')) for item in batch]
        seq_hydrophobicity_list = [parse_feature(item.get('seq_hydrophobicity', '')) for item in batch]
        rec_bulkiness_list = [parse_feature(item.get('rec_bulkiness', '')) for item in batch]
        rec_charge_list = [parse_feature(item.get('rec_charge', '')) for item in batch]
        rec_hydrophobicity_list = [parse_feature(item.get('rec_hydrophobicity', '')) for item in batch]

        # Tokenize sequences and receptors
        tokenizer = self.get_tokenizer()
        seq_tokens_list = [torch.tensor(tokenizer.encode(seq)[1:-1]) for seq in sequences] # Remove BOS/EOS
        rec_tokens_list = [torch.tensor(tokenizer.encode(rec)[1:-1]) for rec in receptors]

        # Pad sequences and receptors individually first to get masks
        seq_tokens_padded = pad_sequence(seq_tokens_list, batch_first=True, padding_value=tokenizer.padding_idx)
        rec_tokens_padded = pad_sequence(rec_tokens_list, batch_first=True, padding_value=tokenizer.padding_idx)
        seq_mask = (seq_tokens_padded != tokenizer.padding_idx)
        rec_mask = (rec_tokens_padded != tokenizer.padding_idx)

        # Combine tokens: <cls> seq <sep> rec <eos> (or another strategy)
        # Let's use: seq <sep> rec (no special tokens added here, handled by tokenizer if needed)
        combined_tokens_list = []
        sep_token_id = tokenizer.sep_token_id # Or another separator if desired
        for seq_tok, rec_tok in zip(seq_tokens_list, rec_tokens_list):
            combined = torch.cat([seq_tok, torch.tensor([sep_token_id]), rec_tok], dim=0)
            combined_tokens_list.append(combined)

        combined_tokens_padded = pad_sequence(combined_tokens_list, batch_first=True, padding_value=tokenizer.padding_idx)
        combined_mask = (combined_tokens_padded != tokenizer.padding_idx)

        # Pad chemical features
        max_seq_len = seq_mask.shape[1]
        max_rec_len = rec_mask.shape[1]
        max_combined_len = combined_mask.shape[1]

        # Pad individual features first (to max_seq_len or max_rec_len)
        def pad_feature_list(feature_list, max_len):
            padded = []
            for feat in feature_list:
                # feat_tensor = torch.tensor(feat[:max_len], dtype=torch.float).unsqueeze(-1) # Add feature dim
                # Ensure correct length before padding
                feat = feat[:max_len]
                feat_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(-1)
                padding_size = max_len - len(feat)
                if padding_size > 0:
                    padded_feat = F.pad(feat_tensor, (0, 0, 0, padding_size), "constant", 0.0) # Pad sequence dim
                else:
                    padded_feat = feat_tensor
                padded.append(padded_feat)
            return torch.stack(padded, dim=0) # (batch_size, max_len, 1)

        # Pad features to their respective max lengths first
        seq_bulk_padded = pad_feature_list(seq_bulkiness_list, max_seq_len)
        seq_charge_padded = pad_feature_list(seq_charge_list, max_seq_len)
        seq_hydro_padded = pad_feature_list(seq_hydrophobicity_list, max_seq_len)
        rec_bulk_padded = pad_feature_list(rec_bulkiness_list, max_rec_len)
        rec_charge_padded = pad_feature_list(rec_charge_list, max_rec_len)
        rec_hydro_padded = pad_feature_list(rec_hydrophobicity_list, max_rec_len)

        # Pad features further to match max_combined_len for use in forward pass
        # Padded for seq features (apply to first part of combined)
        seq_bulk_combined = F.pad(seq_bulk_padded, (0, 0, 0, max_combined_len - max_seq_len), "constant", 0)
        seq_charge_combined = F.pad(seq_charge_padded, (0, 0, 0, max_combined_len - max_seq_len), "constant", 0)
        seq_hydro_combined = F.pad(seq_hydro_padded, (0, 0, 0, max_combined_len - max_seq_len), "constant", 0)

        # Padded for rec features (apply after seq part and separator)
        # Need to account for the separator token position
        sep_pos = max_seq_len # Separator is right after max_seq_len
        rec_start_pos = sep_pos + 1
        rec_end_pos = rec_start_pos + max_rec_len
        # Pad receptor features to place them correctly in the combined length
        rec_bulk_combined = F.pad(rec_bulk_padded, (0, 0, rec_start_pos, max_combined_len - rec_end_pos), "constant", 0)
        rec_charge_combined = F.pad(rec_charge_padded, (0, 0, rec_start_pos, max_combined_len - rec_end_pos), "constant", 0)
        rec_hydro_combined = F.pad(rec_hydro_padded, (0, 0, rec_start_pos, max_combined_len - rec_end_pos), "constant", 0)

        batch_dict = {
            'seq_tokens': seq_tokens_padded,
            'rec_tokens': rec_tokens_padded,
            'combined_tokens': combined_tokens_padded,
            'seq_mask': seq_mask,
            'rec_mask': rec_mask,
            'combined_mask': combined_mask,
            # Pass the features padded to combined length
            'seq_bulkiness': seq_bulk_combined,
            'seq_charge': seq_charge_combined,
            'seq_hydrophobicity': seq_hydro_combined,
            'rec_bulkiness': rec_bulk_combined,
            'rec_charge': rec_charge_combined,
            'rec_hydrophobicity': rec_hydro_combined,
            'labels': labels
        }
        return batch_dict

    def batch_decode(self, batch):
        """
        Decode a batch of tokenized sequences back to text.
        Args:
            batch (dict): Batch containing tokenized sequences (output from collate_fn)
        Returns:
            list: Decoded sequences in format "peptide:receptor"
        """
        # Access tokens directly from the batch dict
        peptide_decoded_ls = self.get_tokenizer().batch_decode(batch['seq_tokens'], skip_special_tokens=True)
        receptor_decoded_ls = self.get_tokenizer().batch_decode(batch['rec_tokens'], skip_special_tokens=True)
        
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

    def save_hyperparameters(self, args):
         # Simple placeholder if not using PyTorch Lightning
         self.hparams = args
         print("Note: 'save_hyperparameters' called. Ensure compatibility if not using PyTorch Lightning.")