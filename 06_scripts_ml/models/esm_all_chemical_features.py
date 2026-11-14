import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

# Per-residue features written by 05_chemical_conversion.R ("all" option).
# Order matters: it matches the stacking order in forward().
CHEM_FEATURES = ('bulkiness', 'charge', 'hydrophobicity')
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")  # residues the R lookup tables cover


def _clean_seq(seq):
    return "".join(str(seq).split())


def _parse_r_cell(cell, seq):
    """
    One comma-separated cell from 05_chemical_conversion.R -> one float per residue of `seq`.
    Residues outside the R tables are written as "NULL" (or dropped) by R and become 0 here.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        tokens = []
    else:
        tokens = [t.strip() for t in str(cell).split(",") if t.strip() != ""]

    def to_float(t):
        return 0.0 if t in ("NULL", "NA", "NaN", "nan") else float(t)

    if len(tokens) == len(seq):
        return [to_float(t) for t in tokens]
    if len(tokens) == sum(aa in _STANDARD_AA for aa in seq):   # non-standard residues were dropped
        it = iter(tokens)
        return [to_float(next(it)) if aa in _STANDARD_AA else 0.0 for aa in seq]
    raise ValueError(f"{len(tokens)} chemistry values for a {len(seq)}-residue sequence; "
                     "re-run 05_chemical_conversion.R on this CSV.")


def load_chemistry_lookup(csv_paths):
    """
    Read the chemistry columns from the training/test CSVs written by 05_chemical_conversion.R.
    Returns {(peptide, receptor): {'sequence_<feat>': [...], 'receptor_<feat>': [...]}}.
    """
    lookup = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        cols = [f"{p}_{f.capitalize()}" for p in ("Sequence", "Receptor") for f in CHEM_FEATURES]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing chemistry columns {missing}. Use the "
                             "train_data_with_all_* / test_data_with_all_* files from 05_chemical_conversion.R.")
        for _, row in df.iterrows():
            pep, rec = _clean_seq(row["Sequence"]), _clean_seq(row["Receptor Sequence"])
            lookup[(pep, rec)] = {
                **{f"sequence_{f}": _parse_r_cell(row[f"Sequence_{f.capitalize()}"], pep) for f in CHEM_FEATURES},
                **{f"receptor_{f}": _parse_r_cell(row[f"Receptor_{f.capitalize()}"], rec) for f in CHEM_FEATURES},
            }
    return lookup

from models.esm_utils import (
    DEFAULT_ESM_NAME,
    DEFAULT_UNFREEZE_LAYERS,
    apply_esm_unfreeze,
    resolve_esm_name,
    resolve_unfreeze_layers,
)

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
        
        # Backbone size and unfreeze depth are configurable; must match the
        # checkpoint used at load time. Default: esm2_t30_150M, last encoder layer.
        esm_name = resolve_esm_name(args, default=DEFAULT_ESM_NAME)
        n_unfreeze = resolve_unfreeze_layers(args, default=DEFAULT_UNFREEZE_LAYERS)
        self.esm_model_name = esm_name
        self.unfreeze_layers = n_unfreeze
        self.esm = AutoModel.from_pretrained(esm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_name)
        print(f"Loaded ESM backbone: {esm_name}")
        apply_esm_unfreeze(self.esm, n_unfreeze)
        
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

        # Per-residue chemistry, read from the same CSVs 06_main_train.py loads
        csv_paths = [p for p in (getattr(args, "train_csv", None),
                                 getattr(args, "test_csv", None),
                                 getattr(args, "eval_only_data_path", None)) if p]
        data_dir = getattr(args, "data_dir", None)
        if data_dir:
            csv_paths += [f"{data_dir}/train_immuno_stratify.csv", f"{data_dir}/test_immuno_stratify.csv"]
        csv_paths = [p for p in dict.fromkeys(csv_paths) if os.path.exists(p)]
        if not csv_paths:
            raise ValueError("No CSV found to read chemistry from (--train_csv / --test_csv / --eval_only_data_path).")
        self.chem_lookup = load_chemistry_lookup(csv_paths)
        print(f"Loaded per-residue chemistry for {len(self.chem_lookup)} peptide-receptor pairs from {csv_paths}")

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
        """
        Collate function for batching.

        Token layout per row:  <cls> peptide <eos> receptor <eos> <pad>...
        Built from per-residue ids so the position of every residue is known, which lets the
        per-residue chemistry from 05_chemical_conversion.R land on the matching token:
          - seq_* (peptide channels) hold the peptide residue's value at peptide tokens
          - rec_* (receptor channels) hold the receptor residue's value at receptor tokens
          - every other position is 0
        Pairs longer than 1024 tokens are truncated from the receptor's C-terminal end.
        """
        tokenizer = self.tokenizer
        cls_id, eos_id, pad_id = tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        max_len = 1024

        rows = []
        for item in batch:
            pep = "".join(str(item['peptide_x']).split())
            rec = "".join(str(item['receptor_x']).split())
            pep = pep[:max_len - 4]                       # room for 3 special tokens + 1 receptor residue
            rec = rec[:max_len - 3 - len(pep)]
            # ESM2 has one token per residue letter; unknown letters map to <unk>
            ids = ([cls_id] + tokenizer.convert_tokens_to_ids(list(pep)) + [eos_id]
                   + tokenizer.convert_tokens_to_ids(list(rec)) + [eos_id])
            rows.append((pep, rec, ids))

        B = len(batch)
        L = max(len(ids) for _, _, ids in rows)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        feats = {f"{p}_{f}": torch.zeros((B, L), dtype=torch.float32)
                 for p in ('seq', 'rec') for f in CHEM_FEATURES}

        for i, (item, (pep, rec, ids)) in enumerate(zip(batch, rows)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids)
            attention_mask[i, :n] = 1
            pep_start = 1                      # after <cls>
            rec_start = 1 + len(pep) + 1       # after <cls>, peptide, <eos>
            key = (_clean_seq(item['peptide_x']), _clean_seq(item['receptor_x']))
            if key not in self.chem_lookup:
                raise KeyError(f"No chemistry row for peptide {key[0][:20]}... in the loaded CSVs.")
            chem = self.chem_lookup[key]
            for feat in CHEM_FEATURES:
                pep_vals = chem[f"sequence_{feat}"][:len(pep)]
                rec_vals = chem[f"receptor_{feat}"][:len(rec)]
                feats[f"seq_{feat}"][i, pep_start:pep_start + len(pep_vals)] = torch.tensor(pep_vals, dtype=torch.float32)
                feats[f"rec_{feat}"][i, rec_start:rec_start + len(rec_vals)] = torch.tensor(rec_vals, dtype=torch.float32)

        labels = torch.tensor([item['y'] for item in batch], dtype=torch.long)
        return {
            'x': {
                'combined_tokens': input_ids,
                'combined_mask': attention_mask,
                **feats,
            },
            'y': labels
        }

    def get_tokenizer(self):
        return self.tokenizer

    def batch_decode(self, batch):
        """Decode tokenized pairs to 'peptide:receptor' strings."""
        tokens = batch['x']['combined_tokens'] if isinstance(batch, dict) and 'x' in batch else batch['combined_tokens']
        eos_id = self.tokenizer.eos_token_id
        out = []
        for ids in tokens.tolist():
            sep = ids.index(eos_id, 1)                       # <eos> between peptide and receptor
            end = ids.index(eos_id, sep + 1)                 # final <eos>
            pep = "".join(self.tokenizer.convert_ids_to_tokens(ids[1:sep]))
            rec = "".join(self.tokenizer.convert_ids_to_tokens(ids[sep + 1:end]))
            out.append(f"{pep}:{rec}")
        return out

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