import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np

class SimpleConcatenation(nn.Module):
    """Simple concatenation of features followed by MLP"""
    def __init__(self, feature_dim):
        super().__init__()
        self.seq_proj = nn.Linear(1, feature_dim)
        self.rec_proj = nn.Linear(1, feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, x, z, seq_bulkiness, rec_bulkiness):
        seq_bulk = self.seq_proj(seq_bulkiness.unsqueeze(-1))
        rec_bulk = self.rec_proj(rec_bulkiness.unsqueeze(-1))
        combined = torch.cat([x, z, seq_bulk, rec_bulk], dim=-1)
        return self.fusion(combined)

class CrossAttention(nn.Module):
    """Cross-attention between sequence and receptor features"""
    def __init__(self, feature_dim):
        super().__init__()
        self.seq_proj = nn.Linear(1, feature_dim)
        self.rec_proj = nn.Linear(1, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x, z, seq_bulkiness, rec_bulkiness):
        seq_bulk = self.seq_proj(seq_bulkiness.unsqueeze(-1))
        rec_bulk = self.rec_proj(rec_bulkiness.unsqueeze(-1))
        
        # Cross attention between bulkiness features
        attn_out, _ = self.attention(seq_bulk.unsqueeze(1), 
                                   rec_bulk.unsqueeze(1), 
                                   rec_bulk.unsqueeze(1))
        
        combined = torch.cat([attn_out.squeeze(1), seq_bulk], dim=-1)
        return self.fusion(combined)

class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning"""
    def __init__(self, feature_dim):
        super().__init__()
        self.seq_gamma = nn.Linear(1, feature_dim)
        self.seq_beta = nn.Linear(1, feature_dim)
        self.rec_gamma = nn.Linear(1, feature_dim)
        self.rec_beta = nn.Linear(1, feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x, z, seq_bulkiness, rec_bulkiness):
        # Generate FiLM parameters
        seq_g = self.seq_gamma(seq_bulkiness.unsqueeze(-1))
        seq_b = self.seq_beta(seq_bulkiness.unsqueeze(-1))
        rec_g = self.rec_gamma(rec_bulkiness.unsqueeze(-1))
        rec_b = self.rec_beta(rec_bulkiness.unsqueeze(-1))
        
        # Apply FiLM conditioning
        x_film = (x * seq_g) + seq_b
        z_film = (z * rec_g) + rec_b
        
        combined = torch.cat([x_film, z_film], dim=-1)
        return self.fusion(combined)

class MLPMixer(nn.Module):
    """MLP Mixer-based fusion"""
    def __init__(self, feature_dim):
        super().__init__()
        self.seq_proj = nn.Linear(1, feature_dim)
        self.rec_proj = nn.Linear(1, feature_dim)
        
        # Channel mixing
        self.channel_mixing = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 8),
            nn.GELU(),
            nn.Linear(feature_dim * 8, feature_dim)
        )
        
        # Feature mixing
        self.feature_mixing = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, x, z, seq_bulkiness, rec_bulkiness):
        seq_bulk = self.seq_proj(seq_bulkiness.unsqueeze(-1))
        rec_bulk = self.rec_proj(rec_bulkiness.unsqueeze(-1))
        
        # Concatenate all features
        combined = torch.cat([x, z, seq_bulk, rec_bulk], dim=-1)
        
        # Apply channel mixing
        channel_mixed = self.channel_mixing(combined)
        
        # Apply feature mixing
        feature_mixed = self.feature_mixing(channel_mixed)
        
        return feature_mixed

class GatedFusion(nn.Module):
    """Gated fusion mechanism"""
    def __init__(self, feature_dim):
        super().__init__()
        self.seq_proj = nn.Linear(1, feature_dim)
        self.rec_proj = nn.Linear(1, feature_dim)
        
        # Gates
        self.seq_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.rec_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x, z, seq_bulkiness, rec_bulkiness):
        seq_bulk = self.seq_proj(seq_bulkiness.unsqueeze(-1))
        rec_bulk = self.rec_proj(rec_bulkiness.unsqueeze(-1))
        
        # Compute gates
        seq_gate_val = self.seq_gate(torch.cat([x, seq_bulk], dim=-1))
        rec_gate_val = self.rec_gate(torch.cat([z, rec_bulk], dim=-1))
        
        # Apply gated fusion
        x_gated = x * seq_gate_val
        z_gated = z * rec_gate_val
        
        combined = torch.cat([x_gated, z_gated], dim=-1)
        return self.fusion(combined)

class ESMReceptorChemicalFusion(nn.Module):
    """
    Enhanced ESM-based model with multiple fusion architecture options for
    combining sequence and receptor features with bulkiness information.
    """
    def __init__(self, args, num_classes=3, fusion_type='simple_concat'):
        super().__init__()
        
        # Load ESM model
        self.esm = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        
        # Freeze embedding and first 20 layers
        modules_to_freeze = [
            self.esm.embeddings,
            *self.esm.encoder.layer[:20]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        feature_dim = self.esm.config.hidden_size
        
        # Initialize fusion module based on type
        fusion_modules = {
            'simple_concat': SimpleConcatenation,
            'cross_attention': CrossAttention,
            'film': FiLMConditioning,
            'mlp_mixer': MLPMixer,
            'gated': GatedFusion
        }
        
        if fusion_type not in fusion_modules:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        self.fusion_module = fusion_modules[fusion_type](feature_dim)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.losses = ["ce"]

    def forward(self, batch_x):
        # Get ESM embeddings
        with torch.set_grad_enabled(True):
            sequence_output = self.esm(**batch_x['peptide_x']).last_hidden_state
            receptor_output = self.esm(**batch_x['receptor_x']).last_hidden_state
        
        # Process bulkiness features
        seq_bulkiness = batch_x.get('sequence_bulkiness')
        rec_bulkiness = batch_x.get('receptor_bulkiness')
        
        if seq_bulkiness is not None:
            seq_bulkiness = seq_bulkiness.float()
            seq_bulkiness = torch.nan_to_num(seq_bulkiness, nan=0.0)
            seq_bulkiness = (seq_bulkiness - seq_bulkiness.mean()) / (seq_bulkiness.std() + 1e-8)
            
        if rec_bulkiness is not None:
            rec_bulkiness = rec_bulkiness.float()
            rec_bulkiness = torch.nan_to_num(rec_bulkiness, nan=0.0)
            rec_bulkiness = (rec_bulkiness - rec_bulkiness.mean()) / (rec_bulkiness.std() + 1e-8)
        
        # Get sequence representations
        x_pool = torch.mean(sequence_output, dim=1)
        z_pool = torch.mean(receptor_output, dim=1)
        
        # Apply fusion
        combined = self.fusion_module(x_pool, z_pool, seq_bulkiness, rec_bulkiness)
        
        # Get logits
        logits = self.classifier(combined)
        
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['x'])
        
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)
        
        loss = self.criterion(logits, batch['y']) + l2_lambda * l2_reg
        return loss

    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}
        x_dict = {}

        x_dict['peptide_x'] = self.tokenizer([example['peptide_x'] for example in batch],
                return_tensors='pt', padding=True)
        x_dict['receptor_x'] = self.tokenizer([example['receptor_x'] for example in batch],
                return_tensors='pt', padding=True)
        
        if 'sequence_bulkiness' in batch[0]:
            x_dict['sequence_bulkiness'] = torch.tensor([example['sequence_bulkiness'] for example in batch])
        if 'receptor_bulkiness' in batch[0]:
            x_dict['receptor_bulkiness'] = torch.tensor([example['receptor_bulkiness'] for example in batch])
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        inputs['x'] = x_dict
 
        return inputs

    def batch_decode(self, batch):
        peptide_decoded_ls = self.tokenizer.batch_decode(batch['x']['peptide_x']['input_ids'], skip_special_tokens=True)
        receptor_decoded_ls = self.tokenizer.batch_decode(batch['x']['receptor_x']['input_ids'], skip_special_tokens=True)
        return [f"{peptide}:{receptor}" for peptide, receptor in zip(peptide_decoded_ls, receptor_decoded_ls)]

    def get_pr(self, logits):
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
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