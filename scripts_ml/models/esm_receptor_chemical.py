import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np

class FiLMWithAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FiLMWithAttention, self).__init__()
        # Original FiLM layers
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim * 2)
        
        # New bulkiness projection layers
        self.seq_bulkiness_proj = nn.Linear(1, feature_dim)
        self.rec_bulkiness_proj = nn.Linear(1, feature_dim)

    def forward(self, x, z, x_mask, z_mask, seq_bulkiness=None, rec_bulkiness=None):
        # Project bulkiness features
        if seq_bulkiness is not None and rec_bulkiness is not None:
            seq_bulk_feat = self.seq_bulkiness_proj(seq_bulkiness.unsqueeze(-1))
            rec_bulk_feat = self.rec_bulkiness_proj(rec_bulkiness.unsqueeze(-1))
            
            # Add bulkiness features to sequence and receptor embeddings
            x = x + seq_bulk_feat.unsqueeze(1).expand(-1, x.size(1), -1)
            z = z + rec_bulk_feat.unsqueeze(1).expand(-1, z.size(1), -1)

        # Original FiLM attention mechanism
        q = self.query_proj(x)
        k = self.key_proj(z)
        v = self.value_proj(z)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1)))
        
        # Apply mask to attention scores
        if x_mask is not None and z_mask is not None:
            mask = torch.matmul(x_mask.unsqueeze(-1).float(), z_mask.unsqueeze(1).float())
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Project to output dimension
        output = self.out_proj(context)
        
        # Split into scale and bias
        scale, bias = output.chunk(2, dim=-1)
        
        # Apply FiLM transformation
        transformed = (x * (scale + 1)) + bias
        
        # Global max pooling
        pooled = torch.max(transformed, dim=1)[0]
        
        return pooled

class ESMReceptorChemical(nn.Module):
    def __init__(self, args, num_classes=3):  # args is now the first parameter
        super(ESMReceptorChemical, self).__init__()
        # Use the backbone parameter from args
        self.esm = AutoModel.from_pretrained(args.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        
        self.film = FiLMWithAttention(self.esm.config.hidden_size)
        self.classifier = nn.Linear(self.esm.config.hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        #self.esm = AutoModel.from_pretrained(model_name)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        #self.film = FiLMWithAttention(self.esm.config.hidden_size)
        #self.classifier = nn.Linear(self.esm.config.hidden_size * 2, num_classes)
        #self.classifier = nn.Linear(self.esm.config.hidden_size, num_classes)  # Changed dimension
        #self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, sequence_ids, sequence_mask, receptor_ids, receptor_mask, 
                seq_bulkiness=None, rec_bulkiness=None):
        sequence_output = self.esm(input_ids=sequence_ids, attention_mask=sequence_mask).last_hidden_state
        receptor_output = self.esm(input_ids=receptor_ids, attention_mask=receptor_mask).last_hidden_state
        
        # Apply FiLM with bulkiness features
        combined = self.film(sequence_output, receptor_output, sequence_mask, receptor_mask,
                           seq_bulkiness, rec_bulkiness)
        
        logits = self.classifier(combined)
        return logits

    def training_step(self, batch, batch_idx):
        sequence_ids = batch["sequence_ids"]
        sequence_mask = batch["sequence_attention_mask"]
        receptor_ids = batch["receptor_ids"]
        receptor_mask = batch["receptor_attention_mask"]
        seq_bulkiness = batch["sequence_bulkiness"]
        rec_bulkiness = batch["receptor_bulkiness"]
        labels = batch["labels"]

        logits = self(sequence_ids, sequence_mask, receptor_ids, receptor_mask,
                     seq_bulkiness, rec_bulkiness)
        
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
        """Get evaluation statistics"""
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