# 06_scripts_ml/models/random_forest_baseline.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class RandomForestBaselineModel:
    """
    Baseline model using Random Forest for peptide-receptor interaction prediction.
    Uses simple sequence encoding (amino acid composition) instead of deep learning embeddings.
    """
    def __init__(self, args=None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.losses = [] # Add an empty list for compatibility
        
    def parameters(self):
        """
        Placeholder for compatibility with PyTorch training loops expecting parameters.
        Scikit-learn models don't have parameters in the same way.
        """
        return iter([]) # Return an empty iterator

    def to(self, device):
        """
        Placeholder for compatibility with PyTorch training loops expecting device placement.
        Scikit-learn models run on CPU, so this is a no-op.
        """
        return self # Return the object itself

    def __call__(self, batch_x):
        """
        Make the model instance callable, like a PyTorch model.
        Delegates to the forward method.
        """
        return self.forward(batch_x)

    def named_parameters(self):
        """
        Placeholder for compatibility with PyTorch training loops expecting named parameters.
        Scikit-learn models don't have named parameters in the PyTorch sense.
        """
        return iter([]) # Return an empty iterator

    def eval(self):
        """
        Placeholder for compatibility with PyTorch training loops expecting an eval mode.
        Scikit-learn models don't have a separate eval mode like PyTorch models.
        """
        pass # No operation needed

    def _encode_sequence(self, sequence):
        """
        Simple sequence encoding using amino acid composition (frequency of each amino acid)
        """
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        counts = Counter(sequence)
        features = [counts.get(aa, 0) / len(sequence) for aa in amino_acids]
        return features
    
    def _prepare_features(self, batch):
        """Convert sequences to feature vectors"""
        if isinstance(batch, dict):  # Handle both dictionary and list inputs
            peptides = [seq for seq in batch['peptide_x']]
            receptors = [seq for seq in batch['receptor_x']]
        else:
            peptides = [example['peptide_x'] for example in batch]
            receptors = [example['receptor_x'] for example in batch]
            
        # Encode each sequence
        peptide_features = [self._encode_sequence(seq) for seq in peptides]
        receptor_features = [self._encode_sequence(seq) for seq in receptors]
        
        # Concatenate peptide and receptor features
        X = np.hstack([peptide_features, receptor_features])
        return X
    
    def forward(self, batch_x):
        """
        Forward pass (prediction) for compatibility with other models
        Handles cases where the model is not yet fitted by returning uniform probabilities.
        """
        X = self._prepare_features(batch_x)
        num_samples = X.shape[0]
        num_classes = 3 # Assuming 3 classes based on get_stats logic

        try:
            # Check if the model is fitted before predicting
            check_is_fitted(self.model)
            probas = self.model.predict_proba(X)
            return torch.tensor(probas, dtype=torch.float32)
        except NotFittedError:
            # Return uniform probabilities if the model is not fitted
            print("Warning: RandomForestBaselineModel is not fitted yet. Returning uniform probabilities.")
            uniform_probas = torch.ones(num_samples, num_classes, dtype=torch.float32) / num_classes
            return uniform_probas
    
    def fit(self, batch):
        """
        Train the random forest model
        """
        X = self._prepare_features(batch['x'])
        y = batch['y'].numpy()
        self.model.fit(X, y)
    
    def get_pr(self, logits):
        """Get predictions from logits (probabilities)"""
        return logits
    
    def get_stats(self, gt, pr, train=False):
        """
        Calculate various evaluation metrics for model performance
        Args:
            gt: Ground truth labels
            pr: Model predictions (probabilities)
            train: Boolean indicating if these are training or test metrics
        Returns:
            Dictionary containing various evaluation metrics
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
            
        except:
            # Handle cases where metrics cannot be calculated
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0
            
        return stats

    def collate_fn(self, batch):
        """
        Collate function for compatibility with data loaders
        """
        inputs = {}
        
        x_dict = {}
        x_dict['peptide_x'] = [example['peptide_x'] for example in batch]
        x_dict['receptor_x'] = [example['receptor_x'] for example in batch]
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        inputs['x'] = x_dict
        
        return inputs

    def batch_decode(self, batch):
        """
        Return the original sequences for interpretability
        """
        return [f"{peptide}:{receptor}" for peptide, receptor in 
               zip(batch['x']['peptide_x'], batch['x']['receptor_x'])]