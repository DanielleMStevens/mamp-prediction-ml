"""
A regression model based on Facebook's ESM-2 for protein sequence analysis.
This model adapts the ESM architecture for predicting continuous values instead of classes.

The model:
1. Uses a frozen ESM-2 model as the base encoder
2. Takes the CLS token embedding for sequence representation
3. Passes through a 3-layer MLP to predict a single continuous value

This implementation is useful for predicting continuous protein properties
(e.g., binding affinity, stability scores, etc.) rather than discrete classes.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ESMRegressionModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        E = self.esm_model.config.hidden_size

        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 1),
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  

    def forward(self, batch_x):
        embeddings = self.esm_model(**batch_x).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        return self.net(embeddings) # (B, S)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = self.tokenizer([example['x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
 
        return inputs

    def batch_decode(self, batch):
        return self.tokenizer.batch_decode(batch['x']['input_ids'], skip_special_tokens=True)