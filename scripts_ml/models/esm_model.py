"""
A model that uses Facebook's ESM-2 (Evolutionary Scale Modeling) for protein sequence analysis.
This model leverages the pre-trained ESM2 150M parameter model for protein understanding.

The model:
1. Uses a frozen ESM-2 model as the base encoder
2. Takes the CLS token embedding for sequence representation
3. Passes through a 3-layer MLP to predict 3 classes

ESM-2 is a state-of-the-art protein language model trained on evolutionary data.
This implementation keeps the base ESM model frozen and only trains the classification head.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ESMModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # Load pre-trained ESM-2 model and freeze its parameters
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        # Freeze the base model parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        E = self.esm_model.config.hidden_size  # Get embedding dimension from model config

        # Classification head: 3-layer MLP with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),
        )
        # Initialize tokenizer for protein sequences
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