"""
A model that uses the gLM2 (Generative Language Model for Proteins) for protein sequence analysis.
This model leverages the pre-trained gLM2 650M parameter model from TattaBio for protein sequence understanding.

The model:
1. Uses a frozen gLM2 model as the base encoder
2. Processes protein sequences with special '<+>' tokens
3. Takes the first token's embedding (CLS token) for classification
4. Passes through a 3-layer MLP to predict 3 classes

Note: The model keeps the base gLM2 frozen and only trains the classification head.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class GLMModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # Load pre-trained gLM2 model and freeze its parameters
        self.model = AutoModel.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)
        # self.model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True)
        # Freeze the base model parameters
        for param in self.model.parameters():
            param.requires_grad = False     

        E = self.model.config.dim  # Get embedding dimension from model config

        # Classification head: 3-layer MLP with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),
        )
        # Initialize tokenizer for protein sequences
        self.tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M", trust_remote_code=True)  

    def forward(self, batch_x):
        embeddings = self.model(batch_x.input_ids, attention_mask=batch_x.attention_mask, output_hidden_states=True).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        return self.net(embeddings) # (B, S)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = self.tokenizer(["<+>" + example['x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        return inputs
    
    def batch_decode(self, batch):
        return self.tokenizer.batch_decode(batch['x'].input_ids, skip_special_tokens=True)
