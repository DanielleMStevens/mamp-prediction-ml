"""
A model that uses the AMPLIFY language model for antimicrobial peptide analysis.
This model leverages the pre-trained AMPLIFY 350M parameter model, which is specifically
designed for understanding antimicrobial peptide sequences.

The model:
1. Uses a frozen AMPLIFY model as the base encoder
2. Takes the first token's embedding for sequence representation
3. Passes through a 3-layer MLP to predict 3 classes

AMPLIFY is specialized for antimicrobial peptide understanding, making it particularly
suitable for tasks related to antimicrobial activity prediction.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class AMPModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # Load and freeze AMPLIFY model
        self.model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False     

        E = self.model.config.hidden_size

        # Classification head: 3-layer MLP with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),
        )
        # Initialize tokenizer for antimicrobial peptide sequences
        self.tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)

    def forward(self, batch_x):
        embeddings = self.model(batch_x.input_ids, attention_mask=batch_x.attention_mask.float(), output_hidden_states=True).hidden_states[-1][:, 0, :]
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