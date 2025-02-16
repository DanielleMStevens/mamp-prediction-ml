import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class ESMMidModel(nn.Module):
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
            nn.Linear(int(E // 2), 3),
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  

    def forward(self, batch_x):
        embeddings = self.esm_model(**batch_x, output_hidden_states =True).hidden_states[15][:, 0, :] # (B, E_esm) # Get CLS token
        return self.net(embeddings) # (B, S)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = self.tokenizer([example['x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
        return inputs
