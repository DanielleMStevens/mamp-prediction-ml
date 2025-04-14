import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel  # type: ignore

class GLMWithReceptorSingleSeqModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.model = AutoModel.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False     

        E = self.model.config.dim

        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2)),
            nn.ReLU(),
            nn.Linear(int(E // 2), 3),
        )

        self.tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M", trust_remote_code=True) 

    def forward(self, batch_x):

        attention_mask = batch_x.attention_mask
        
        embeddings = self.model(batch_x.input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state 

        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())

        mean_pooled_output = (embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        return self.net(mean_pooled_output)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['peptide_x'] = self.tokenizer(['<+>' + example['peptide_x'] for example in batch],
                return_tensors='pt', padding=True)
        inputs['receptor_x'] = self.tokenizer(['<+>' + example['receptor_x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])

        inputs['x'] = self.tokenizer([f"<+>{example['peptide_x']}<+>{example['receptor_x'] }" for example in batch],
                return_tensors='pt', padding=True)
 
        return inputs

    def batch_decode(self, batch):
        peptide_decoded_ls = self.tokenizer.batch_decode(batch['peptide_x'].input_ids, skip_special_tokens=True)
        receptor_decoded_ls = self.tokenizer.batch_decode(batch['receptor_x'].input_ids, skip_special_tokens=True)
        
        return [f"{peptide}:{receptor}" for peptide, receptor in zip(peptide_decoded_ls, receptor_decoded_ls)]