import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ESMWithReceptorSingleSeqModel(nn.Module):
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
        embeddings = self.esm_model(**batch_x).pooler_output
        return self.net(embeddings) # (B, S)
    
    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['peptide_x'] = self.tokenizer([example['peptide_x'] for example in batch],
                return_tensors='pt', padding=True)
        inputs['receptor_x'] = self.tokenizer([example['receptor_x'] for example in batch],
                return_tensors='pt', padding=True)
        

        inputs['x'] = self.tokenizer([f"{example['peptide_x']}<eos><cls>{example['receptor_x'] }" for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])
 
        return inputs

    def batch_decode(self, batch):
        peptide_decoded_ls = self.tokenizer.batch_decode(batch['peptide_x']['input_ids'], skip_special_tokens=True)
        receptor_decoded_ls = self.tokenizer.batch_decode(batch['receptor_x']['input_ids'], skip_special_tokens=True)
        
        return [f"{peptide}:{receptor}" for peptide, receptor in zip(peptide_decoded_ls, receptor_decoded_ls)]