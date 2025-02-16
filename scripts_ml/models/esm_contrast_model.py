"""
A contrastive learning model based on Facebook's ESM-2 for protein sequence analysis.
This model extends the base ESM model to perform both contrastive learning and classification.

Key features:
1. Uses frozen ESM-2 as the base encoder
2. Implements a contrastive learning head for learning protein sequence similarities
3. Includes a classification head for 3-class prediction
4. Supports both contrastive and classification losses ('supcon' and 'ce')

The model can output either:
- Just classification logits (default)
- Both classification logits and contrastive features (when contrastive_output=True)

This architecture is useful for learning protein sequence representations that capture
both similarity relationships and classification targets.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

class ESMContrastiveModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        for param in self.esm_model.parameters():
            param.requires_grad = False     

        E = self.esm_model.config.hidden_size

        self.net = nn.Sequential(
            nn.Linear(E, E),
            nn.ReLU(),
            nn.Linear(E, int(E // 2))
        )
        self.head = nn.Linear(int(E // 2), 3)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")  

        self.losses = ['ce', 'supcon']

        self.contrastive_output = False
        if args and args.contrastive_output:
            self.contrastive_output = True

    def forward(self, batch_x):
        embeddings = self.esm_model(**batch_x).last_hidden_state[:, 0, :] # (B, E_esm) # Get CLS token
        contrastive_embeddings = self.net(embeddings)
        logits = self.head(contrastive_embeddings)
        if self.contrastive_output:
            return {"logits": logits, "features": contrastive_embeddings}
    
    def get_tokenizer(self):
        return self.tokenizer
    

    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = self.tokenizer([example['x'] for example in batch],
                return_tensors='pt', padding=True)
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])

        inputs['y_mask'] = ( inputs['y'].unsqueeze(0) ==  inputs['y'].unsqueeze(1)).int()

        return inputs

    
    def get_pr(self, output):
        logits = output['logits']
        preds = logits.sigmoid().detach().cpu()
        return preds
    
    def get_stats(self, gt, pr, train):
        if train:
            s = 'train'
        else:
            s = 'test'
        stats = {
            f"{s}_acc@1": accuracy_score(gt.cpu(), pr.argmax(dim=-1)),
        }
        return stats
    

    def batch_decode(self, batch):
        return self.tokenizer.batch_decode(batch['x']['input_ids'], skip_special_tokens=True)
