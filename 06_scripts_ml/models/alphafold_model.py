"""
A simple neural network model that processes AlphaFold embeddings.
This model takes pre-computed AlphaFold embeddings (256-dimensional) as input
and performs classification into 3 classes through a 2-layer neural network.

The model architecture:
1. Adaptive average pooling to handle variable input sizes
2. Two linear layers with ReLU activation in between
3. Final output of 3 classes (likely representing some protein property)
"""
import torch
import torch.nn as nn


class AlphaFoldModel(nn.Module):
    def __init__(self, args=None, input_size=256):
        super(AlphaFoldModel, self).__init__()
        # Convert variable sized input to fixed size through pooling
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # First linear layer maintains dimensionality
        self.linear = nn.Linear(input_size, input_size)
        # Second linear layer reduces to 3 classes
        self.linear2 = nn.Linear(input_size, 3)
        
        self.tokenizer = None
    
    def forward(self, x):
        # Pool the input to handle variable sizes
        x = self.adaptive_avg_pool(x).squeeze(dim=(-1, -2))
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        return x
    

    def get_tokenizer(self):
        return self.tokenizer

    def collate_fn(self, batch):
        inputs = {}

        inputs['x'] = torch.stack([example['x'] for example in batch])
        
        inputs['y'] = torch.tensor([example['y'] for example in batch])

        inputs['seqs'] = [example['seqs'] for example in batch]
        return inputs


    def batch_decode(self, batch):
        return batch['seqs']