import torch
import numpy as np
import pytorch_autograd_checkpointing as c

def test_something():
    pass

########### HELPERS FROM TESTING UTILS #########################################
_DEFAULT_LAYER_DIM = 10
def mk_seq(N, dim=_DEFAULT_LAYER_DIM, device='cpu'):
    return torch.nn.Sequential(
        *(layer(dim) for _ in range(N))
    ).to(device)

def layer(dim=_DEFAULT_LAYER_DIM):
    return ThresholdedLinear(dim, dim)

class ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThresholdedLinear, self).__init__()
    
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))
