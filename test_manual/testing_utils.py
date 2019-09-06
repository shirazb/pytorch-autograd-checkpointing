import numpy as np
import torch

_DIM = 100

def warm_up_device(device):
    '''
      Perform some arbitrary computation on device that will be immediately
      discarded to warm up the device to peak performance.
    '''
    (torch.randn(3000, 3000, device=device) * torch.randn(3000, 3000, device=device)).sum()

def mk_seq(N, device, dim=_DIM):
    return torch.nn.Sequential(
        *(layer(dim) for _ in range(N))
    ).to(device)

def layer(dim=_DIM):
    return ThresholdedLinear(dim, dim)

class ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThresholdedLinear, self).__init__()

        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))
