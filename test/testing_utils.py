import torch

_DEFAULT_LAYER_DIM = 10

# run_function should already by on correct device
def calc_grad(run_function, device='cpu', dtype=torch.float32):
    t = torch.randn(10, 10, requires_grad=True, device=device, dtype=dtype)

    y = run_function(t).sum()
    y.backward()

    return t.grad

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

