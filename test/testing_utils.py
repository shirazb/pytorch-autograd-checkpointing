import torch

def calc_grad(run_function, device='cpu'):
    t = torch.randn(10, 10, requires_grad=True, device=device)

    y = run_function(t).sum()
    y.backward()

    return t.grad

    