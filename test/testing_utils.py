import torch

def calc_grad(run_function):
    t = torch.randn(10, 10, requires_grad=True)

    y = run_function(t).sum()
    y.backward()

    return t.grad

    