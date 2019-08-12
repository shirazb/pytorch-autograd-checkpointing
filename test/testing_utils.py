import torch
    
# run_function should already by on correct device
def calc_grad(run_function, device='cpu', dtype=torch.float32):
    t = torch.randn(10, 10, requires_grad=True, device=device, dtype=dtype)

    y = run_function(t).sum()
    y.backward()

    return t.grad

    