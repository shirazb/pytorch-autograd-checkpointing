import pytest

import torch
import pytorch_autograd_checkpointing as c

def _calc_grad(run_function):
    t = torch.randn(10, 10, requires_grad=True)

    y = run_function(t).sum()
    y.backward()

    return t.grad

def test_does_not_change_grad():
    ## Calc grad using original model
    
    # For reproducability
    torch.manual_seed(23)

    original_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
    )

    grad_original = _calc_grad(original_model)

    ## Calc grad using checkpointed model

    torch.manual_seed(23)

    checkpointed_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            c.Drop(
                torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    c.Drop(
                        torch.nn.Linear(10, 10)
                    , 2),
                    torch.nn.Linear(10, 10)
                )
            , 1),
            torch.nn.Linear(10, 10)
    )

    grad_with_checkpointing = _calc_grad(checkpointed_model)

    ## Compare equal
    assert (grad_with_checkpointing == grad_original).all(), (
            'Grads not equal.\n'    
            '    With:\n'
            '      {}\n'
            '    Without:\n'
            '      {}'
            .format(grad_with_checkpointing, grad_original)
    )
    