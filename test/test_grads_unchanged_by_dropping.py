import pytest

import torch
import pytorch_autograd_checkpointing as c

import testing_utils as ut

def test_grads_unchanged_by_dropping():
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

    grad_original = ut.calc_grad(original_model)

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

    grad_with_checkpointing = ut.calc_grad(checkpointed_model)

    ## Compare equal
    assert (grad_with_checkpointing == grad_original).all(), (
            'Grads not equal.\n'    
            '    With:\n'
            '      {}\n'
            '    Without:\n'
            '      {}'
            .format(grad_with_checkpointing, grad_original)
    )
    