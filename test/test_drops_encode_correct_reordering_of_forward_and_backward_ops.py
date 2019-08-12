import pytest

from collections import namedtuple

import torch
import pytorch_autograd_checkpointing as c

import testing_utils as ut

# OP enum to identify 'f' or 'b' ops in log
OpDirs = namedtuple('OP', ['F', 'B'])
OP = OpDirs(0, 1)

def test_drops_encode_correct_reordering_of_forward_and_backward_ops():
    """
      Runs a model with drops encoded that logs the order in which the forward
      and backward operations of each layer is invoked. Compares the log to an 
      expected log.
      
      Also checks that the dropping did not affect the computed gradient, by
      comparing to a reference model with no dropping encoded.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Declare log and forward/backward hooks that will write to it.

    op_log = []

    def forward_hook(module, input, output):
        if isinstance(module, LogOpsModule):
            op_log.append((module.i, OP.F))
    
    def backward_hook(module, grad_input, grad_output):
        if isinstance(module, LogOpsModule):
            op_log.append((module.i, OP.B))
    
    def layer(i):
        return _log_layer_with_hooks(i, forward_hook, backward_hook)
    
    # For reproducability.
    torch.manual_seed(86)

    model = torch.nn.Sequential(
            layer(1),
            c.Drop(_seq(
                layer(2),
                c.Drop(_seq(
                    c.Drop(_seq(
                        layer(3),
                        layer(4)
                    ), 3),
                    layer(5),
                    c.Drop(layer(6), 3)
                ), 2)
            ), 1),
            c.Drop(_seq(
                layer(7),
                layer(8)
            ), 1),
            layer(9)
    ).to(device)
    
    ## 1. Assert grads the same.

    grad_actual = ut.calc_grad(model, device)

    # Reference is same as above with no `Drop`-ing.
    torch.manual_seed(86)
    grad_expected = ut.calc_grad(_reference_model(), device)

    assert (grad_actual == grad_expected).all(), ('Grads not equal.\n'    
            '    With:\n'
            '      {}\n'
            '    Without:\n'
            '      {}'
            .format(grad_actual, grad_expected))

    ## 2. Assert logs the same.

    expected_log = _get_expected_log()
    actual_log = op_log

    assert actual_log == expected_log, ('Actual operator log does not'
            ' match the expected log.\n'
            'actual: {}\n'
            'expected: {}'
            .format(actual_log, expected_log))

####### HELPERS #######

class LogOpsModule(torch.nn.Module):
    def __init__(self, child, i, forward_hook, backward_hook):
        super(LogOpsModule, self).__init__()
        self.i = i
        self.child = child

        self.register_forward_hook(forward_hook)
        self.register_backward_hook(backward_hook)
    
    def forward(self, x):
        return self.child(x)

def _seq(*args):
    return torch.nn.Sequential(*args)

def _base_layer():
    return torch.nn.Linear(10, 10)

def _log_layer_with_hooks(i, forward_hook, backward_hook):
    return LogOpsModule(_base_layer(), i, forward_hook, backward_hook)

def _no_log_layer():
    return _base_layer()

def _reference_model():
    return torch.nn.Sequential(
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer(),
            _no_log_layer()
    )

def _get_expected_log():
    return [
        (1, OP.F), (2, OP.F), (3, OP.F), (4, OP.F), (5, OP.F), (6, OP.F), (7, OP.F), (8, OP.F), (9, OP.F),
                                                                                                (9, OP.B),
                                                                          (7, OP.F), (8, OP.F),
                                                                                     (8, OP.B),
                                                                          (7, OP.B),
                   (2, OP.F), (3, OP.F), (4, OP.F), (5, OP.F), (6, OP.F),
                              (3, OP.F), (4, OP.F), (5, OP.F), (6, OP.F),
                                                               (6, OP.F),
                                                               (6, OP.B),
                                                    (5, OP.B),
                              (3, OP.F), (4, OP.F),
                                         (4, OP.B),
                              (3, OP.B),
                   (2, OP.B),
        (1, OP.B)
    ]