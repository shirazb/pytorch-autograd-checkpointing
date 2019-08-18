import pytest

import numpy as np
import torch
import pytorch_autograd_checkpointing as c

from .testing_utils import mk_seq

def test_run_sequence():
    N = 20
    M = 300
    compute_costs = np.random.rand(2, N+2) * 10
    memory_costs = np.random.randint(2, high=60, size=(2,N+2), dtype=np.int16)

    err, C, D = c.solve_optimal_policy(N, M, compute_costs, memory_costs, c.Logger(c.LOG_LEVEL_VERBOSE))

    if err != 0:
        assert False, 'Solver failed with code: {}'.format(err)
    
    x = torch.randn(10, 10)
    t = torch.randn(10, 10)
    seq = mk_seq(N-1)
    loss = torch.nn.CrossEntropyLoss
    seq.append(loss)

    # TODO: Accept targets
    c.run_sequence(
            seq, D, N, M, compute_costs, memory_costs, x
    )
