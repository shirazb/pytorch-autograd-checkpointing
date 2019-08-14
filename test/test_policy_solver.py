import pytest

import numpy as np
import torch
import pytorch_autograd_checkpointing as c

def test_policy_solver():
    N = 20
    M = 300
    compute_costs = np.random.rand(2, N+2) * 10
    memory_costs = np.random.randint(2, high=60, size=(2,N+2), dtype=np.int16)

    #memory_costs = np.array([[2, 4, 2, 1, 2, 3, 1, 3, 3, 2, 1, 2],
     #                        [2, 1, 1, 3, 3, 2, 2, 3, 1, 2, 1, 3]])

    print(memory_costs)
    print(compute_costs)

    err, C, D = c.solve_optimal_policy(N, M, compute_costs, memory_costs, c.Logger(c.LOG_LEVEL_VERBOSE))

    if err != 0:
        assert False, 'Solver failed with code: {}'.format(err)
    
    print("cost =", C[0, -1, -1])
    print("policy =", D[0, -1, -1])
    print('C:')
    print(np.triu(C[:, :, -1], 1))
    print('D:')
    print(np.triu(D[:, :, -1], 2))
