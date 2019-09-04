import pytest

import numpy as np
import torch
import pytorch_autograd_checkpointing as c

from testing_utils import *

def test_policy_solver():
    N = 20
    M = 300
    compute_costs = np.random.rand(2, N+2) * 10
    memory_costs = np.random.randint(2, high=60, size=(2,N+2), dtype=np.int16)

    #memory_costs = np.array([[2, 4, 2, 1, 2, 3, 1, 3, 3, 2, 1, 2],
     #                        [2, 1, 1, 3, 3, 2, 2, 3, 1, 2, 1, 3]])

    print('costs:')
    print(memory_costs)
    print(compute_costs)

    c_s = c.CheckpointedSequential(mk_seq(N))

    C, D = c_s.solve_optimal_policy(
        M, 
        compute_costs=compute_costs, memory_costs=memory_costs,
        profile_compute=False, profile_memory=False
    )

    print("Cost Lower Bound: {}".format(_calc_lower_bound(N, compute_costs)))
    print("Cost Upper Bound: {}".format(_calc_upper_bound(N, compute_costs)))
    
    print("cost =", C[0, -1, -1])
    print("policy =", D[0, -1, -1])
    print('C:')
    print(np.triu(C[:, :, -1], 1))
    print('D:')
    print(np.triu(D[:, :, -1], 2))

def test_returns_valid_traversable_policy():
    pass

def test_no_recomputation_when_memory_plentiful():
    pass

def test_picks_quadratic_when_memory_small():
    pass

def test_fails_when_memory_insufficient():
    pass

def _calc_upper_bound(N, compute_costs):
    acc = 0.0
    for k in range(0, N+1):
        acc += (k+1) * compute_costs[0, N-k] + compute_costs[1, N-k]
    
    return acc

def _calc_lower_bound(N, compute_costs):
    return np.sum(compute_costs) - np.sum(compute_costs[np.array([0, 0, 1]), np.array([0, N+1, N+1])])