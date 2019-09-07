import os
import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import pytorch_autograd_checkpointing as c
import models.densenet as DenseNetFactory

from testing_utils import *

_DEFAULT_OUTFILE_PREFIX = 'results/'
_DEFAULT_OUTFILE_NAME = 'optimal_cost_vs_bucket_size.png'

_DEFAULT_DATA_DIR = 'data/'
_DEFAULT_RESULTS_NAME = 'optimal_cost_vs_bucket_size.p'

def plot_optimal_cost_against_bucket_size():
    mb = int(1e6)
    bucket_sizes = [
        int(b * mb) for b in range(1, 11, 3)
    ] + [
        int(b * mb) for b in range(15, 56, 10)
    ]

    batch_size = 32
    budget_leeway = 0.2
    gpu_mem_capcity_bytes = int(6e9)

    models = [
        {
            'name': 'DenseNet-121',
            'chkpter': mk_checkpointed_densenet(DenseNetFactory.densenet121()),
            'x': densenet_dummy_input(batch_size),
            'b': (torch.tensor(1.),)
        }
    ]

    results = {}

    # For each model, find the optimal policy for each bucket size, and plot corresponding optimal costs.
    for model in models:
        name = model['name']
        chkpter = model['chkpter']
        x = model['x']
        b = model['b']

        # Init results for this model
        results[name] = { 'b': [], 'c': [], 't': [] }

        # Use same profile results for each bucket size
        chkpter.profile_sequence(x, b)

        for bucket_size in bucket_sizes:
            compute_costs, memory_costs = bucket_costs(chkpter, bucket_size, log=True)
            
            _warm_up_device('cpu')

            start_cpu_s = time.time()
            _, C, _ = solve_policy_using_costs(
                    chkpter,
                    compute_costs, memory_costs,
                    gpu_mem_capcity_bytes,
                    bucket_size,
                    budget_leeway
            )
            end_cpu_s = time.time()

            results[name]['b'].append(bucket_size)
            results[name]['c'].append(C[0, -1, -1])
            results[name]['t'].append((end_cpu_s - start_cpu_s))

            _serialise(results, os.path.join(_DEFAULT_DATA_DIR, _DEFAULT_RESULTS_NAME))

        print('Done {}'.format(name))

    # Plot optimal cost vs bucket size graph for each model.

    cols = 2
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(cols*7, rows*7))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        for j in range(cols):
            n = i * cols + j

            if n >= len(models):
                break

            model = models[n]
            name = model[name]

            ax = axes[i, j]

            bs = int(results[name]['b'] // mb)

            ax.plot(bs, results[name]['c'], 'b')
            ax.set_xlabel('Bucket Size, MB')
            ax.set_ylabel('Optimal (Simulated) Computational Cost, ms')

            ax = ax.twinx()
            ax.plot(results[name]['t'], 'r-')
            ax.set_ylabel('Solver Execution Time, s')

            ax.set_title(name)

    #fig.suptitle('Compute-Memory Trade-Off for Varying Number Layers, N')
    
    outfile_path = _DEFAULT_OUTFILE_PREFIX + _DEFAULT_OUTFILE_NAME
    plt.savefig(outfile_path, bbox_inches='tight')

def _serialise(results, results_path):
    pickle.dump(
            results,
            open(results_path, "wb")
    )

if __name__ == "__main__":
    plot_optimal_cost_against_bucket_size()
