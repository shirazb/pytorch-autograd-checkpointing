import numpy as np
import torch
import matplotlib.pyplot as plt

import pytorch_autograd_checkpointing as c
import models.densenet as DenseNetFactory

from testing_utils import *

_DEFAULT_OUTFILE_PREFIX = 'results/'
_DEFAULT_OUTFILE_NAME = 'compute_mem_tradeoff_densenet.png'

def plot_compute_memory_tradeoff():
    results = {}

    models = [
        (DenseNetFactory.densenet121(), 'DenseNet-121'),
        (DenseNetFactory.densenet201(), 'DenseNet-201'),
        (DenseNetFactory.densenet264(), 'DenseNet-264')
    ]

    # models = [
    #     (mk_seq(10, 'cuda', 1000), 'DenseNet-121'),
    #     (mk_seq(10, 'cuda', 1000), 'DenseNet-201'),
    #     (mk_seq(10, 'cuda', 1000), 'DenseNet-264')
    # ]

    for (model, model_name) in models:
        # Set up results
        dense_chkpter = mk_checkpointed_densenet(model)

        batch_size = 32
        bucket_size_mb = int(4e6)
        budget_leeway = 0.2
        gpu_mem_capcity = int(9e9)

        x = densenet_dummy_input(batch_size)
        b = (torch.tensor(1.),)

        _, C, _ = prof_and_solve_policy(
                dense_chkpter,
                x, b,
                gpu_mem_capcity,
                bucket_size_mb,
                budget_leeway,
                log=True
        )

        # The x axis, up to the internal memory budget in increments of 20 buckets.
        MS = np.arange(4, C.shape[2]-1, 20 * bucket_size_mb, dtype=np.int16)

        mb = int(1e6)
        results[model_name] = { 'm': MS * int(bucket_size_mb // mb), 'c': C[0, -1, MS] }

        print('Done {}'.format(model_name))


    # Plot C-M graphs for each N.

    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(cols*7, rows*7))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        for j in range(cols):
            n = i * cols + j

            if n >= len(models):
                break
            
            model_name = models[n][1]

            axes[i, j].plot(results[model_name]['m'], results[model_name]['c'])
            axes[i, j].set_xlabel('Memory Budget, MB')
            axes[i, j].set_ylabel('Simulated Computational Cost, ms')
            axes[i, j].set_title(model_name)

    #fig.suptitle('Compute-Memory Trade-Off for Varying Number Layers, N')
    
    outfile_path = _DEFAULT_OUTFILE_PREFIX + _DEFAULT_OUTFILE_NAME
    plt.savefig(outfile_path, bbox_inches='tight')

if __name__ == "__main__":
    plot_compute_memory_tradeoff()
