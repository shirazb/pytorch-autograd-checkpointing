import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_autograd_checkpointing as c

from testing_utils import *

_DEFAULT_OUTFILE_PREFIX = 'results/'
_DEFAULT_OUTFILE_NAME = 'compute_mem_tradeoff.png'

def plot_compute_memory_tradeoff():
    device = 'cpu'

    # Set up results
    M = 60
    MS = np.arange(4, M, 2, dtype=np.int16)
    NS = [20, 40, 100]

    results = np.ones((len(MS),len(NS)), dtype=np.single)

    # Set up dummy data, with uniform costs.
    compute_costs = np.ones((2, NS[-1] + 2), dtype=np.single)
    memory_costs = np.ones((2, NS[-1] + 2), dtype=np.int16)

    # Perform experiments.
    for j, n in enumerate(NS):
        chkp_seq = c.CheckpointedSequential(mk_seq(n, device, dim=10), device)
        _, C, _ = chkp_seq.solve_optimal_policy(
            M,
            memory_costs=memory_costs[:, :n+2], compute_costs=compute_costs[:, :n+2],
            profile_memory=False, profile_compute=False
        )

        results[:len(MS), j] = C[0, -1, MS]

        print("Done n =", n)

    # Plot a different C-M graph for each N.

    cols = 3
    rows = int(np.ceil(len(NS) / cols))
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(cols*7, rows*7))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        for j in range(cols):
            n = i * cols + j

            if n >= len(NS):
                break

            axes[i, j].plot(MS, results[:, n])
            axes[i, j].set_xlabel('Memory Budget')
            axes[i, j].set_ylabel('Simulated Computational Cost')
            axes[i, j].set_title('N = {}'.format(NS[n]))

    fig.suptitle('Compute-Memory Trade-Off for Varying Number Layers, N')
    
    outfile_path = _DEFAULT_OUTFILE_PREFIX + _DEFAULT_OUTFILE_NAME
    plt.savefig(outfile_path, bbox_inches='tight')

if __name__ == "__main__":
    plot_compute_memory_tradeoff()
