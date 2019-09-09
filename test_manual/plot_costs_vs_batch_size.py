import os
import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import pytorch_autograd_checkpointing as c
import models.densenet as DenseNetFactory
import models.resnet as ResNetFactory

from testing_utils import *


_DEFAULT_OUTFILE_PREFIX = 'results/'
_DEFAULT_OUTFILE_NAME = 'cost_vs_batch_size.png'

_DEFAULT_DATA_DIR = 'data/'
_DEFAULT_RESULTS_NAME = 'cost_vs_batch_size.p'

optim_types = ['profile_both', 'profile_comp', 'profile_mem', 'uniform_both']
optim_types = ['profile_both', 'uniform_both']


def plot_costs_vs_batch_size(skip, read):
    mb = int(1e6)

    bucket_size = int(4 * mb)
    budget_leeway = 0.125 # (% percent)

    # results = {
    #     'profile_both': {
    #         'bs': [2, 4, 6, 7, 8],
    #         'time': [10, 20, 22, 23, 24],
    #         'peak': [5, 6, 10, 11, 12]
    #     },
    #     'profile_comp': {
    #         'bs': [2, 4, 6],
    #         'time': [11, 20, 22],
    #         'peak': [5, 8, 11]
    #     },
    #     'profile_mem': {
    #         'bs': [2, 4, 6],
    #         'time': [11, 22, 23],
    #         'peak': [6, 8, 12]
    #     },
    #     'uniform_both': {
    #         'bs': [2, 4],
    #         'time': [12, 23],
    #         'peak': [10, 12]
    #     }
    # }
    results = {
        'profile_both': {
            'bs': [],
            'time': [],
            'peak': []
        },
        'profile_comp': {
            'bs': [],
            'time': [],
            'peak': []
        },
        'profile_mem': {
            'bs': [],
            'time': [],
            'peak': []
        },
        'uniform_both': {
            'bs': [],
            'time': [],
            'peak': []
        }
    }

    if read:
        with open(os.path.join(_DEFAULT_DATA_DIR, _DEFAULT_RESULTS_NAME), "rb") as f:
            results = pickle.load(f)

    def incr_bs(bs):
        if bs == 16:
            bs += 16
        elif bs == 32:
            bs += 32
        else:
            bs += 64
        return bs

    def incr2_bs(bs):
        if bs == 8:
            bs += 8
        else:
            bs += 16
        return bs

    if not skip:
        for optim_type in optim_types:
            bs = 8
            while True:
                # get model, input, upstream grads, max mem
                chkpter = mk_checkpointed_resnet(ResNetFactory.resnet_c_50())
                x = resnet_dummy_input(bs)
                b = (torch.tensor(1.),)
                M = int(8.5e9)

                # do stupid CPU thing to stop cuda error (wtf?)
                # print('cpu dumb thing...')
                # tmp = torch.nn.Sequential(*chkpter.sequence)
                # tmp2 = tmp(x)
                # print('...done')

                # run profiler (if optim_type != 'uniform')
                # set up kwargs for solver as required for this optim_type
                # if profiler fails (gpu out of memory) then stop collecting data for this optim_type
                try:
                    chkpter.profile_sequence(x, b)
                    real_compute_costs, real_memory_costs = bucket_costs(chkpter, bucket_size)
                    solver_compute_costs, solver_memory_costs = real_compute_costs, real_memory_costs
                    if optim_type == 'uniform_both' or optim_type == 'profile_mem':
                        solver_compute_costs = None
                    if optim_type == 'uniform_both' or optim_type == 'profile_comp':
                        # Uniform costs setting all fwd and bwd costs to the average.
                        # M is still correct this way, and the costs are uniform.
                        solver_memory_costs = _average_mem_costs(real_memory_costs)
                except RuntimeError as err:
                    print(err)
                    print('Assuming profiler ran out of memory for optim_type=%s at bs=%d' % (optim_type, bs))
                    break

                # run solver
                # if solver fails (not enough memory) then stop collecting data for this optim_type
                try:
                    budget = budget_bucketed_with_leeway(M, bucket_size, budget_leeway)
                    
                    B, C, policy = chkpter.solve_optimal_policy(
                        budget,
                        compute_costs=solver_compute_costs, memory_costs=solver_memory_costs,
                        profile_compute=False, profile_memory=False
                    )
                except c.CheckpointSolverFailure as err:
                    print(err)
                    print('Solver exceeded memory for optim_type=%s at bs=%d' % (optim_type, bs))
                    break

                # run simulator to estimate executor time and peak mem usage
                # if simulator estimates peak memory greater than budget, stop collecting data for this optim_type
                time, peak = chkpter.simulate_sequence(policy, real_memory_costs, real_compute_costs)
                if peak > budget:
                    print('Simulator exceeded memory with budget=%d peak=%d at optim_type=%s bs=%d' % (
                        budget, peak, optim_type, bs
                    ))
                    break

                print('{} Sim Results: bs = {}, time = {}, peak = {}'.format(optim_type, bs, time, peak))

                # append results (and write results so far to file)
                results[optim_type]['bs'].append(bs)
                results[optim_type]['time'].append(time)
                results[optim_type]['peak'].append((peak * bucket_size) / mb)

                _serialise(results, os.path.join(_DEFAULT_DATA_DIR, _DEFAULT_RESULTS_NAME))
                print('completed optim_type=%s batch_size=%d' % (optim_type, bs))

                bs = incr2_bs(bs)

    # PLOTTING

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    ax1 = axes[0]
    ax2 = axes[1]
    fmts = {
        'profile_both': 'r+',
        'profile_comp': 'bo',
        'profile_mem': 'g-',
        'uniform_both': 'm--'
    }

    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Simulated Time (ms)')
    for optim_type in optim_types:
        batch_sizes = results[optim_type]['bs']
        times = results[optim_type]['time']
        fmt = fmts[optim_type]
        ax1.plot(batch_sizes, times, fmt, label=prettify(optim_type))
    ax1.legend()

    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Simulated Peak Memory (MB)')
    for optim_type in optim_types:
        batch_sizes = results[optim_type]['bs']
        peaks = results[optim_type]['peak']
        fmt = fmts[optim_type]
        ax2.plot(batch_sizes, peaks, fmt, label=prettify(optim_type))
    ax2.legend()

    outfile_path = os.path.join(_DEFAULT_OUTFILE_PREFIX, _DEFAULT_OUTFILE_NAME)
    plt.savefig(outfile_path)


def _serialise(results, results_path):
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

def prettify(optim_type):
    if optim_type == 'profile_both':
        return 'Profile Compute and Memory'
    elif optim_type == 'profile_comp':
        return 'Profile Compute Only'
    elif optim_type == 'profile_mem':
        return 'Profile Memory Only'
    elif optim_type == 'uniform_both':
        return 'Uniform Compute and Memory'
    else:
        print('Called prettify(optim_type=%s) with unknown optim_type.' % optim_type)
        return optim_type

def _average_mem_costs(memory_costs):
    uniform_memory_costs = np.empty_like(memory_costs)
    uniform_memory_costs[0, :] = np.ceil(np.average(memory_costs[0, :])).astype(np.int16)
    uniform_memory_costs[1, :] = np.ceil(np.average(memory_costs[1, :])).astype(np.int16)
    return uniform_memory_costs

if __name__ == "__main__":
    skip = False
    read = False
    plot_costs_vs_batch_size(skip, read)



"""
profile_both:

chkpter.solve_optimal_policy(
                budget,
                compute_costs=compute_costs, memory_costs=memory_costs,
                profile_compute=False, profile_memory=False
            )


profile_comp:

chkpter.solve_optimal_policy(
                budget,
                compute_costs=compute_costs, memory_costs=None,
                profile_compute=False, profile_memory=False
            )

profile_mem:

chkpter.solve_optimal_policy(
                budget,
                compute_costs=None, memory_costs=memory_costs,
                profile_compute=False, profile_memory=False
            )

uniform_both:

chkpter.solve_optimal_policy(
                budget,
                profile_compute=False, profile_memory=False
            )
"""
