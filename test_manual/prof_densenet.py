import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest, time, sys

import models.densenet as DenseNetFactory
import pytorch_autograd_checkpointing as c

def prof_densenet():
    N = 32
    total_iters = 20    # (warmup + benchmark)
    iterations = 1

    x = Variable(torch.randn(N, 3, 224, 224).fill_(1.0), requires_grad=True)
    target = Variable(torch.randn(N).fill_(1)).type("torch.LongTensor")
    # model = DenseNetFactory.densenet100()
    model = DenseNetFactory.densenet121()
    # model = DenseNetFactory.densenet201()
    # model = DenseNetFactory.densenet264()

    # switch the model to train mode
    model.train()

    # convert the model and input to cuda
    model = model.cuda()
    input_var = x.cuda()
    target_var = target.cuda()

    # declare the optimizer and criterion
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with cudnn.flags(enabled=True, benchmark=True):
        for i in range(total_iters):
            start.record()
            start_cpu = time.time()
            for j in range(iterations):
                output = model(input_var)
                loss = criterion(output, target_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_cpu = time.time()
            end.record()
            torch.cuda.synchronize()
            gpu_msec = start.elapsed_time(end)
            print("Baseline densenet ({:2d}): ({:8.3f} usecs gpu) ({:8.3f} usecs cpu)".format(
                i, gpu_msec * 1000, (end_cpu - start_cpu) * 1000000,
                file=sys.stderr))

class SumLayer(nn.Module):
    def forward(self, x):
        return x.sum()

def run_solver_densenet(densenet):
    batch_size = 32
    
    modules = [module for k, module in densenet._modules.items()][0]

    assert isinstance(modules, nn.Sequential), 'run_solver_densenet(): modules was not a nn.Sequential'

    modules.add_module('sum_loss', SumLayer())
    c_s = c.CheckpointedSequential(modules)

    x = torch.randn(batch_size, 3, 224, 224).fill_(1.0)
    b = (torch.tensor(1.),)

    # profile
    c_s.profile_sequence(x, b)

    # gett costs
    memory_costs = c_s.memory_costs
    memory_costs = memory_costs // 1e4
    compute_costs = c_s.compute_costs

    # print costs
    print('Costs: (m,c)')
    print(memory_costs)
    print(compute_costs)

    M = 1e4
    C, D = c_s.solve_optimal_policy(
        M, 
        compute_costs=compute_costs, memory_costs=memory_costs,
        profile_compute=False, profile_memory=False
    )

    print("Cost Lower Bound: {}".format(_calc_lower_bound(compute_costs)))
    print("Cost Upper Bound: {}".format(_calc_upper_bound(compute_costs)))
    
    print("cost =", C[0, -1, -1])
    print("policy =", D[0, -1, -1])
    print('C:')
    print(np.triu(C[:, :, -1], 1))
    print('D:')
    print(np.triu(D[:, :, -1], 2))


def _calc_upper_bound(compute_costs):
    acc = 0.0
    N = compute_costs.shape[1]
    for k in range(0, N):
        acc += (k+1) * compute_costs[0, N-k] + compute_costs[1, N-k]
    
    return acc

def _calc_lower_bound(compute_costs):
    return np.sum(compute_costs) - np.sum(compute_costs[np.array([0, 0, 1]), np.array([0, -1, -1])])


############ MAIN ###########################
if __name__ == '__main__':
    # model = DenseNetFactory.densenet100()
    model = DenseNetFactory.densenet121()
    # model = DenseNetFactory.densenet201()
    # model = DenseNetFactory.densenet264()

    run_solver_densenet(model)
