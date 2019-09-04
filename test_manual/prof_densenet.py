import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest, time, sys

import models.densenet as DenseNetFactory

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

if __name__ == '__main__':
    prof_densenet()
