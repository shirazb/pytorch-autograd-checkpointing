from __future__ import absolute_import, division, print_function, unicode_literals

import torch


class RecomputatableFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, recomp_depth, preserve_rng_state, *args):
        # if recomp_depth == 0:
        #    checkpoint the input
        #    save for backward
        # with no grad run the function and return the output
        pass

    @staticmethod
    def backward(ctx, *args):
        # recover checkpoint from ctx
        # run function with grad
        # run backward
        pass
