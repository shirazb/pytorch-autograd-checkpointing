from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from .RecomputableFunction import RecomputableFunction

def checkpoint_sequential(model, M):
    # 1. Profile layers of model
    # 2. Solve for optimal policy
    # 3. Encode policy into model using Drop
    pass

def checkpoint(run_function, *args, **kwargs):
    recomp_depth = kwargs.pop('recomp_depth')
    preserve = kwargs.pop('preserve_rng_state', True)
    
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return RecomputableFunction.apply(run_function, recomp_depth, preserve, *args)
