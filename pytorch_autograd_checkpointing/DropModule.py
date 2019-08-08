from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from .checkpoint_sequential import checkpoint


class Drop(torch.nn.Module):
    """
    Drop module used to encode a checkpointing policy into a sequence of 
    modules.
    """
    def __init__(self, child, recomp_depth):
        super(Drop, self).__init__()

        self.child = child
        self.recomp_depth = recomp_depth # must be > 0
    
    def forward(self, x):
        self.recomp_depth -= 1
        return checkpoint(self.child, recomp_depth=self.recomp_depth)
    