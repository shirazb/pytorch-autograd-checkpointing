import torch
import numpy as np

class CheckpointedSequential():
    def __init__(self, sequence):
        if isinstance(sequence, list):
            self.sequence = sequence
        elif isinstance(sequence, torch.nn.Sequential):
            self.sequence = list(sequence.children())
        else:
            raise TypeError('CheckpointedSequential: `sequence` must be either '
                    'a list of, or `torch.nn.Sequential` of, torch '
                    'modules/functions, but got type {} instead.'
                    .format(type(sequence))
            )
        self.has_profiled = False

    def profile_sequence(self):
        self.compute_costs = []
        self.memory_costs = []
        self.has_profiled = True

    def solve_optimal_policy(self, M, profile_memory=True, profile_compute=True):
        if not self.has_profiled and (profile_memory or profile_compute):
            self.profile_sequence()
        
        # TODO: Handle setting of uniform costs if no profiling

        # TODO: solve optimal policy
        policy = []

        return policy

    def backprop_sequence(self, policy, inputs, upstream_gradient, callbacks={}):
        pass

