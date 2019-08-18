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

    def profile_sequence(self, inputs, upstream_gradients):
        """
          Dimensions:

          [N+1][N+1][M].
          i: f_0 ... f_N   (size N+1)
          j: b_1 ... b_N+1 (size N+1)
          ==> Use regular indexes, both have size N+2, i is padded at end, 
              j at beginning.

          Boundaries:

          - f_0 is input.
              - Compute = N/A
              - Memory  = size of given input
          - f_N is output.
              - Compute = layer N-1
              - Memory  = sizeof output of layer N-1
          - b_1 is grad input
              - Compute = layer 0'
              - Memory  = size of output of layer 0'
          - b_N+1 is grad output
              - Compute = N/A
              - Memory  = size of given upstream grad
        """
        self.compute_costs = []
        self.memory_costs = []
        self.has_profiled = True

    def solve_optimal_policy(
            self,
            M,
            inputs=None, upstream_gradients=None,
            profile_memory=True, profile_compute=True
    ):
        if not self.has_profiled and (profile_memory or profile_compute):
            if inputs is None or upstream_gradients is None:
                raise TypeError('CheckpointedSequential.solve_optimal_policy():'
                        ' If profiling and not already called `profile_sequence'
                        '()`, inputs and upstream gradients MUST be provided.'
                )
            else:
                self.profile_sequence(inputs, upstream_gradients)

        # TODO: Handle setting of uniform costs if no profiling

        # TODO: solve optimal policy
        policy = []

        return policy

    # TODO: Callbacks
    def backprop_sequence(self, policy, inputs, upstream_gradients, callbacks={}):
        pass

