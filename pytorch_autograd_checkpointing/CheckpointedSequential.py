import torch
import numpy as np

import warnings

class CheckpointedSequential():
    def __init__(self, sequence):
        if not torch.cuda.is_available():
            warnings.warn('CheckpointedSequential: Constructing when CUDA is '
                    'not available, though current implementation assumes CUDA '
                    'and thus may fail.'
            )
        
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

          [N+2][M].
          i: f_0 ... f_N   (size N+1)
          j: b_1 ... b_N+1 (size N+1)
          ==> Use regular indexes, size N+2, i is padded at end, 
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
        _, policy = _solver(len(self.sequence), M, self.compute_costs, self.memory_costs)

        return policy

    # TODO: Callbacks
    def backprop_sequence(self, policy, inputs, upstream_gradients, callbacks={}):
        pass



######## CONSTANTS #############################################################

# Set B, C to this value to mark a failed search (strategy does not
# fit in memory).
_COST_SEARCH_FAILURE = -1

# Set D to this value to mark no checkpoint; resort to constant-
# memory quadratic-cost strategy.
_POLICY_CONST_MEM = -1

####### POLICY SOLVER ##########################################################

"""
solve_optimal_policy(
        N: int,
        M: int,
        compute_costs: float[2, N+1],
        memory_costs: int[2, N+1],
        logger: Logger
) --> (C: float[N, N, M], D: int[N, N, M])

Arguments:
    N: Number layers of the network. This means we have N forward
       tensors to compute, given the input; and N backwards to
       compute given the targets.
    M: Total memory budget. This should not include the input or
       targets.
    compute_costs: [(0|1), l] represents computational cost of
                   forward and backward tensor l, respectively.
                   This includes the inputs and targets tensors.
    memory_costs: [(0|1), l] represents memory cost of forward
                  and backward tensor l, respectively. This
                  includes the inputs and targets tensors.
    logger: A logger.

Returns:
    C: `C[i, j, m]` is the optimal computational cost of performing one
       pass of backpropagation on the subsequence `[i, j]`; that is,
       given the forward tensor `i` and the backward tensor `j`,
       compute the backward tensor `i`.

    D: `D[i, j, m-1]` is the corresponding optimal policy that
       specifies what node to next checkpoint when computing
       subsequence `[i, j]`. From this, an optimal execution plan can
       be constructed.
"""
def _solver(N, M, compute_costs, memory_costs):
    # NB: [i, j, m-1] indexes subsequence [i, j] and memory budget m.
    
    # TODO: Optimise j dimension to not waste N/2 memory, as always j>i.
    
    # first dimension only N+1
    # compute_costs: float[2, N+2]
    # memory_costs: int[2, N+2]
    
    # Largest subproblem [0, N+2-1] means given inputs already in memory,
    # so the max memory budget available does not include that memory.
    M = M - memory_costs[0, 0]
    
    if M < 1:
        raise RuntimeError("Not enough memory. Internal budget: ", M) # TODO
        
    # Optimal peak memory cost.
    B = np.empty((N+1, N+2, M), dtype=np.int16)
    
    # Optimal compute cost.
    C = np.empty((N+1, N+2, M), dtype=np.single)
        
    # Optimal policy.
    D = np.empty((N+1, N+2, M), dtype=np.int16)
    
    #TODO: m_min = calc_min_per_layer_usage(memory_costs, N)
    m_min = 1
    # must be <= M
    
    for m in range(m_min, M+1):
        #logger.verbose("m =", m)
        # Traverse the possible subsequences in an order such that
        # we have already solved all of its subproblems:
        # [N, N+1],
        # [N-1, N], [N-1, N+1],
        # [N-2, N-1], [N-2, N], [N-2, N+1],
        # ...
        # [0, 1], [0, 2],    ...   , [0, N+1]
        for i in range(N, -1, -1):
            
            # Base Case: [i, i+1, m-1]
            
            b = np.sum(memory_costs[1, i:i+2])
            c = compute_costs[1, i]
            
            if b > m:
                B[i, i+1:N+2, m-1] = _COST_SEARCH_FAILURE
                continue
            
            B[i, i+1, m-1] = b
            C[i, i+1, m-1] = c
            
            # Recursive Case: Choose the next tensor to checkpoint.
            
            # Partially memoize (across j-loop only) max per-layer memory cost 
            # that will be incurred by quadratic strategy.
            max_mem_per_layer_seq = b
            
            for j in range(i+2, N+2):
                # Update for next j.
                max_mem_per_layer_seq = max(
                    max_mem_per_layer_seq,
                    
                    # Compute this forward from the last, not including f_i.
                    (0 if j-2 == i else memory_costs[0, j-2]) + memory_costs[0, j-1],
                    
                    # Compute backward.
                    np.sum(memory_costs[np.array([0, 1, 1]), np.array([j-1, j, j-1])])
                )
                
                # Initialise
                c_min = np.finfo(C.dtype).max
                sum_forwards_to_checkpoint = 0.0
                max_mem_per_layer_forwards_to_checkpoint = -1
                failed = True

                # Iterate through possible checkpoints f_k
                for k in range(i+1, j):
                    # Accumulate cost of computing forwards to k.
                    sum_forwards_to_checkpoint += compute_costs[0, k]
                    
                    # Update peak memory whilst computing forwards to k.
                    max_mem_per_layer_forwards_to_checkpoint = max(
                        max_mem_per_layer_forwards_to_checkpoint,
                        
                        # Do not include f_i.
                        (0 if k-1 == i else memory_costs[0, k-1]) + memory_costs[0, k]
                    )
                    
                    # Pre-check the stages of this strategy for failure.
                    
                    # 1: Forwards to k in-place whilst holding b_j.
                    
                    m_b_j = memory_costs[1, j]
                    b_fs = max_mem_per_layer_forwards_to_checkpoint + m_b_j
                    
                    if (b_fs > m):
                        continue
                    
                    # 2, 3: Right and left subproblems
                    
                    m_f_k = memory_costs[0, k]                    
                    m_r = m - m_f_k
                    m_l = m
                    
                    if (m_r < m_min or m_l < m_min):
                        continue
                    
                    b_r = B[k, j, m_r-1]
                    b_l = B[i, k, m_l-1]
                    
                    if (b_r == _COST_SEARCH_FAILURE or b_l == _COST_SEARCH_FAILURE):
                        continue                    

                    # Get peak memory usage across this entire strategy's execution.
                    b_k = max(
                        b_fs,
                        m_f_k + b_r,
                        b_l
                    )
                        
                    if (b_k > m):
                        continue
                    
                    # Success, a strategy works!
                    failed = False
                    
                    c_k = sum_forwards_to_checkpoint + C[k, j, m_r-1] + C[i, k, m_l-1]
                    
                    if c_k < c_min:
                        c_min = c_k
                        B[i, j, m-1] = b_k
                        C[i, j, m-1] = c_k
                        D[i, j, m-1] = k
            
                ################################################
                
                # All possible checkpointing strategies failed.
                
                # If the max per-layer size of the subsequence fits
                # into memory, we can do quadratic, else we have failed.
                if failed:
                    b_quad = max_mem_per_layer_seq
                    if b_quad > m:
                        B[i, j:N+2, m-1] = _COST_SEARCH_FAILURE
                        break

                    # Base case: constant memory, quadratic cost. C[i, j] is:
                    # sum k=1..j-i: k * compute_f_(j-k) +
                    #               compute_b_(j-k)
                    acc = 0.0
                    for k in range(1, j-i+1):
                        acc += k * compute_costs[0, j-k] + compute_costs[1, j-k]

                    B[i, j, m-1] = b_quad
                    C[i, j, m-1] = acc
                    D[i, j, m-1] = _POLICY_CONST_MEM
                    
    if B[0, N+1, M-1] == _COST_SEARCH_FAILURE:
        raise RuntimeError('Failed to solve') # TODO
    
    return (C, D)

################################################################################