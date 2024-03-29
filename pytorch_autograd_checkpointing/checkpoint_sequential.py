from __future__ import absolute_import, division, print_function, unicode_literals
import weakref

import numpy as np
import torch
import warnings

def detach_variable(inputs, requires_grad):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states):
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def _get_tuple_of_weak_grads(xs):
    return tuple(map(xs,
            lambda x: weakref.ref(x.grad) if isinstance(x, torch.Tensor) else x 
    )) # TODO: Is x.grad always a tensor? (think yes)

def _any_requires_grad(inputs):
    return any((
            inp.requires_grad if isinstance(inp, torch.Tensor) else False
            for inp in inputs
    ))

def run_sequence(functions, D, N, M, compute_costs, memory_costs, *args):
    # Invariants: b_j is a weak ref.
    #             b_i (return value) is a weak ref.
    #             f_i has been detached from the proceeding sequence.
    #             All (f|b)_l are tuples of tensors.
    def backprop_segment(i, j, m, f_i, b_j):       
        # Base Case: Single layer.
        if i + 1 == j:
            # f_i detached by invariant - backward will not propagate further.
            torch.autograd.backward(f_i, b_j)
            b_i = _get_tuple_of_weak_grads(f_i)
            return b_i
        
        k = D[i, j, m-1]

        # FIXME: Surely everything must always require grad.
        # As this is a sequence, any single input to operator i+1 requiring grad
        # means all subsequent outputs require grad, including the output of
        # operator i+1, f_i+1, and so on, to f_k.
        requires_grad = _any_requires_grad(f_i)

        # Base Case: Constant memory / quadtratic compute strategy.
        if (k == POLICY_CONST_MEM):
            return backprop_segment_const_mem(i, j, f_i, b_j, requires_grad)
        
        # TODO: RNG STATE BS

        with torch.no_grad():
            f_k = f_i
            for f in range(i+1, k+1):
                f_k = functions[f](f_k)
        
        f_k = detach_variable(f_k, requires_grad)

        # b_i, b_k, b_j are weak refs by invariant. 
        m_r = m - memory_costs[0, k]
        b_k = backprop_segment(k, j, m_r, f_k, b_j)
        b_i = backprop_segment(i, k, m, f_i, b_k)

        return b_i

    def backprop_segment_const_mem(i, j, f_i, b_j, requires_grad):
        # Compute in-place forwards f_i -> f_p and backward b_p,
        # for p = j-1 down to i.
        
        b_prev = b_j

        for p in range(j-1, i-1):
            # Run forwards to p, keeping track of input and output of current
            # layer. Do all but p^th layer without grad (in-place).
            x = f_i
            y = x
            with torch.no_grad():
                for q in range(i+1, p):
                    x = y
                    y = functions[q](x)

            x = detach_variable(y, requires_grad)
            y = functions[p](x)
            
            torch.autograd.backward(y, b_prev)
            b_prev = _get_tuple_of_weak_grads(x)

        return b_prev                

    ######## BEGIN run_sequence ########

    check_backward_validity(args)

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # inputs = detach_variable(args, True)
    # grad_outputs = _get_grads() make ones of output dim of last function
    # M = M - sizeof(f_0 and b_N+1)

    # return backprop_segment(0, N+1, M, inputs, grad_outputs)

#####################################################
LOG_LEVEL_VERBOSE = 2
LOG_LEVEL_QUIET = 1
LOG_LEVEL_NONE = 0

class Logger:
    def __init__(self, level=LOG_LEVEL_QUIET):
        self.level = level
    
    def _log(self, msg, *args, **kwargs):
        print(msg, *args, **kwargs)
    
    def quiet(self, msg, *args, **kwargs):
        if (self.level >= LOG_LEVEL_QUIET):
            self._log(msg, *args, **kwargs)
    
    def verbose(self, msg, *args, **kwargs):
        if (self.level >= LOG_LEVEL_VERBOSE):
            self._log(msg, *args, **kwargs)
            

###############################################################

# Set B, C to this value to mark a failed search (strategy does not
# fit in memory).
COST_SEARCH_FAILURE = -1

# Set D to this value to mark no checkpoint; resort to constant-
# memory quadratic-cost strategy.
POLICY_CONST_MEM = -1

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
def solve_optimal_policy(N, M, compute_costs, memory_costs, logger):
    # NB: [i, j, m-1] indexes subsequence [i, j] and memory budget m.
    
    # TODO: Optimise j dimension to not waste N/2 memory, as always j>i.
    
    # first dimension only N+1
    # compute_costs: float[2, N+2]
    # memory_costs: int[2, N+2]
    
    # Largest subproblem [0, N+2-1] means given inputs already in memory,
    # so the max memory budget available does not include that memory.
    M = M - memory_costs[0, 0]
    
    if M < 1:
        logger.quiet("Not enough memory. Internal budget: ", M)
        return (1, None, None)
    
    logger.verbose("Internal Memory Budget: ", M)
    
    # Optimal peak memory cost.
    B = np.empty((N+1, N+2, M), dtype=np.int16)
    
    # Optimal compute cost.
    C = np.empty((N+1, N+2, M), dtype=np.single)
        
    # Optimal policy.
    D = np.empty((N+1, N+2, M), dtype=np.int16)
    
    #TODO: m_min = calc_min_per_layer_usage(memory_costs, N)
    m_min = 1
    # must be <= M
    
    c_lb = np.min(compute_costs[1, :])
    
    logger.verbose("Cost Lower Bound: {}".format(np.sum(compute_costs) - np.sum(compute_costs[np.array([0, 0, 1]), np.array([0, N+1, N+1])])))
    logger.verbose(calc_upper_bound(N, compute_costs))
    
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
                B[i, i+1:N+2, m-1] = COST_SEARCH_FAILURE
                continue
            
            if m == M and c < c_lb:
                logger.verbose("Base Case: Setting C[{}, {}, {}] to {}".format(i, i+1, m, c))
            
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
                    
                    if (b_r == COST_SEARCH_FAILURE or b_l == COST_SEARCH_FAILURE):
                        continue                    

                    # Get peak memory usage across this entire strategy's execution.
                    b_k = max(
                        b_fs,
                        m_f_k + b_r,
                        b_l
                    )
                        
                    if (b_k > m):
                        logger.verbose("B[{}, {}, {}]: b_k cost failed".format(i, j, m))
                        continue
                    
                    # Success, a strategy works!
                    failed = False
                    
                    c_k = sum_forwards_to_checkpoint + C[k, j, m_r-1] + C[i, k, m_l-1]
                    
                    if c_k < c_min:
                        c_min = c_k
                        B[i, j, m-1] = b_k
                        C[i, j, m-1] = c_k
                        
                        if m == M and c_k < c_lb:
                            logger.verbose("Rec Case: Setting C[{}, {}, {}] to {}".format(i, j, m, c_k))
                            logger.verbose("    sum_fowards =", sum_forwards_to_checkpoint)
                            logger.verbose("    right=", C[k, j, m_r-1])
                            logger.verbose("    left=", C[i, k, m_l-1])
                        
                        D[i, j, m-1] = k
            
                ################################################
                
                # All possible checkpointing strategies failed.
                
                # If the max per-layer size of the subsequence fits
                # into memory, we can do quadratic, else we have failed.
                if failed:
                    b_quad = max_mem_per_layer_seq
                    if b_quad > m:
                        if m == M:
                            logger.verbose("Quad Case: Failing C[{}, {}, {}]".format(i, j, m))
                        B[i, j:N+2, m-1] = COST_SEARCH_FAILURE
                        break

                    # Base case: constant memory, quadratic cost. C[i, j] is:
                    # sum k=1..j-i: k * compute_f_(j-k) +
                    #               compute_b_(j-k)
                    acc = 0.0
                    for k in range(1, j-i+1):
                        acc += k * compute_costs[0, j-k] + compute_costs[1, j-k]

                    B[i, j, m-1] = b_quad
                    C[i, j, m-1] = acc
                    D[i, j, m-1] = POLICY_CONST_MEM
                    #if m == M and acc < c_lb:
                    logger.verbose("Quad Case: Setting C[{}, {}, {}] to {}".format(i, j, m, acc))
        
                if m == M and B[i, j, m-1] != COST_SEARCH_FAILURE and C[i, j, m-1] < c_lb:
                    logger.verbose("C[{}, {}, {}] < c_lb".format(i, j, m))
    
    err = 0 if B[0, N+1, M-1] != COST_SEARCH_FAILURE else 1
    
    return (err, C, D)

def calc_upper_bound(N, compute_costs):
    acc = 0.0
    for k in range(0, N+1):
        acc += (k+1) * compute_costs[0, N-k] + compute_costs[1, N-k]
    
    return "Cost Upper Bound: {}".format(acc)
