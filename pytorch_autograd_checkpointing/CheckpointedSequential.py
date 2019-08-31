import torch
import numpy as np

import warnings
import weakref

from .pytorch_modelsize import SizeEstimator

######## CONSTANTS #############################################################

# Set B, C to this value to mark a failed search (strategy does not
# fit in memory).
_COST_SEARCH_FAILURE = -1

# Set D to this value to mark no checkpoint; resort to constant-
# memory quadratic-cost strategy.
_POLICY_CONST_MEM = -1

######## Class #################################################################

class CheckpointedSequential():
    def __init__(self, sequence, device):
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
                    'modules and functions, but got type {} instead.'
                    .format(type(sequence))
            )

        self.has_profiled = False
        self.device = device
        # NOTE: need to enforce all modules are on CPU when init'd?? (otherwise profiling correctly hard)


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
              - Compute = 0, is given
              - Memory  = size of given input
          - f_N is output.
              - Compute = layer N-1
              - Memory  = sizeof output of layer N-1
          - b_1 is grad input
              - Compute = layer 0'
              - Memory  = size of output of layer 0'
          - b_N+1 is grad output
              - Compute = 0, is given
              - Memory  = size of given upstream grad
        """
        num_runs = 5
        self.compute_costs, self.memory_costs = self._profile_compute_memory_costs(
            num_runs,
            inputs,
            upstream_gradients)
        self.has_profiled = True


    def _profile_compute_memory_costs(self, inputs, num_runs, upstream_gradients):
        """per layer compute and memory costs,
            alpha_f_{0 to N},
            alpha_b_{1 to N+1},
            beta_f_{0 to N},
            beta_b_{1 to N+1}
        """
        device = self.device
        _warm_up_device(device)

        num_layers = len(self.sequence)

        # begin rows with empty element so we use 1-based indexing
        # e.g. Beta[0][1] is β^{f}_{1}
        Alpha = np.zeros((2, N+2))
        Beta = np.zeros((2, N+2))

        # input
        x = inputs # device should be 'cpu' with requires_grad=True
        x.requires_grad = True

        for k in range(1, num_runs+1):
            # forward pass

            cpu_xs = [x]

            start_mem = torch.cuda.memory_allocated(device)
            x = x.to(device)
            Beta[0][0] = abs(start_mem - torch.cuda.memory_allocated(device))
            Alpha[0][0] = 0.0

            i = 1
            for layer in self.sequence:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                # Don't time how long to alloc input/model, but do measure its memory usage.
                # torch.cuda.reset_max_memory_allocated()
                layer = layer.to(device)
                start_time.record()

                x = layer(x).sum()
                cpu_xs.append(x.detach().to('cpu'))

                end_time.record()
                torch.cuda.synchronize() # NOTE why this required?
                # del x, y, model # required?

                elapsed = start_time.elapsed_time(end_time)
                # peak_mem = float(torch.cuda.max_memory_allocated(device)) / 1.0e6
                mem_used = abs(start_mem - torch.cuda.memory_allocated(device))

                Alpha[0][i] += (1 / k) * (elapsed - Alpha[0][i])
                Beta[0][i] += (1 / k) * (mem_used - Beta[0][i])

                i += 1
            del x

            # backward pass

            j = N + 1
            b = (b_i.to(device) for b_i in upstream_gradients)
            Beta[1][j] = abs(start_mem - torch.cuda.memory_allocated(device))
            j -= 1

            for layer in reversed(self.sequence):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                x = c_xs[j].to(device)
                x.requires_grad = True
                start_time.record()
                before_mem = torch.cuda.memory_allocated(device)

                torch.autograd.backward(x, b)
                b = _get_tuple_of_weak_grads(layer)

                end_time.record()
                torch.cuda.synchronize()

                elapsed = start_time.elapsed_time(end_time)
                mem_used = abs(before_mem - torch.cuda.memory_allocated(device))

                Alpha[1][j] +=(1 / k) * (elapsed - Alpha[1][i])
                Beta[1][j] += (1 / k) * (mem_used - Beta[1][i])
            del b

        return Alpha, Beta


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
        _, policy = self._solver(M)

        return policy


    # TODO: Callbacks
    def backprop_sequence(self, policy, inputs, upstream_gradients, callbacks={}):
        _check_backward_validity(inputs)

        # Convert to weak reference so will be freed (unless user has some
        # other strong reference to it).
        upstream_gradients = _get_tuple_of_weak_grads(upstream_gradients)
        M = policy.shape[2]
        N = len(self.sequence)

        # TODO: Callbacks
        self._backprop_segment(policy, 0, N+1, M, inputs, upstream_gradients)


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
    def _solver(self, M):
        N = len(self.sequence)

        # NB: [i, j, m-1] indexes subsequence [i, j] and memory budget m.

        # TODO: Optimise j dimension to not waste N/2 memory, as always j>i.

        # first dimension only N+1
        # compute_costs: float[2, N+2]
        # memory_costs: int[2, N+2]

        # Largest subproblem [0, N+2-1] means given inputs already in memory,
        # so the max memory budget available does not include that memory.
        M = M - self.memory_costs[0, 0]

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
            # Traverse the possible subsequences in an order such that
            # we have already solved all of its subproblems:
            # [N, N+1],
            # [N-1, N], [N-1, N+1],
            # [N-2, N-1], [N-2, N], [N-2, N+1],
            # ...
            # [0, 1], [0, 2],    ...   , [0, N+1]
            for i in range(N, -1, -1):

                # Base Case: [i, i+1, m-1]

                b = np.sum(self.memory_costs[1, i:i+2])
                c = self.compute_costs[1, i]

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
                        (0 if j-2 == i else self.memory_costs[0, j-2]) + self.memory_costs[0, j-1],

                        # Compute backward.
                        np.sum(self.memory_costs[[0, 1, 1], [j-1, j, j-1]])
                    )

                    # Initialise
                    c_min = np.finfo(C.dtype).max
                    sum_forwards_to_checkpoint = 0.0
                    max_mem_per_layer_forwards_to_checkpoint = -1
                    failed = True

                    # Iterate through possible checkpoints f_k
                    for k in range(i+1, j):
                        # Accumulate cost of computing forwards to k.
                        sum_forwards_to_checkpoint += self.compute_costs[0, k]

                        # Update peak memory whilst computing forwards to k.
                        max_mem_per_layer_forwards_to_checkpoint = max(
                            max_mem_per_layer_forwards_to_checkpoint,

                            # Do not include f_i.
                            (0 if k-1 == i else self.memory_costs[0, k-1]) + self.memory_costs[0, k]
                        )

                        # Pre-check the stages of this strategy for failure.

                        # 1: Forwards to k in-place whilst holding b_j.

                        m_b_j = self.memory_costs[1, j]
                        b_fs = max_mem_per_layer_forwards_to_checkpoint + m_b_j

                        if (b_fs > m):
                            continue

                        # 2, 3: Right and left subproblems

                        m_f_k = self.memory_costs[0, k]
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
                            acc += k * self.compute_costs[0, j-k] + self.compute_costs[1, j-k]

                        B[i, j, m-1] = b_quad
                        C[i, j, m-1] = acc
                        D[i, j, m-1] = _POLICY_CONST_MEM

        if B[0, N+1, M-1] == _COST_SEARCH_FAILURE:
            raise RuntimeError('Failed to solve') # TODO

        return (C, D)

####### EXECUTOR ###############################################################

    # Invariants: b_j is a weak ref.
    #             b_i (return value) is a weak ref.
    #             f_i has been detached from the proceeding sequence.
    #             All (f|b)_l are tuples of tensors.
    def _backprop_segment(self, policy, i, j, m, f_i, b_j):
        # Base Case: Single layer.
        if i + 1 == j:
            # f_i detached by invariant - backward will not propagate further.
            torch.autograd.backward(f_i, b_j)
            b_i = _get_tuple_of_weak_grads(f_i)
            return b_i

        k = policy[i, j, m-1]

        # FIXME: Surely everything must always require grad.
        # As this is a sequence, any single input to operator i+1 requiring grad
        # means all subsequent outputs require grad, including the output of
        # operator i+1, f_i+1, and so on, to f_k.
        requires_grad = _any_requires_grad(f_i)

        # Base Case: Constant memory / quadtratic compute strategy.
        if (k == _POLICY_CONST_MEM):
            return self._backprop_segment_const_mem(i, j, f_i, b_j, requires_grad)

        # TODO: RNG STATE BS

        with torch.no_grad():
            f_k = f_i
            for f in range(i+1, k+1):
                f_k = self.sequence[f](f_k)

        f_k = _detach_variable(f_k, requires_grad)

        # b_i, b_k, b_j are weak refs by invariant.
        m_r = m - self.memory_costs[0, k]
        b_k = self._backprop_segment(policy, k, j, m_r, f_k, b_j)
        b_i = self._backprop_segment(policy, i, k, m, f_i, b_k)

        return b_i


    def _backprop_segment_const_mem(self, i, j, f_i, b_j, requires_grad):
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
                    y = self.sequence[q](x)

            x = _detach_variable(y, requires_grad)
            y = self.sequence[p](x)

            torch.autograd.backward(y, b_prev)
            b_prev = _get_tuple_of_weak_grads(x)

        return b_prev

####### NON-INSTANCE EXECUTOR HELPERS ##########################################

def _detach_variable(inputs, requires_grad):
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


def _check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def _get_device_states(*args):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                            if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def _set_device_states(devices, states):
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


def _warm_up_device(device):
    '''
      Perform some arbitrary computation on device that will be immediately
      discarded to warm up the device to peak performance.
    '''
    (torch.randn(3000, 3000, device=device) * torch.randn(3000, 3000, device=device)).sum()
    torch.cuda.synchronize()
