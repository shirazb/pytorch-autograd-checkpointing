import torch
import torch.nn
import numpy as np

import warnings
import weakref

######## CONSTANTS #############################################################

# Set B, C to this value to mark a failed search (strategy does not
# fit in memory).
_COST_SEARCH_FAILURE = -1

# Set D to this value to mark no checkpoint; resort to constant-
# memory quadratic-cost strategy.
_POLICY_CONST_MEM = -1

######## Class #################################################################
class CheckpointSolverFailure(RuntimeError):
    pass

class CheckpointedSequential():
    def __init__(self, sequence, device='cuda'):
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
        num_runs = 4
        self.compute_costs, self.memory_costs = self._profile_compute_and_memory_costs(
            inputs,
            num_runs,
            upstream_gradients)
        self.has_profiled = True


    def solve_optimal_policy(
            self,
            M,
            inputs=None, upstream_gradients=None,      # if profiling
            memory_costs=None, compute_costs=None,     # use given costs
            profile_memory=True, profile_compute=True
    ):
        # If user wants to profile but hasn't already called profiler, do it now.
        if not self.has_profiled and (profile_memory or profile_compute):
            if inputs is None or upstream_gradients is None:
                raise TypeError('CheckpointedSequential.solve_optimal_policy():'
                        ' If profiling and not already called `profile_sequence'
                        '()`, inputs and upstream gradients MUST be provided.'
                )
            self.profile_sequence(inputs, upstream_gradients)

        N = len(self.sequence)

        # If user does not want to profile, if costs provided set to those,
        # otherwise set to uniform.

        if not profile_memory:
            before_memory_costs = self.memory_costs
            self.memory_costs = np.ones((2,N+2), dtype=np.int16) if memory_costs is None else memory_costs
        if not profile_compute:
            before_compute_costs = self.compute_costs
            self.compute_costs = np.ones((2,N+2), dtype=np.int16) if compute_costs is None else compute_costs

        sim_mem, sim_comp, policy = self._solver(M)

        if not profile_memory:
            self.memory_costs = before_memory_costs

        if not profile_compute:
            self.compute_costs = before_compute_costs

        return sim_mem, sim_comp, policy


    # TODO: Callbacks
    def backprop_sequence(self, policy, inputs, upstream_gradients, callbacks={}):
        _check_backward_validity(inputs)

        # Convert to weak reference so will be freed (unless user has some
        # other strong reference to it).
        upstream_gradients = tuple(weakref.ref(b) for b in upstream_gradients)
        M = policy.shape[2]
        N = len(self.sequence)

        # TODO: Callbacks
        return self._backprop_segment(policy, 0, N+1, M, inputs, upstream_gradients)

    # Default is to use given costs; must pass them in.
    def simulate_sequence(
            self, policy,
            memory_costs, compute_costs,
            use_profiled_mem=False, use_profiled_compute=False
    ):
        time, peak = BackpropSimulator(policy, self).sim_sequence(
                use_profiled_mem, use_profiled_compute, memory_costs, compute_costs
        )
        return time, peak



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
            raise CheckpointSolverFailure("CheckpointedSequential: Policy Solver: Not "
                    "enough memory for even just the inputs. Internal budget: %d" % M)

        m_min = self._calc_min_per_layer_usage(N)
        if m_min > M:
            raise CheckpointSolverFailure("CheckpointedSequential: Policy Solver: Not "
                    "enough memory to even run quadratic on sequence. Internal  "
                    "budget: {}, Mem required: {}".format(M, m_min))

        print()
        print('---- LOG: m_min = {}'.format(m_min))
        print('---- LOG: Internal M = {}'.format(M))
        print()

        # TODO: Strip away memory of j<=i and m<=m_min in following arrays.

        # Optimal peak memory cost.
        B = np.empty((N+1, N+2, M), dtype=np.int16)

        # Optimal compute cost.
        C = np.empty((N+1, N+2, M), dtype=np.single)

        # Optimal policy.
        D = np.empty((N+1, N+2, M), dtype=np.int16)

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

                #b = np.sum(self.memory_costs[1, i:i+2])
                b = self.memory_costs[1][i] # profiler includes real b_i+1 in b_i
                c = self.compute_costs[1, i]

                if b > m:
                    B[i, i+1:N+2, m-1] = _COST_SEARCH_FAILURE
                    continue

                B[i, i+1, m-1] = b
                C[i, i+1, m-1] = c

                # Recursive Case: Choose the next tensor to checkpoint.

                # Partially memoize (across j-loop only) max per-layer memory cost
                # that will be incurred by quadratic strategy.
                quad_compute_fs = 0
                quad_compute = quad_compute_fs + c

                quad_peak_fs = 0
                quad_peak = max(quad_peak_fs, b)

                for j in range(i+2, N+2):
                    # Update memoised quadratic costs for subproblem (i, j).
                    quad_compute_fs += self.compute_costs[0, j-1]
                    quad_compute += quad_compute_fs + self.compute_costs[1, j-1]

                    quad_peak_fs = max(quad_peak_fs,
                            (0 if j-2 == i else self.memory_costs[0, j-2]) + self.memory_costs[0, j-1]
                    )
                    quad_peak = max(quad_peak,
                            self.memory_costs[1, j] + quad_peak_fs,
                            #np.sum(self.memory_costs[[0, 1, 1], [j-1, j, j-1]])
                            np.sum(self.memory_costs[[0, 1], [j-1, j-1]]) # profiler includes real b_j in b_j-1
                    )

                    # Initialise variables for k loop.
                    c_min = np.finfo(C.dtype).max
                    sum_forwards_to_checkpoint = 0.0
                    max_mem_per_layer_forwards_to_checkpoint = -1
                    failed = True

                    # Iterate through possible checkpoints f_k.
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
                        b_quad = quad_peak
                        c_quad = quad_compute
                        if b_quad > m:
                            B[i, j:N+2, m-1] = _COST_SEARCH_FAILURE
                            break

                        B[i, j, m-1] = b_quad
                        C[i, j, m-1] = c_quad
                        D[i, j, m-1] = _POLICY_CONST_MEM

            # if m%25 == 0: print('  Done m = {}'.format(m))

        if B[0, N+1, M-1] == _COST_SEARCH_FAILURE:
            raise CheckpointSolverFailure('ASSERTION FAILED: CheckpointedSequential.solve_optimal_policy(): Solver '
                    'failed even though m_min was ok (<= M)!')
            # raise RuntimeError("CheckpointedSequential: Policy Solver: Failed, not enough memory.")

        return (B, C, D)

    # Computes quadratic-strategy memory cost of whole network.
    def _calc_min_per_layer_usage(self, N):
        # initialise for j=1
        peak_fs = 0
        peak = max(peak_fs, np.sum(self.memory_costs[[1,1], [1,0]]))

        # j=2
        peak_fs = max(peak_fs, self.memory_costs[0, 1])
        peak = max(peak,
                peak_fs + self.memory_costs[1, 2],
                np.sum(self.memory_costs[[0,1,1], [1,2,1]])
        )

        # Computed here as is memoised in solver.
        for j in range(2, N+2):
            peak_fs = max(peak_fs, np.sum(self.memory_costs[0, j-2:j]))
            peak = max(peak,
                    self.memory_costs[1, j] + peak_fs,
                    np.sum(self.memory_costs[[0, 1, 1], [j-1, j, j-1]])
            )

        return peak
####### EXECUTOR ###############################################################

    # Invariants: b_j is a weak ref.
    #             b_i (return value) is a weak ref.
    #             f_i hasbeen detached from the proceeding sequence.
    #             All (f|b)_l are tuples of tensors.
    def _backprop_segment(self, policy, i, j, m, f_i, b_j):
        # Base Case: Single layer.
        # if i + 1 == j:
        #     # f_i detached by invariant - backward will not propagate further.
        #     torch.autograd.backward(f_i, b_j)
        #     b_i = _get_tuple_of_weak_grads(f_i)
        #     return b_i
        print('****BEGIN i=%d j=%d m=%d****' % (i, j, m))

        if b_j is None:
            print("what the fuck????")

        if i + 1 == j:
            print("this shouldn't happen")

        # if i + 1 == j:
        #     print('base case: f_%d b_%d' % (i, j))
        #     return b_j

        if i + 2 == j:
            print('base case: f_%d b_%d' % (i, j))
            # run forwards pass of layer i on f_i
            f_i.requires_grad = True
            f_i_plus_1 = self.sequence[i](f_i)
            print('forward  : layer %d on f_%d' % (i+1, i))

            # now backprop back through it with upstream b_j
            torch.autograd.backward(f_i_plus_1, _get_grads_from_weak_refs_tuple(b_j))
            print('backward : layer %d with f_%d b_%d' % (i+1, i+1, j))

            # convert to weak grad to maintain invariant and return
            # need to keep strong ref directly to f_i.grad else weakref doesn't work for some reason
            f_i_grad = f_i.grad
            b_i = (weakref.ref(f_i_grad),)

            return b_i

        k = policy[i, j, m-1]
        print('recursive: f_%d b_%d k=%d' % (i, j, k))

        # FIXME: Surely everything must always require grad.
        # As this is a sequence, any single input to operator i+1 requiring grad
        # means all subsequent outputs require grad, including the output of
        # operator i+1, f_i+1, and so on, to f_k.
        requires_grad = _any_requires_grad(f_i)

        # Base Case: Constant memory / quadtratic compute strategy.
        if (k == _POLICY_CONST_MEM):
            print('quad case: executed')
            return self._backprop_segment_const_mem(i, j, f_i, b_j, requires_grad)

        # TODO: RNG STATE BS

        with torch.no_grad():
            f_k = f_i
            print_i = i # NOTE just have this to print the ix
            for f in range(i+1, k+1):
                # print('f_k.device=%s, layer_f.device=%s' % (f_k.device, next(self.sequence[f].parameters()).device))
                print('forward  : layer %d on f_%d' % (f, print_i))
                f_k = self.sequence[f](f_k)
                print_i += 1

        print('detach   : f_%d (k=%d)' % (print_i - 1, k))
        # f_k = _detach_variable((f_k,), requires_grad)[0]
        # NOTE we're assuming f_k isn't a tuple (sequence is linear, else need bit different code)
        f_k = f_k.detach()
        f_k.requires_grad = requires_grad

        # b_i, b_k, b_j are weak refs by invariant.
        m_r = m - self.memory_costs[0, k]

        # if either is +1 of other, just do it inline, don't recurse
        if k + 1 == j:
            # do base case here
            print('base case: k + 1 == j')
            print('forward  : layer %d on f_%d' % (k+1, k))
            f_k.requires_grad = True
            f_j = self.sequence[k](f_k)
            torch.autograd.backward(f_j, _get_grads_from_weak_refs_tuple(b_j))
            b_k = (weakref.ref(f_k.grad),)
        else:
            # recurse
            # FIXME
            # f_k.grad (b_k) is a weakref to None, because:
            #    - we have to return weakref of b_k
            #    - but, no strong ref to it exists (f_k.grad doesn't count?!)
            b_k = self._backprop_segment(policy, k, j, m_r, f_k, b_j)

        if i + 1 == k:
            # do base case here
            print('base case: i + 1 == k')
            print('forward  : layer %d on f_%d' % (i+1, i))
            f_i.requires_grad = True
            f_k = self.sequence[i](f_i)
            print(f_k)
            print(b_k)
            print(_get_grads_from_weak_refs_tuple(b_k))
            torch.autograd.backward(f_k, _get_grads_from_weak_refs_tuple(b_k))
            b_i = (weakref.ref(f_i.grad),)
        else:
            # recurse
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

            torch.autograd.backward(y, _get_grads_from_weak_refs_tuple(b_prev))
            b_prev = _get_tuple_of_weak_grads(x)

        return b_prev


####### PROFILER ###############################################################
    def _profile_compute_and_memory_costs(self, inputs, num_runs, upstream_gradients):
        """per layer compute and memory costs,
            alpha_f_{0 to N},
            alpha_b_{1 to N+2},
            beta_f_{0 to N},
            beta_b_{1 to N+2}
        """
        device = self.device
        _warm_up_device(device, self.sequence, inputs)

        num_layers = len(self.sequence)

        # begin rows with empty element so we use 1-based indexing
        # e.g. Beta[0][1] is Beta^{f}_{1}
        # e.g.
        #       Five layers 1..5 (N=5). Loss is 6th layer.
        #       f0 is input
        #       b1 is dL/dw1 = layer 1'
        #       fN is output of final layer (layer 5)
        #       fN+1 is output of loss
        #       bN+2 is dL/dL = 1
        #       bN+1 is dL/dL = 1 ????
        #       bN is dL/dw5 = layer 5'
        #   So, if self.sequence includes the loss layer, then length of Alpha
        #   and Beta should be N+2 = 7 = len(self.sequence) + 1 = num_layers + 1.
        #
        #   We want to index Alpha[0][0] = f_0 upto Alpha[0][N+1] = f_{N+1}.
        #    Alpha[1][0] is unused. Alpha[1][1] = b_1. Alpha[1][N+2] = b_{N+2}
        #   Therefore we need Alpha/Beta second dim. to have length N+3 = num_layers+2
        Alpha = np.zeros((2, num_layers+2))
        Beta = np.zeros((2, num_layers+2))

        for k in range(1, num_runs+1):
            # forward pass

            # input
            x = inputs.detach() # device should be 'cpu' with requires_grad=True

            cpu_xs = [x]

            start_mem = torch.cuda.memory_allocated(device)
            x = x.to(device)
            x.requires_grad = True
            Beta[0][0] = abs(start_mem - torch.cuda.memory_allocated(device))
            Alpha[0][0] = 0.0
            start_mem = torch.cuda.memory_allocated(device)

            i = 1
            for layer in self.sequence:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_max_memory_allocated(device)

                # Don't time how long to alloc input/model, but do measure its memory usage.
                layer = layer.to(device)

                torch.cuda.synchronize()
                start_time.record()
                x = layer(x)
                end_time.record()
                torch.cuda.synchronize()

                elapsed = start_time.elapsed_time(end_time)
                mem_used = abs(start_mem - torch.cuda.max_memory_allocated(device))

                cpu_xs.append(x.detach_().to('cpu')) # need to detach in place else get extra tensor in GPU memory

                Alpha[0][i] += (1 / k) * (elapsed - Alpha[0][i])
                Beta[0][i] += (1 / k) * (mem_used - Beta[0][i])

                i += 1
            del x

            # backward pass

            start_mem = torch.cuda.memory_allocated(device)
            j = num_layers + 1 # = N+2
            b = tuple(b_.to(device) for b_ in upstream_gradients)
            Beta[1][j] = abs(start_mem - torch.cuda.memory_allocated(device))
            j -= 1

            # now j = N+1

            # we have added upstream gradients b_N+2 onto device mem
            # but don't want to inlude that in costs of other backwards layers
            start_mem = torch.cuda.memory_allocated(device)

            for layer in reversed(self.sequence):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_max_memory_allocated(device)

                # to compute layer j',
                # we move f_{j-1} to GPU, set required_grad=True, compute f_j
                # then backward on f_j (i.e. compute j') with upstream gradients b_{j+1} -> gives b_j
                # before_mem = torch.cuda.memory_allocated(device)
                x_prev = cpu_xs[j-1].to(device)
                x_prev.requires_grad = True
                # print(x_prev)
                x = layer(x_prev)

                if isinstance(x, torch.Tensor):
                    x = (x,)

                torch.cuda.synchronize()
                start_time.record()
                torch.autograd.backward(x, b)
                end_time.record()
                torch.cuda.synchronize()

                b = x_prev.grad

                elapsed = start_time.elapsed_time(end_time)
                mem_used = abs(start_mem - torch.cuda.max_memory_allocated(device))

                Alpha[1][j] +=(1 / k) * (elapsed - Alpha[1][i])
                Beta[1][j] += (1 / k) * (mem_used - Beta[1][i])

                j -= 1
            del b

        return Alpha, np.ceil(Beta).astype(int)


####### SIMULATOR ###############################################################

class BackpropSimulator():
    def __init__(self, policy, chkseq):
        self.time = 0
        self.policy = policy
        self.cur_mem = 0
        self.peak = 0
        self.chkseq = chkseq

    def _update_time(self, f_or_b, l):
        self.time += self.chkseq.compute_costs[f_or_b, l]

    def _alloc_mem(self, f_or_b, l):
        self.cur_mem += self.chkseq.memory_costs[f_or_b, l]
        self.peak = max(self.peak, self.cur_mem)

    def _free_mem(self, f_or_b, l):
        self.cur_mem -= self.chkseq.memory_costs[f_or_b, l]

    def sim_sequence(self, use_profiled_mem, use_profiled_compute, memory_costs, compute_costs):
        ## Select which costs to use. Save if not profiled
        N = len(self.chkseq.sequence)
        M = self.policy.shape[2]

        if not use_profiled_mem:
            before_memory_costs = self.chkseq.memory_costs
            self.chkseq.memory_costs = np.ones((2,N+2), dtype=np.int16) if memory_costs is None else memory_costs
        if not use_profiled_compute:
            before_compute_costs = self.chkseq.compute_costs
            self.chkseq.compute_costs = np.ones((2,N+2), dtype=np.int16) if compute_costs is None else compute_costs


        # f_0, b_N+1 computed and in mem
        self._update_time(0, 0)
        self._update_time(1, N+1)
        self._alloc_mem(0, 0)
        self._alloc_mem(1, N+1)

        self.sim_segment(0, N+1, M+1)

        # Subcall frees f_0, b_N+1, not b_0
        self._free_mem(1, 0)

        if self.cur_mem != 0:
            print('    Warning: Got self.cur_mem = {} when done.'.format(self.cur_mem))

        ## Restore costs
        if not use_profiled_mem:
            self.chkseq.memory_costs = before_memory_costs

        if not use_profiled_compute:
            self.chkseq.compute_costs = before_compute_costs


        return self.time, self.peak

    def sim_segment(self, i, j, m):
        if i + 1 == j:
            self._update_time(1, i)

            self._free_mem(0, i)
            self._free_mem(1, i+1)
            self._alloc_mem(1, i)
            return

        k = self.policy[i, j, m-1]

        if k == _POLICY_CONST_MEM:
            self.sim_segment_const_mem(i, j, m)
            return

        # compute forwards to k
        for l in range(i+1, k+1):
            self._update_time(0, l)

            # free (f_l-2 + f_l-1) then allocate (f_l-1 + f_l)
            if l-1 != i: self._free_mem(0, l-1) # dont free f_i
            self._alloc_mem(0, l)

        """
        e.g. i=1, k=3.
        have f1, want to compute f3
        f2 = sequence[1](f1)
        f3 = sequence[2](f2)
        Do these forwards whilst holding b_j
        """
        #peak_fs = max(self.memory_costs[0][i+1:k+1]) + self.memory_costs[1][j]
        # XXX need to keep track of before and after tensor for each sequence??

        # we have alloc'd f_k

        m_r = m - self.chkseq.memory_costs[0, k]
        self.sim_segment(k, j, m_r)

        # Subcall frees f_k

        self.sim_segment(i, k, m)

        # Will recursively update time
        #peak_r = self.sim_segment(policy, k, j, m_r)
        #peak_l = self.sim_segment(policy, i, k, m)

        #return max(peak_fs, self.memory_costs[0][i] + peak_r, peak_l)

    def sim_segment_const_mem(self, i, j, m):
        # Memoises this calculation as in the solver

        # j = i + 1
        # peak_fs = 0
        # peak = max(peak_fs + self.memory_costs[1][i+1], self.memory_costs[1][i])

        # for l in range(i+1, j):
        #     peak_fs = max(peak_fs, self.memory_costs[0][l])
        #     peak = max(peak, peak_fs + self.memory_costs[1][l+1], self.memory_costs[1][l])
        print('SIM QUAD i=%d j=%d m=%d' % (i, j, m))

        for p in range(j-1, i-1, -1):
            # Run forwards to p, keeping track of input and output of current
            # layer. Do all but p^th layer without grad (in-place).
            for q in range(i+1, p+1):
                self._update_time(0, q)

                if q-1 != i: self._free_mem(0, q-1)
                self._alloc_mem(0, q)

            self._update_time(1, p)

            self._free_mem(1, p+1)
            self._free_mem(0, p)
            self._alloc_mem(1, p)



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
    return tuple(map(
        lambda x: weakref.ref(x.grad), #if isinstance(x, torch.Tensor) else x,
        xs
    )) # TODO: Is x.grad always a tensor? (think yes)

def _get_grads_from_weak_refs_tuple(xs):
    return tuple(map(
        lambda x: x(),
        xs
    ))

def _any_requires_grad(inputs):
    return any((
            inp.requires_grad if isinstance(inp, torch.Tensor) else False
            for inp in inputs
    ))


def _warm_up_device(device, model=None, inputs=None):
    '''
      Perform some arbitrary computation on device that will be immediately
      discarded to warm up the device to peak performance.
    '''
    (torch.randn(3000, 3000, device=device) * torch.randn(3000, 3000, device=device)).sum()
    torch.cuda.synchronize()

    if model is not None and inputs is not None:
        if isinstance(model, list):
            model = torch.nn.Sequential(*model)
        model = model.to(device)
        for _ in range(10):
            x = inputs.to(device)
            x = model(x)
            x.sum().backward()
