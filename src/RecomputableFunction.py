from __future__ import absolute_import, division, print_function, unicode_literals
import warnings

import torch


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
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


class RecomputableFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, recomp_depth, preserve_rng_state, *args):
        # if recomp_depth == 0:
        #    checkpoint the input
        #    save for backward
        # with no grad run the function and return the output
        
        # FIXME: Some of this is unecessarily run every recomputation, e.g.
        #        `check_backward_validity`.
        #        Is the cuda rng stuff required each forward or only the last?

        if recomp_depth == 0: check_backward_validity(args)
        
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        # Only checkpoint input if this is the last drop.
        if recomp_depth == 0:
            ctx.save_for_backward(*args)
        
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # recover checkpoint from ctx
        # run function with grad
        # run backward
        
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        
        # Recover checkpointed input.
        inputs = ctx.saved_tensors
        
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrouding state
        # when we're done.
        rng_devices = []
        
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            
            breakpoint

            # Recompute the forward.
            print()
            print('inputs: ', inputs)
            detached_inputs = detach_variable(inputs)
            print('detached: ', detached_inputs)
            print()
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        
        # Run the backward.
        torch.autograd.backward(outputs, args)
        
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        
        return (None, None, None) + grads
