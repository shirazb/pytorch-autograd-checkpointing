import torch
import pytorch_autograd_checkpointing as c

from math import sqrt

def plot_compute_and_memory_costs_for_sqrt_N_checkpointing(start_N, end_N, skip_N=1, num_runs=3, device='cuda'):
    '''
      1. check cuda
      2. for N in given range:
      3.   model of N layers with and without sqrt(N) checkpointing encoded
      4.   train k steps 
      5.   avg peak memory and compute cost
      6. plot results:
      7.   graph memory against N also showing 2sqrt(N) of baseline model
      8.   graph compute against N also showing 2N of baseline model
    '''
    assert torch.cuda.is_available(), (
            'Test Failed: plot_compute_and_memory_cost_for_sqrt_N_checkpointing\n'
            '    CUDA not available'
    )

    NS = range(start_N, end_N, skip_N)

    compute_baseline = []
    compute_results = []
    memory_baseline = []
    memory_results = []

    # Profile compute and memory for checkpointed and baseline model, for each N
    for N in NS:
        baseline_model = _mk_seq(N, device)
        checkpointed_model = _mk_seq_with_drops(N, device)

        baseline_compute_ms, baseline_peak_mem_mb = _profile(
                baseline_model, num_runs, device
        )
        checkpointed_compute_ms, checkpointed_peak_mem_mb = _profile(
                checkpointed_model, num_runs, device
        )

        compute_baseline.append(baseline_compute_ms)
        compute_results.append(checkpointed_compute_ms)
        memory_baseline.append(baseline_peak_mem_mb)
        memory_results.append(checkpointed_peak_mem_mb)
    
    print(compute_baseline)
    print(memory_baseline)

###### HELPERS ######

def _profile(model, num_runs, device, input_dim=10):
    _warm_up_device(device)
    
    mean_compute_ms = 0.0
    mean_peak_mem_mb = 0.0

    for k in range(1, num_runs+1):
        # Record compute and peak mem
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_time=True)

        torch.cuda.reset_max_memory_allocated()
        start.record()

        x = torch.randn(input_dim, input_dim, device=device, requires_grad=True)
        y = model(x).sum()
        y.backward()

        del x, y

        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)
        peak_mem = float(torch.cuda.max_memory_allocated()) / 1.0e6

        mean_compute += (1 / k) * (elapsed - mean_compute)
        mean_peak_mem_mb += (1 / k) * (peak_mem - mean_peak_mem_mb)
    
    return mean_compute_ms, mean_peak_mem_mb

def _warm_up_device(device):
    '''
      Perform some arbitrary computation on device that will be immediately
      discarded to warm up the device to peak performance.
    '''
    (torch.randn(3000, 3000, device=device) * torch.randn(3000, 3000, device=device)).sum()

def _mk_seq(N, device):
    return torch.nn.Sequential(
        *(_layer(device) for _ in range(N))
    ).to(device)

def _mk_seq_with_drops(N, device):
    segments = round(sqrt(N))
    segment_size = N // segments

    seq = ()

    i = 0
    while i < N - segment_size:
        seq += c.Drop(torch.nn.Sequential(
                *(_layer(device) for _ in range(segment_size))
        ), 1)
    
    while i < N:
        seq += _layer(device)
    
    return torch.nn.Sequential(*seq).to(device)

def _layer(dim=10):
    return _ThresholdedLinear(dim, dim)

class _ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(_ThresholdedLinear, self).__init__()
    
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return torch.functional.relu(self.fc(x))
    