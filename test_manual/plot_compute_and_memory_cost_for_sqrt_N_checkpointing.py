import torch
import pytorch_autograd_checkpointing as c

from math import sqrt

def plot_compute_and_memory_cost_for_sqrt_N_checkpointing(start_N, end_N, skip_N=1, num_runs=3):
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

    for N in NS:
        baseline_model = _mk_seq(N)
        checkpointed_model = _mk_seq_with_drops(N)


###### HELPERS ######

def _run(model, num_runs, input_dim=10):
    torch.cuda.reset_max_memory_allocated()
    # with profiler

    # do k times and average
    x = torch.randn(input_dim, input_dim, requires_grad=True)
    y = model(x).sum()
    y.backward()

    peak_mem = torch.cuda.max_memory_allocated()
    # get running time

def _mk_seq(N):
    return torch.nn.Sequential(
        _layer() for _ in range(N)
    )

def _mk_seq_with_drops(N):
    segments = round(sqrt(N))
    segment_size = N // segments

    seq = ()

    i = 0
    while i < N - segment_size:
        seq += c.Drop(torch.nn.Sequential(
                _layer() for _ in range(segment_size)
        ), 1)
    
    while i < N:
        seq += _layer()
    
    return torch.nn.Sequential(seq)

def _layer(dim=10):
    return _ThresholdedLinear(dim, dim)

class _ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(_ThresholdedLinear, self).__init__()
    
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return torch.functional.relu(self.fc(x))
    