import torch
import matplotlib.pyplot as plt
import pytorch_autograd_checkpointing as c
#from torch.utils.checkpoint import checkpoint_sequential
from math import sqrt

_DEFAULT_OUTFILE_PREFIX = 'results/'
_DEFAULT_PEAK_MEM_OUTFILE_NAME = 'sqrt_N_checkpointing_peak_mem_vs_N.png'
_DEFAULT_COMPUTE_OUTFILE_NAME = 'sqrt_N_checkpointing_compute_vs_N.png'
_DIM = 1000

def plot_compute_and_memory_costs_for_sqrt_N_checkpointing(
        start_N, end_N, skip_N=1,
        num_runs=3,
        quiet=True,
        graphs=True,
        device='cuda',
        outfile_prefix=_DEFAULT_OUTFILE_PREFIX,
        peak_mem_outfile_name=_DEFAULT_PEAK_MEM_OUTFILE_NAME,
        compute_outfile_name=_DEFAULT_COMPUTE_OUTFILE_NAME
):
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
        mk_baseline_model = lambda: _mk_seq(N, device)
        mk_checkpointed_model = lambda: _mk_seq_with_sqrt_N_segments(N, device)
        
        # def chk():
        #     model = _mk_seq(N, device)
        #     return lambda x: checkpoint_sequential(model, round(sqrt(N)), x) 
        
        # mk_checkpointed_model = chk

        baseline_compute_ms, baseline_peak_mem_mb = _profile(
                mk_baseline_model, num_runs, device
        )
        checkpointed_compute_ms, checkpointed_peak_mem_mb = _profile(
                mk_checkpointed_model, num_runs, device
        )

        compute_baseline.append(baseline_compute_ms)
        compute_results.append(checkpointed_compute_ms)
        memory_baseline.append(baseline_peak_mem_mb)
        memory_results.append(checkpointed_peak_mem_mb)

        if not quiet: print('Done N = {}'.format(N))
    
    if not quiet:
        print('Done')
        print()
        print('Compute (baseline then results):')
        print(compute_baseline)
        print(compute_results)
        print('Memory (baseline then results):')
        print(memory_baseline)
        print(memory_results)

    if graphs:
        _plot_results(
                NS,
                compute_baseline, compute_results,
                memory_baseline, memory_results,
                outfile_prefix, peak_mem_outfile_name, compute_outfile_name
        )

###### HELPERS ######

def _profile(mk_model, num_runs, device):
    _warm_up_device(device)
    
    mean_compute_ms = 0.0
    mean_peak_mem_mb = 0.0

    for k in range(1, num_runs+1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Don't time how long to alloc input/model, but do measure its memory usage.
        torch.cuda.reset_max_memory_allocated()
        model = mk_model()
        x = torch.randn(_DIM, _DIM, device=device, requires_grad=True)
        start.record()

        y = model(x).sum()
        y.backward()

        end.record()
        torch.cuda.synchronize()
        del x, y, model # required?

        elapsed = start.elapsed_time(end)
        peak_mem = float(torch.cuda.max_memory_allocated()) / 1.0e6

        mean_compute_ms += (1 / k) * (elapsed - mean_compute_ms)
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
        *(_layer() for _ in range(N))
    ).to(device)

def _mk_seq_with_sqrt_N_segments(N, device):
    segments = max(1, round(sqrt(N)))
    segment_size = N // segments

    seq = ()

    i = 0
    while i < N - segment_size:
        seq += (
                c.Drop(torch.nn.Sequential(
                    *(_layer() for _ in range(segment_size))
                ), 1)
        ,)
        i += segment_size
    
    while i < N:
        seq += (_layer(),)
        i += 1
    
    return torch.nn.Sequential(*seq).to(device)

def _layer():
    return _ThresholdedLinear(_DIM, _DIM)

class _ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(_ThresholdedLinear, self).__init__()
    
        self.fc = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))

def _plot_results(
        NS,        
        compute_baseline, compute_results,
        memory_baseline, memory_results,
        outfiles_prefix, peak_mem_outfile_name, compute_outfile_name
):    
    metrics = ['Total Wall Clock Time', 'Peak Memory Usage']
    baselines = [compute_baseline, memory_baseline]
    results = [compute_results, memory_results]
    x_label = 'Number Layers, N'
    y_labels = metrics
    title_end = 'for Varying Number Layers With and Without O(sqrt(N)) Checkpointing'
    titles = [m + title_end for m in metrics]
    outfile_paths = [outfiles_prefix + name for name in [compute_outfile_name, peak_mem_outfile_name]]

    for i in range(2):
        _plot_and_write_out_metric(
            NS,
            baselines[i], results[i],
            x_label, y_labels[i], titles[i],
            outfile_paths[i]
        )
        

def _plot_and_write_out_metric(
        xs, ys_baseline, ys_result,
        x_label, y_label, title,
        outfile_path
):
    fig, ax = plt.sublpots(figsize=(15,15))

    ax.plot(xs, ys_baseline, '+-b', label='baseline')
    ax.plot(xs, ys_result, 'x-r', label='with checkpointing')

    ax.legend()
    ax.set_xlabel('x_label')
    ax.set_ylabel('y_label')
    ax.set_title(title)

    plt.savefig(outfile_path, bbox_inches='tight')

