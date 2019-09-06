import numpy as np
import torch
import pytorch_autograd_checkpointing as c

_DIM = 100

def warm_up_device(device):
    '''
      Perform some arbitrary computation on device that will be immediately
      discarded to warm up the device to peak performance.
    '''
    (torch.randn(3000, 3000, device=device) * torch.randn(3000, 3000, device=device)).sum()

def mk_seq(N, device, dim=_DIM):
    return torch.nn.Sequential(
        *(layer(dim) for _ in range(N))
    ).to(device)

def layer(dim=_DIM):
    return ThresholdedLinear(dim, dim)

class ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThresholdedLinear, self).__init__()

        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))

class SumLayer(torch.nn.Module):
    def forward(self, x):
        return x.sum()

def mk_checkpointed_densenet(
        densenet,
        loss=SumLayer()
):
    modules = [module for k, module in densenet._modules.items()][0]

    assert isinstance(modules, torch.nn.Sequential), 'run_solver_densenet(): modules was not a torch.nn.Sequential'

    modules.add_module('fake_loss', loss)
    
    return c.CheckpointedSequential(modules)

def bucket_costs(
        model_chkpter,
        bucket_size,
        log=False
):
    memory_costs = model_chkpter.memory_costs
    memory_costs = np.ceil(memory_costs / float(bucket_size)).astype(int)
    compute_costs = model_chkpter.compute_costs 

    if log:
        print('Costs: (m,c)')
        print(memory_costs)
        print(compute_costs)
    
    return compute_costs, memory_costs

def prof_model(
        model_chkpter,
        x, b,
        bucket_size=int(3e6),
        log=False
):
    model_chkpter.profile_sequence(x, b)
    
    return bucket_costs(
        model_chkpter,
        bucket_size,
        log
    )

def solve_policy_using_costs(
        model_chkpter,
        compute_costs,
        memory_costs,
        M,
        bucket_size,
        budget_leeway=0.2
):
    budget = budget_bucketed_with_leeway(M, bucket_size, budget_leeway)

    return model_chkpter.solve_optimal_policy(
        budget, 
        compute_costs=compute_costs, memory_costs=memory_costs,
        profile_compute=False, profile_memory=False
    )

def prof_and_solve_policy(
        model_chkpter,
        x, b,
        M,
        bucket_size,
        budget_leeway=0.2,
        log=False
):
    compute_costs, memory_costs = prof_model(model_chkpter, x, b, bucket_size, log)
    
    return solve_policy_using_costs(
        model_chkpter,
        compute_costs, memory_costs,
        M,
        bucket_size, budget_leeway
    )

def budget_bucketed_with_leeway(budget, bucket_size, leeway):
    return int((budget // bucket_size) * (1 - leeway))

def densenet_dummy_input(batch_size):
    return torch.randn(batch_size, 3, 224, 224).fill_(1.0)
