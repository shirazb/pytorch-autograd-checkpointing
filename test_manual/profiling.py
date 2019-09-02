import torch
import torch.nn as nn
import pytorch_autograd_checkpointing as c

class MyLoss(nn.Module):
    def __init__(self, targets, device):
        super(MyLoss, self).__init__()
        self.targets = targets
        self.loss_fn = nn.MSELoss()
        self.device = device

    def forward(self, x):
      return self.loss_fn(x, self.targets.to(self.device))

class ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThresholdedLinear, self).__init__()

        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))

def test_profile_stuff():
  # prepare
  N = 5
  bs = 4
  layer_width = 10
  device = torch.device('cuda:0')

  inp = torch.randn(bs, layer_width)
  targets = torch.randn(bs, layer_width)

  sequence = [ThresholdedLinear(layer_width, layer_width) for _ in range(N)]
  sequence.append(MyLoss(targets, device))

  checkpointed_model = c.CheckpointedSequential(sequence, device)

  upstream_gradients = (torch.tensor(1.).to(device),)

  # act
  checkpointed_model.profile_sequence(inp, upstream_gradients)
  compute_costs = checkpointed_model.compute_costs
  memory_costs = checkpointed_model.memory_costs

  # assert
  assert checkpointed_model.has_profiled == True
  assert compute_costs.shape == (2, N+3)
  assert memory_costs.shape == (2, N+3)
  print("Alpha")
  print(compute_costs)
  print("Beta")
  print(memory_costs)

if __name__ == "__main__":
    test_profile_stuff()


"""
in profiler and solver, everything is given on CPU.
But, in backprop_sequeunce, should be given on GPU (e.g. inputs),
and Loss needs to be a Lambda that moves targets to GPU.
"""
