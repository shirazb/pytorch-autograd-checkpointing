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


class FatMiddleLinear(nn.Module):
    def __init__(self, thin_dim, fat_dim):
        super(FatMiddleLinear, self).__init__()
        self.fcs = nn.Sequential(
            ThresholdedLinear(thin_dim, thin_dim),
            ThresholdedLinear(thin_dim, fat_dim),
            ThresholdedLinear(fat_dim, fat_dim),
            ThresholdedLinear(fat_dim, thin_dim),
            ThresholdedLinear(thin_dim, thin_dim)
        )

    def forward(self, x):
        return self.fcs(x)


def test_profile_stuff_1():
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


def test_profile_stuff_2():
  # prepare
  N = 11
  bs = 32
  in_width = 64
  out_width = 10
  device = torch.device('cuda:0')

  inp = torch.randn(bs, in_width)
  targets = torch.randn(bs, out_width)

  sequence = [
      ThresholdedLinear(in_width, 64),
      ThresholdedLinear(64, 128),
      ThresholdedLinear(128, 128),
      ThresholdedLinear(128, 128),
      ThresholdedLinear(128, 256),
      ThresholdedLinear(256, 256),
      ThresholdedLinear(256, 128),
      ThresholdedLinear(128, 128),
      ThresholdedLinear(128, 128),
      ThresholdedLinear(128, 64),
      ThresholdedLinear(64, out_width)
  ]
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


def test_profile_stuff_3():
  # prepare
  bs = 32
  device = torch.device('cuda:0')

  inp = torch.randn(bs, 64)
  targets = torch.randn(bs, 64)

  sequence = [
      ThresholdedLinear(64, 64),
      ThresholdedLinear(64, 64),
      FatMiddleLinear(64, 256),
      ThresholdedLinear(64, 64),
      ThresholdedLinear(64, 64)
  ]
  sequence.append(MyLoss(targets, device))

  checkpointed_model = c.CheckpointedSequential(sequence, device)

  upstream_gradients = (torch.tensor(1.).to(device),)

  # act
  checkpointed_model.profile_sequence(inp, upstream_gradients)
  compute_costs = checkpointed_model.compute_costs
  memory_costs = checkpointed_model.memory_costs

  # assert
  assert checkpointed_model.has_profiled == True
  assert compute_costs.shape == (2, 5+3)
  assert memory_costs.shape == (2, 5+3)
  print("Alpha")
  print(compute_costs)
  print("Beta")
  print(memory_costs)


def test_profile_stuff_4():
  # prepare
  bs = 256
  device = torch.device('cuda:0')

  inp = torch.randn(bs, 1024)
  targets = torch.randn(bs, 10)

  sequence = [
      ThresholdedLinear(1024, 1024),
      ThresholdedLinear(1024, 2048),
      FatMiddleLinear(2048, 8192),
      ThresholdedLinear(2048, 1024),
      ThresholdedLinear(1024, 1024),
      ThresholdedLinear(1024, 512),
      ThresholdedLinear(512, 100),
      ThresholdedLinear(100, 10),
      ThresholdedLinear(10, 10)
  ]
  sequence.append(MyLoss(targets, device))

  checkpointed_model = c.CheckpointedSequential(sequence, device)

  upstream_gradients = (torch.tensor(1.).to(device),)

  # act
  checkpointed_model.profile_sequence(inp, upstream_gradients)
  compute_costs = checkpointed_model.compute_costs
  memory_costs = checkpointed_model.memory_costs / 1.e6

  # assert
  assert checkpointed_model.has_profiled == True
  assert compute_costs.shape == (2, 9+3)
  assert memory_costs.shape == (2, 9+3)
  print("Alpha")
  print(compute_costs)
  print("Beta")
  print(memory_costs)



if __name__ == "__main__":
    test_profile_stuff_1()
    test_profile_stuff_2()
    test_profile_stuff_3()
    test_profile_stuff_4()

"""
in profiler and solver, everything is given on CPU.
But, in backprop_sequeunce, should be given on GPU (e.g. inputs),
and Loss needs to be a Lambda that moves targets to GPU.
"""
