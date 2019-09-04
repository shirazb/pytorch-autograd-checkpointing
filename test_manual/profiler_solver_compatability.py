import torch
import numpy as np
import pytorch_autograd_checkpointing as c

########### HELPERS FROM TESTING UTILS #########################################

_DEFAULT_LAYER_DIM = 10
def mk_seq(N, dim=_DEFAULT_LAYER_DIM, device='cpu'):
    return torch.nn.Sequential(
        *(layer(dim) for _ in range(N))
    ).to(device)

def layer(dim=_DEFAULT_LAYER_DIM):
    return ThresholdedLinear(dim, dim)

class ThresholdedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ThresholdedLinear, self).__init__()

        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))


def test_solver():
    # prepare
    N = 5
    bs = 4
    layer_width = 8
    device = torch.device('cuda:0')

    inp = torch.randn(bs, layer_width)
    targets = torch.randn(bs, layer_width)

    sequence = mk_seq(N, layer_width)
    m = c.CheckpointedSequential(sequence, device)
    upstream_gradients = (torch.ones(bs, layer_width).to(device),)

    # act
    M = 22
    m.profile_sequence(inp, upstream_gradients)

    m.memory_costs = m.memory_costs // 1000
    print(m.memory_costs)

    _, policy = m.solve_optimal_policy(M)

    # assert
    print("Policy (:,:,M)")
    print(np.triu(policy[:, :, -1], 2))

if __name__ == "__main__":
    test_solver()
