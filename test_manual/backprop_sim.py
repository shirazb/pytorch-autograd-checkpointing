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


def test_sim():
    # prepare
    N = 5
    bs = 4
    layer_width = 8
    device = torch.device('cuda:0')

    inputs = torch.randn(bs, layer_width)
    targets = torch.randn(bs, layer_width)

    sequence = mk_seq(N, layer_width)
    m = c.CheckpointedSequential(sequence, device)
    upstream_gradients = (torch.ones(bs, layer_width).to(device),)

    M = 22
    m.profile_sequence(inputs, upstream_gradients)
    m.memory_costs = m.memory_costs // 1000
    print(m.memory_costs)
    B, C, policy = m.solve_optimal_policy(M)
    print("Policy (:,:,M)")
    print(np.triu(policy[:, :, -1], 2))

    # act
    time, peak = m.simulate_sequence(policy)

    # assert
    print("time", time)
    print("Predicted time", C[0, N+1, M-1] + np.sum(m.compute_costs[[0,0], [0,N+1]]))
    print(" mem", peak)
    print("Predicted peak", B[0, N+1, M-1] + m.memory_costs[0][0])


if __name__ == "__main__":
    test_sim()
