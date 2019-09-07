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


def test_executor():
    # prepare
    N = 5
    bs = 4
    layer_width = 8
    device = torch.device('cuda:0')

    inputs = torch.randn(bs, layer_width)
    inputs_duplicate = torch.empty_like(inputs).copy_(inputs)
    targets = torch.randn(bs, layer_width)

    sequence = mk_seq(N, layer_width)
    m = c.CheckpointedSequential(sequence, device)
    upstream_gradients = (torch.ones(bs, layer_width).to(device),)
    upstream_gradients_duplicate = (torch.ones(bs, layer_width).to(device),)
    print("b_N+2.grad = ", upstream_gradients[0].grad)

    M = 22
    m.profile_sequence(inputs, upstream_gradients)
    m.memory_costs = m.memory_costs // 1000
    print(m.memory_costs)
    _, _, policy = m.solve_optimal_policy(M)
    print("Policy (:,:,M)")
    print(np.triu(policy[:, :, -1], 2))

    # act: recomp_grads
    inputs = inputs.to(device)
    inputs.requires_grad = True
    recomp_grads = m.backprop_sequence(policy, inputs, upstream_gradients)() # call is to get strong ref out of weakref

    # act: normal_grads
    inputs_duplicate = inputs_duplicate.to(device)
    inputs_duplicate.requires_grad = True
    out = sequence(inputs_duplicate)
    torch.autograd.backward(out, upstream_gradients_duplicate)
    normal_grads = inputs_duplicate.grad

    # assert
    print(recomp_grads)
    print(normal_grads)
    print(recomp_grads == normal_grads)

if __name__ == "__main__":
    test_executor()
