import torch
import lltm # <--- import our compiled package
import time

if __name__ == "__main__":
    iter = 100

    input_features = 1024
    state_size = 1024
    batch = 256

    input = torch.rand((batch, input_features)).float()
    weights = torch.rand((3 * state_size, input_features + state_size)).float()
    bias = torch.rand((batch, 3 * state_size)).float()
    old_h = torch.rand((batch, state_size)).float()
    old_cell = torch.rand((batch, state_size)).float()

    tic = time.time()
    for i in range(iter):
        lltm.forward(input, weights, bias, old_h, old_cell)
    toc = time.time()

    print("Took {} seconds".format(toc-tic))
