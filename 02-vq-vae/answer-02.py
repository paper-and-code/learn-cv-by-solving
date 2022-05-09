import torch

## Input
mu = torch.rand(1, 2048)
logvar = torch.rand(1, 2048)
beta = 0.50

## Output
kld_loss = torch.mean(0.5 * beta *
                      torch.sum(logvar.exp() + logvar - 1 - mu**2, dim=1),
                      dim=0)
