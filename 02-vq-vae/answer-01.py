import torch

# Input
mu = torch.rand(1, 2048)
logvar = torch.rand(1, 2048)

# Output
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
eps = torch.randn(std.size(),
                  dtype=std.dtype,
                  layout=std.layout,
                  device=std.device)
z = eps * std + mu
