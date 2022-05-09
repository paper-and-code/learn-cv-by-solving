import torch
import torch.nn as nn

## Input
k = 50
d = 2048
embedding = nn.Embedding(k, d)
latents = torch.rand(3, 2048, 16, 16)  # [B x D x H x W]

## Output
latents = latents.permute(0, 2, 3,
                          1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
b, h, w, d = latents.shape
flat_latents = latents.view(-1, d)  # [BHW x D]

# Compute L2 distance between latents and embedding weights
dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding.weight ** 2, dim=1) - \
        2 * torch.matmul(flat_latents, embedding.weight.t())  # [BHW x K]

# Get the encoding that has the min distance
encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

# Convert to one-hot encodings
device = latents.device
encoding_one_hot = torch.zeros(encoding_inds.size(0), k, device=device)
encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

# Quantize the latents
quantized_latents = torch.matmul(encoding_one_hot,
                                 embedding.weight)  # [BHW, D]
quantized_latents = quantized_latents.view(b, h, w,
                                           d).permute(0, 3, 1,
                                                      2)  # [B x H x W x D]

print(f'quantized_latents : {quantized_latents.shape}')