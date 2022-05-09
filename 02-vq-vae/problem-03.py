import torch
import torch.nn as nn

k = 50
d = 2048
embedding = nn.Embedding(k, d)
latents = torch.rand(3, 2048, 16, 16)  # [B x D x H x W]

latents = latents.permute(0, 2, 3,
                          1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
b, h, w, d = latents.shape
flat_latents = latents.view(-1, d)  # [BHW x D]

# Compute L2 distance between latents and embedding weights
dist = 1  # TODO # [BHW x K]

# Get the encoding that has the min distance
encoding_inds = 1  # TODO # [BHW, 1]

# Convert to one-hot encodings
device = latents.device
encoding_one_hot = torch.zeros(encoding_inds.size(0), k, device=device)
encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

# Quantize the latents
quantized_latents = 1  # TODO # [BHW, D]
quantized_latents = quantized_latents.view(b, h, w,
                                           d).permute(0, 3, 1,
                                                      2)  # [B x H x W x D]

print(f'quantized_latents : {quantized_latents.shape}')