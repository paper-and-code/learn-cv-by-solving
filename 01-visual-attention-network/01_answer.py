### Q1 : LKA(Large Kernel Attention)を実装してみましょう。
# - ただし、
#   - 1層目のDepth Wise ConvolutionのPaddingは2, Kernel Size 5
#   - 2層目のDepth Wise Dilated ConvolutionのKernel Size 7, Stride 3
#   - 1,2,3層目のOutputのChannel数は入力と同じChannel数とする。

import torch.nn as nn
import torch


class LKA(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=input_channel,
                               out_channels=input_channel,
                               kernel_size=5,
                               padding=2,
                               groups=input_channel)
        self.conv_spatial = nn.Conv2d(in_channels=input_channel,
                                      out_channels=input_channel,
                                      kernel_size=7,
                                      stride=1,
                                      padding=9,
                                      groups=input_channel,
                                      dilation=3)
        self.conv1 = nn.Conv2d(in_channels=input_channel,
                               out_channels=input_channel,
                               kernel_size=1)

    def forward(self, x):
        u = x.clone()
        print(f'x : {x.shape}')
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        print(f'attn : {attn.shape}')
        return u * attn


model = LKA(3)
x = torch.rand(1, 3, 256, 256)
output = model(x)
print(f'output : {output.shape}')