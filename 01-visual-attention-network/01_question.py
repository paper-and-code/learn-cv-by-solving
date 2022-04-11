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
        self.conv0 = nn.Conv2d(in_channels=input_channel #??? , ... 
        )
    def forward(self, x):
        # ??