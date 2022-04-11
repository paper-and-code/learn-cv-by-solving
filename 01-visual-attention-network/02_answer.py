### Q2 : LKAのParameter数を計算してみましょう。

import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## (1) Depth Wise
input_channel = 3
layer1 = nn.Conv2d(input_channel,
                   input_channel,
                   5,
                   padding=2,
                   groups=input_channel)
print(
    f"(1) : {count_parameters(layer1)}"
)  # Input Channel x K x K x output_Channel + output_Channel (Bias) = 1 x 5 x 5 x 3 + 3 = 78

## (2) Depth Wise Dilated Convolution
layer2 = nn.Conv2d(in_channels=input_channel,
                   out_channels=input_channel,
                   kernel_size=7,
                   stride=1,
                   padding=9,
                   groups=input_channel,
                   dilation=3)
print(
    f"(2) : {count_parameters(layer2)}"
)  # Input Channel x K x K x output_cihannel + output_Channel (Bias) = 1 x 7 x 7 x 3 + 3 = 150

## (3) Point Wise Convolution
layer3 = nn.Conv2d(in_channels=input_channel,
                   out_channels=input_channel,
                   kernel_size=1)
print(
    f"(3) : {count_parameters(layer3)}"
)  # input_channel x K x K x output_channel + output_channel (Bias) = 3 x 1 x 1 x 3 + 3 = 1
