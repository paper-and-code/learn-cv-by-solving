### Q2 : LKAのParameter数を計算してみましょう。

import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## (1) Depth Wise
input_channel = 3
tmp = 5
layer1 = nn.Conv2d(
    tmp,
    tmp,
    tmp  # ???
)
print(f"(1) : {count_parameters(layer1)}")

## (2) Depth Wise Dilated Convolution
layer2 = nn.Conv2d(
    tmp,
    tmp,
    tmp  # ???
)
print(f"(2) : {count_parameters(layer2)}")

## (3) Point Wise Convolution
layer3 = nn.Conv2d(
    tmp,
    tmp,
    tmp  # ???
)
print(f"(3) : {count_parameters(layer3)}")
