### Q3 : Self Attentionを実装してみましょう
# - ただし、
#   - AttentionのHead数は1, Headの次元は64,

import torch.nn as nn
import torch
from einops import rearrange


# Implemented by lucidrains
class SelfAttentionLucidrains(nn.Module):

    def __init__(self, dim, heads=1, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q, k = shape (B, Head Number, P, dim_head) = (1, 1, 256, 64)
        # qとkの行列積 (ANSWER)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # attentionとvの行列積 (ANSWER)
        # attn = shape (B, Head Number, P, P) = (1, 1, 256, 256)
        # v = shape (B, Head Number, P, dim_head) = (1, 1, 256, 64)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Implemented by Timm
class SelfAttentionTimm(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        # qとkの行列積 (ANSWER)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attentionとvの行列積 (ANSWER)
        # attn = shape (B, Head Number, P, P) = (1, 1, 256, 256)
        # v = shape (B, Head Number, P, dim_head) = (1, 1, 256, 64)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


model = SelfAttentionLucidrains(dim=64, heads=1, dim_head=64)
input = torch.rand(1, 256, 64)
output = model(input)
print(f'output : {output.shape}')
model = SelfAttentionTimm(dim=64, num_heads=1)
input = torch.rand(1, 256, 64)
output = model(input)
print(f'output : {output.shape}')