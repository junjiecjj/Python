#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:50:14 2022

@author: jack

https://github.com/junjiecjj/vit-pytorch/edit/main/vit_pytorch/vit.py


import torch
import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(20, 3, 256, 256)

preds = v(img)
参数解释：

image_size：int 类型参数，图片大小。 如果您有矩形图像，请确保图像尺寸为宽度和高度的最大值
patch_size：int 类型参数，patches数目。image_size 必须能够被 patch_size整除。
num_classes：int 类型参数，分类数目。
dim：int 类型参数，线性变换nn.Linear(..., dim)后输出张量的尺寸 。
depth：int 类型参数，Transformer模块的个数。
heads：int 类型参数，多头注意力中“头”的个数。
mlp_dim：int 类型参数，多层感知机中隐藏层的神经元个数。
channels：int 类型参数，输入图像的通道数，默认为3。
dropout：float类型参数，Dropout几率，取值范围为[0, 1]，默认为 0.。
emb_dropout：float类型参数，进行Embedding操作时Dropout几率，取值范围为[0, 1]，默认为0。
pool：string类型参数，取值为 cls或者 mean 。

"""


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # 正则化
        self.fn = fn                   # 具体的操作
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 计算最终进行全连接操作时输入神经元的个数
        project_out = not (heads == 1 and dim_head == dim) # 多头注意力并且输入和输出维度相同时为True


        self.heads = heads # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim = -1) # 初始化一个Softmax操作
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 对Q、K、V三组向量先进性线性操作
        
        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer包含多个编码器的叠加
        for _ in range(depth):
             # 编码器包含两大块：自注意力模块和前向传播模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
             # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
         # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        x = self.to_patch_embedding(img) 
        b, n, _ = x.shape  # shape (b, n, 1024)
        
        # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b) 
        x = torch.cat((cls_tokens, x), dim=1) # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)] # 进行位置编码，shape (b, n+1, 1024)
        x = self.dropout(x)

        x = self.transformer(x) # transformer操作

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x) # 线性输出
