#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:09:20 2022

@author: jack
https://zhuanlan.zhihu.com/p/107586681
https://blog.csdn.net/qq_21749493/article/details/103037451


pytorch 文档中有五个相关class：

Transformer
TransformerEncoder
TransformerDecoder
TransformerEncoderLayer
TransformerDecoderLayer

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='talk')

"""
1、Transformer

init:

torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
activation='relu', custom_encoder=None, custom_decoder=None)
transformer模型，该结构基于论文 Attention Is All You Need

用户可以使用相应的参数构建BERT https://arxiv.org/abs/1810.04805

参数：
d_model – 编码器/解码器输入中预期词向量的大小(默认值= 512).
nhead – 多头注意力模型中的头数(默认为8).
num_encoder_layers – 编码器中子编码器层(transformer layers)的数量(默认为6).
num_decoder_layers – 解码器中子解码器层的数量（默认为6).
dim_feedforward – 前馈网络模型的尺寸（默认值= 2048).
dropout – dropout的比例 (默认值=0.1).
activation – 编码器/解码器中间层，激活函数relu或gelu(默认=relu).
custom_encoder – 自定义编码器(默认=None).
custom_decoder – 自定义解码器(默认=None).

例子：
"""

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)

src = torch.rand((10, 32, 512))



tgt = torch.rand((20, 32, 512))

out = transformer_model(src, tgt)





"""
2、TransformerEncoder
init

torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
TransformerEncoder是N个编码器层的堆叠

参数：

coder_layer – TransformerEncoderLayer（）类的实例（必需）。
num_layers –编码器中的子编码器层数（必填）。
norm –层归一化组件（可选）。


"""

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
# out.shape
# Out[346]: torch.Size([10, 32, 512])

# src = torch.rand(20,3, 128, 512)
# out = transformer_encoder(src)
# tgt_len, bsz, embed_dim = query.shape
# ValueError: too many values to unpack (expected 3)

"""
3、TransformerDecoder
init

torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
transformerDecoder是N个解码器层的堆叠

decoder_layer – TransformerDecoderLayer（）类的实例（必需）。
num_layers –解码器中子解码器层的数量（必需）。
norm –层归一化组件（可选）。
"""

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
# out.shape
# Out[348]: torch.Size([20, 32, 512])

# tgt = torch.rand(20,3, 32, 512)
# out = transformer_decoder(tgt, memory)
# ValueError: too many values to unpack (expected 3)


"""
4、TransformerEncoderLayer
init

torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
TransformerEncoderLayer 由self-attn和feedforward组成，此标准编码器层基于“Attention Is All You Need”一文。

d_model – the number of expected features in the input (required).
nhead – the number of heads in the multiheadattention models (required).
dim_feedforward – the dimension of the feedforward network model (default=2048).
dropout – the dropout value (default=0.1).
activation – the activation function of intermediate layer, relu or gelu (default=relu).
"""

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
src = torch.rand(10, 32, 512)
out = encoder_layer(src)
print(f"out.shape = {out.shape}")
# out.shape
# Out[350]: torch.Size([10, 32, 512])

src = torch.rand(10,3, 32, 512)
out = encoder_layer(src)
print(f"out.shape = {out.shape}")


"""
5、TransformerDecoderLayer
init

torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
TransformerEncoderLayer 由self-attn和feedforward组成，此标准编码器层基于“Attention Is All You Need”一文。

d_model – the number of expected features in the input (required).
nhead – the number of heads in the multiheadattention models (required).
dim_feedforward – the dimension of the feedforward network model (default=2048).
dropout – the dropout value (default=0.1).
activation – the activation function of intermediate layer, relu or gelu (default=relu).
"""

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = decoder_layer(tgt, memory)
# out.shape
# Out[348]: torch.Size([20, 32, 512])




"""
6、MultiheadAttention
init

torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, 
bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
embed_dim – total dimension of the model.
num_heads – parallel attention heads.
dropout – a Dropout layer on attn_output_weights. Default: 0.0.
bias – add bias as module parameter. Default: True.
add_bias_kv – add bias to the key and value sequences at dim=0.
add_zero_attn – add a new batch of zeros to the key and value sequences at dim=1.
kdim – total number of features in key. Default: None.
vdim – total number of features in key. Default: None.
Note – if kdim and vdim are None, they will be set to embed_dim such that
key, and value have the same number of features. (query,)


multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
"""

import torch 
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as f
 


# 此模块不改变x的shape,即Q,K,V的shape和return的shape一致。
class MultiHeadedAttention(nn.Module):
     # num_heads=8, d_model=128,
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads    # d_k=16
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # nbatches = 128

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn



A = torch.Tensor(5,2,4)
nn.init.xavier_normal_(A)
# 此模块不改变x的shape,即Q,K,V的shape和return的shape一致。
self_attn = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2, dropout=0.0)
res,weight = self_attn(A,A,A)

print(f"res.shape = {res.shape}")  # res.shape = torch.Size([5, 2, 4])

att1 = MultiHeadedAttention(2,4)
res1 = att1(A,A,A)
print(f"res1.shape = {res1.shape}")  # res1.shape = torch.Size([5, 2, 4])


# https://zhuanlan.zhihu.com/p/358206572

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        print ('Before transform query: ' + str(query.size())) # (batch_size, seq_length, d_model)  

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))] # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)] # (batch_size, h, seq_length, d_k)

        print ('After transform query: ' + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


h = 8
d_model = 512
batch_size = 1
seq_length = 10
model = MultiHeadAttention(h, d_model)

query = torch.randn([batch_size, seq_length, d_model])
key = query
value = query

print ('Input size: ' + str(query.size()))

m = model(query, key, value)

print ('Output size: ' + str(m.size()))

# Input size: torch.Size([1, 10, 512])
# Before transform query: torch.Size([1, 10, 512])
# After transform query: torch.Size([1, 8, 10, 64])
# Output size: torch.Size([1, 10, 512])





































































































































































