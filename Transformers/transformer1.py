#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:51:47 2022

@author: jack
https://hunlp.com/posts/52952.html

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



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        编码器、解码器、输入嵌入层、目标嵌入层、输出层
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src --> memory
        memory + tgt --> output
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        src --> memory
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        memory + tgt --> output
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)



def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        需要自主生成 mask 
        """
        for layer in self.layers:
            x = layer(x, mask)  #每一层的mask共用，但是后一层的x输入为前一层的x输出
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    inputs: batch, seq_len, features
    沿输入数据的特征维度归一化
    """
    def __init__(self, features, eps=1e-6):
        # 需要指定特征数量 features
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):  # 此模块不改变X的shape
        """
        x --> (x - x.mean) / x.std 
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



def test_layernorm():
    x = np.array([[[1., 2., 3.], [2., 4., 5.]],],dtype=np.float)
    print("Before Norm: \n", x)
    x = torch.from_numpy(x)  # batch, seq_len, features
    norm = LayerNorm(x.shape[-1])
    x = norm(x)
    print("After Norm: \n", x.detach().numpy())


test_layernorm()



class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        指定内部的结构 sublayer，是 attention 层，还是 feed_forward 层
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """size: d_model"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # 作为参数用于 layernorm 层
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)





# 解码器一次输入序列中向量，当前步后面的序列需要被遮盖
# 需要被遮盖的单词被标记为 False 

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])



def attention(query, key, value, mask=None, dropout=None):
    """
    query : batch, target_len, feats
    key   : batch, seq_len,    feats
    value : batch, seq_len,    val_feats
    
    return: batch, target_len, val_feats
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



# 3D
def test_attention_3D():
    query = torch.randn(3, 5, 4)  # batch, target_len, feats
    key = torch.randn(3, 6, 4)  # batch, seq_len, feats
    value = torch.randn(3, 6, 8)  # batch, seq_len, val_feats
    attn, _ = attention(query, key, value)
    print(attn.shape)
    assert attn.shape == (3, 5, 8)
    print("Test passed")

# 2D
def test_attention_2D():
    query = torch.randn(3, 5)  # batch, target_len, feats
    key = torch.randn(4,5)  # batch, seq_len, feats
    value = torch.randn(4, 8)  # batch, seq_len, val_feats
    attn, _ = attention(query, key, value)
    print(attn.shape)
    assert attn.shape == (3, 8)
    print("Test passed")

# 4D
def test_attention_4D():
     query = torch.randn(3, 5, 4,6)  # batch, target_len, feats
     key = torch.randn(3, 5, 7,6)  # batch, seq_len, feats
     value = torch.randn(3, 5, 7,9)  # batch, seq_len, val_feats
     attn, _ = attention(query, key, value)
     print(attn.shape)
     assert attn.shape == (3, 5, 4,9)
     print("Test passed")

# 5D
def test_attention_5D():
     query = torch.randn(3, 5,9, 4,6)  # batch, target_len, feats
     key = torch.randn(3, 5, 9,7,6)  # batch, seq_len, feats
     value = torch.randn(3, 5,9, 7,9)  # batch, seq_len, val_feats
     attn, _ = attention(query, key, value)
     print(attn.shape)
     assert attn.shape == (3, 5,9, 4,9)
     print("Test passed")


test_attention_3D()
test_attention_2D()
test_attention_4D()
test_attention_5D()



# 此模块不改变x的shape,即Q,K,V的shape和return的shape一致。
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        h, num_heads
        d_model, features
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 6
        self.h = h              # 2
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value: batch,seq_len,d_model
        print(f"query.shape = {query.shape},\nkey.shape={key.shape},\nvalue.shape = {value.shape}")
        #query.shape = torch.Size([2, 4, 12]),
        #key.shape=torch.Size([2, 4, 12]),
        #value.shape = torch.Size([2, 4, 12])
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 2
        
        for l, x in zip(self.linears, (query, key, value)):
             print(f"l =  {l},\nx = {x}\nx.shape = {x.shape}")
        
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  for l, x in zip(self.linears, (query, key, value)) ]
        #query, key, value = [l(x)  for l, x in zip(self.linears, (query, key, value)) ]
        print(f"query.shape = {query.shape},\nkey.shape={key.shape},\nvalue.shape = {value.shape}")
        """
        x.shape = torch.Size([2, 4, 12])
        query.shape = torch.Size([2, 2, 4, 6]),
        key.shape=torch.Size([2, 2, 4, 6]),
        value.shape = torch.Size([2, 2, 4, 6])
        """
        x, self.attn = attention(
            query,  # batch,num_head,seq_len,feats
            key,
            value,
            mask=mask,
            dropout=self.dropout)
        print(f"x.shape = {x.shape}") # x.shape = torch.Size([2, 2, 4, 6])
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        print(f"x.shape = {x.shape}") # x.shape = torch.Size([2, 4, 12])
        
        # batch,seq_len,num_head*feats
        print(f"self.linears[-1](x).shape = {self.linears[-1](x).shape}") #self.linears[-1](x).shape = torch.Size([2, 4, 12])
        return self.linears[-1](x)




def test_multi_head():
    x = torch.randn(2, 4, 12)
    y = torch.randn(2, 4, 12)
    z = torch.randn(2, 4, 12)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(2, d_model)
    attn = model(x, y, z)
    assert attn.shape == (2, 4, 12)
    print("Test passed!")

test_multi_head()






# 此模块不改变x的shape，即输入和输出的shape一样
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# x.shape = (batch_size,seq_len),则return X.shape = (batch_size,seq_len,d_model=512)
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# x.shape = (batch_size,seq_len,d_model=512),则return X.shape = (batch_size,seq_len,d_model=512)
# 此模块不改变x的shape，即输入和输出的shape一样
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #d_model=512,dropout=0.1,
        #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，一般100或者200足够了。
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        #每个位置用一个512维度向量来表示其位置编码，512也是词向量长度

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model) )
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
        
        #  (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 接受1.Embeddings的词嵌入结果x，
        #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        #例如，假设x是(30,10,512)的一个tensor，
        #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        #保证一个batch中的30个序列，都使用（叠加）一样的位置        
        
        # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数,requires_grad=False
        return self.dropout(x)



plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(33, 100, 20)))
print(f"y.shape = {y.shape}")  #torch.Size([33, 100, 20])
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])



def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)  # h为多头头数，d-model为每个样本的特征数，NLP中是表示词向量的维度 (Transformer-base： d=512d=512，Transformer-big: d = 1024d=1024)。
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)   #d_ff为前向网络的神经节点数
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # N为Encoder和Decoder层数
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

tmp_model = make_model(10, 10, 2)
# 查看模型结构：
print(f"tmp_model = \n{tmp_model}")
print(f"tmp_model.src_embed = \n{tmp_model.src_embed}")

from torchkeras import summary
#print(summary(tmp_model, input_shape=(3,4,3,3,3 )))

from torchsummary import summary
#print(f"summary(tmp_model, (3, 224, 224)) = {summary(tmp_model, (3, 224, 224))}")



# 查看网络参数：
for name, parameters in tmp_model.named_parameters():
    print(name, ':', parameters.size())

# #打印模型每层命名
#for k, v in tmp_model.items():
#     print(k,':',v)
  
  

#打印模型参数
for name, param in tmp_model.named_parameters():
     if param.requires_grad:
          print("-----model.named_parameters()--{}:{}".format(name, ""))



#设置参数是否更新：
opt_para = ['module.classifier.weight', 'module.classifier.bias']
for name, param in tmp_model.named_parameters():
     if name not in opt_para:
          param.requires_grad = False
     if name in opt_para:
          param.requires_grad = True




class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        src: 输入序列
        trg: 目标序列
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) #torch.Size([2, 1, 5])
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        将 pad 产生的 mask，和序列一次预测下一个单词产生的 mask 结合起来
        """
        tgt_mask = (tgt != pad).unsqueeze(-2) #torch.Size([2, 1, 5])
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) #torch.Size([1, 5, 5])
        return tgt_mask




src = torch.tensor([[3, 5, 7, 0, 0], [2, 4, 6, 8, 0]])  # batch=2,seq_len=5
trg = torch.tensor([[2, 3, 4, 5, 0, 0], [3, 5, 6, 0, 0,0]])  # batch=2,seq_len=6

sample = Batch(src, trg)
print(f"sample.src = \n{sample.src}\nsample.trg = \n{sample.trg}")

print(f"sample.src_mask = \n{sample.src_mask}")
print(f"sample.src_mask.shape = \n{sample.src_mask.shape}")

print(f"sample.trg_mask = \n{sample.trg_mask}")
print(f"sample.trg_mask.shape = \n{sample.trg_mask.shape}")




sample.trg_mask, sample.ntokens


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask,
                            batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens  # 总 tokens 数
        tokens += batch.ntokens  # 50 批训练时的总 tokens 数
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens















































































































































































































































































































































































































