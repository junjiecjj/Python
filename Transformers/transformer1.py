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


import sys

def function():
     print(sys._getframe().f_code.co_filename)  # 当前位置所在的文件名
     print(sys._getframe().f_code.co_name)  # 当前位置所在的函数名
     print(sys._getframe().f_lineno)  # 当前位置所在的行号
     print(f"File={sys._getframe().f_code.co_filename}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}")
     
function()



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
        return self.decode(tgt, memory, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        """
        src --> memory
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask,  tgt_mask):
        """
        memory + tgt --> output
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    # x.shape = (batch, seq_len, d_model) ---->  (batch, seq_len, vocab)
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


# test_layernorm()



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
     query = torch.randn(3, 5, 9, 4,6)  # batch, target_len, feats
     key = torch.randn(3, 5, 9, 7, 6)  # batch, seq_len, feats
     value = torch.randn(3, 5, 9, 7, 9)  # batch, seq_len, val_feats
     attn, _ = attention(query, key, value)
     print(attn.shape)
     assert attn.shape == (3, 5, 9, 4, 9)
     print("Test passed")


# test_attention_3D()
# test_attention_2D()
# test_attention_4D()
# test_attention_5D()



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
        self.h = h               # 2
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value: batch,seq_len,d_model
        #print(f"line = {sys._getframe().f_lineno}, query.shape = {query.shape},\nkey.shape={key.shape},\nvalue.shape = {value.shape}")
        #query.shape = torch.Size([2, 4, 12]),
        #key.shape=torch.Size([2, 4, 12]),
        #value.shape = torch.Size([2, 4, 12])
        
        if mask is not None:
            #print(f"..........mask.shape = {mask.shape}")
            mask = mask.unsqueeze(1)
            #print(f"........mask.shape = {mask.shape}")
        nbatches = query.size(0)  # 2

        #for l, x in zip(self.linears, (query, key, value)):
        #     print(f"line = {sys._getframe().f_lineno}, l =  {l},\nx.shape = {x.shape}")

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  for l, x in zip(self.linears, (query, key, value)) ]
        #query, key, value = [l(x)  for l, x in zip(self.linears, (query, key, value)) ]
        print(f"L = {sys._getframe().f_lineno}, query.shape = {query.shape},\nkey.shape={key.shape},\nvalue.shape = {value.shape}")
        """
        x.shape = torch.Size([2, 4, 12])
        query.shape = torch.Size([2, 2, 4, 6]),
        key.shape=torch.Size([2, 2, 4, 6]),
        value.shape = torch.Size([2, 2, 4, 6])
        """
        x, self.attn = self.attention(
            query,  # batch,num_head,seq_len,feats
            key,
            value,
            mask=mask,
            dropout=self.dropout)
        print(f"1  x.shape = {x.shape}") # x.shape = torch.Size([2, 2, 4, 6])
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        print(f"2  x.shape = {x.shape}") # x.shape = torch.Size([2, 4, 12])

        # batch,seq_len,num_head*feats
        #print(f"self.linears[-1](x).shape = {self.linears[-1](x).shape}") #self.linears[-1](x).shape = torch.Size([2, 4, 12])
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
         """
         query : batch, target_len, feats
         key   : batch, seq_len,    feats
         value : batch, seq_len,    val_feats

         return: batch, target_len, val_feats
         """
         d_k = query.size(-1)
         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
         print(f" 1  scores.shape = {scores.shape}")
         
         if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
         print(f" 2  scores.shape = {scores.shape}")
         p_attn = F.softmax(scores, dim=-1)

         if dropout is not None:
             p_attn = dropout(p_attn)
         print(f"torch.matmul(p_attn, value).shape = {torch.matmul(p_attn, value).shape}")
         return torch.matmul(p_attn, value), p_attn




src_vocab=10
tgt_vocab=10
N=6
d_model=512
d_ff=2048
h=8
dropout=0.1






# 此模块不改变x的shape，即输入和输出的shape一样
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def Test_PositionwiseFeedForward():
    x = torch.randn(2, 4, 12)
    d_model = x.shape[-1]
    model = PositionwiseFeedForward(d_model, 33)
    attn = model(x )
    assert attn.shape == (2, 4, 12)
    print("Test passed!")

def Test_PositionwiseFeedForward1():
    x = torch.randn(4, 12)
    d_model = x.shape[-1]
    model = PositionwiseFeedForward(d_model, 33)
    attn = model(x )
    assert attn.shape == (4, 12)
    print("Test passed!")

def Test_PositionwiseFeedForward2():
    x = torch.randn(4, 5, 6, 12)
    d_model = x.shape[-1]
    model = PositionwiseFeedForward(d_model, 33)
    attn = model(x )
    assert attn.shape == (4, 5, 6, 12)
    print("Test passed!")


#Test_PositionwiseFeedForward()
#Test_PositionwiseFeedForward1()
#Test_PositionwiseFeedForward2()
# 以上测试说明 PositionwiseFeedForward() 不改变x的shape



# x.shape = (batch_size,seq_len),则return X.shape = (batch_size,seq_len,d_model=512)
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #print(f"line={sys._getframe().f_lineno},x.shape = {x.shape}")
        #print(f"line={sys._getframe().f_lineno},self.lut(x).shape = {self.lut(x).shape}")
        #print(f"x = {x}, self.d_model = {self.d_model}")
        return self.lut(x) * math.sqrt(self.d_model)



def test_Embeddings():
     x = torch.randint(0,20,(5,9))
     vocab = 132
     d_model = 128
     em = Embeddings(vocab,d_model )
     out = em(x)
     print(out.shape)
     assert out.shape == (5, 9, d_model)
     print("Test passed!")
#test_Embeddings()



def test_Embeddings2():
     x = torch.randint(0,20,(5,))
     vocab = 132
     d_model = 128
     em = Embeddings( vocab, d_model)
     out = em(x)
     print(out.shape)
     assert out.shape == (5,  d_model)
     print("Test passed!")
#test_Embeddings2()


def test_Embeddings3():
     x = torch.randint(0,20,(5,9,12))
     vocab = 132
     d_model = 128
     em = Embeddings( vocab, d_model)
     out = em(x)
     print(out.shape)
     assert out.shape == (5, 9,12, d_model)
     print("Test passed!")
#test_Embeddings3()


# 以上测试说明Embedding只是在x的最后一维增加一个词向量维度


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
        #print(f"line={sys._getframe().f_lineno},x.shape = {x.shape}")
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        #print(f"line={sys._getframe().f_lineno},x.shape = {x.shape}")
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


def Test_PositionalEncoding():
    x = torch.randn(4, 5, 12)
    d_model = x.shape[-1]
    pe = PositionalEncoding(d_model, 0)
    attn = pe(x )
    assert attn.shape == (4, 5,  12)
    print("Test passed!")
Test_PositionalEncoding()


def Test_PositionalEncoding1():
    x = torch.randn(4, 5, 6, 12)
    d_model = x.shape[-1]
    pe = PositionalEncoding(d_model, 0)
    attn = pe(x )
    assert attn.shape == (4, 5, 6,  12)
    print("Test passed!")
#Test_PositionalEncoding1()  #error


def Test_PositionalEncoding2():
    x = torch.randn(4,   12)
    d_model = x.shape[-1]
    pe = PositionalEncoding(d_model, 0)
    attn = pe(x )
    assert attn.shape == (4,    12)
    print("Test passed!")
#Test_PositionalEncoding2() #error


# 以上测试说明PositionalEncoding()的输入x只能是三维的，且不改变x的shape


def test_multi_head():
    x = torch.randint(1, 200, (2, 4, 12), dtype=torch.float)
    y = torch.randint(1, 200, (2, 4, 12), dtype=torch.float)
    z = torch.randint(1, 200, (2, 4, 12), dtype=torch.float)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(2, d_model)
    attn = model(x, y, z)
    assert attn.shape == (2, 4, 12)
    print("Test passed!")
#test_multi_head()


def test_multi_head1():
    x = torch.randn(2, 4, 12)
    y = torch.randn(2, 4, 12)
    z = torch.randn(2, 4, 12)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(2, d_model)
    attn = model(x, y, z)
    assert attn.shape == (2, 4, 12)
    print("Test passed!")
#test_multi_head1()


def test_multi_head2(pad = 0):
    x = torch.randint(1, 200, (128, 31))
    src_mask = (x != pad).unsqueeze(-2)
    print(f"src_mask.shape = {src_mask.shape}") # src_mask.shape = torch.Size([128, 1, 31])

    d_model = 128
    emb = Embeddings(2222, d_model)
    x_e = emb(x)
    print(f"x_e.shape = {x_e.shape}") # x_e.shape = torch.Size([128, 31, 128])

    PE = PositionalEncoding(d_model=d_model, dropout=0.1)

    x_pe = PE(x_e)
    print(f"x_pe.shape = {x_pe.shape}") # x_pe.shape = torch.Size([128, 31, 128])

    d_model = x_pe.shape[-1]

    model = MultiHeadedAttention(8, d_model)
    attn = model(x_pe, x_pe, x_pe, mask = src_mask)
    assert attn.shape == (128, 31, 128)
    print("Test passed!")
# test_multi_head2()

def test_multi_head3():
    x = torch.randn(128, 30, 128)
    m = torch.randn(128, 31, 128)
    mask = torch.randn(128, 1, 31)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(8, d_model)
    attn = model(x, m, m,mask=mask)
    assert attn.shape == (128, 30, 128)
    print("Test passed!")
#test_multi_head3()

def test_multi_head4():
    x = torch.randn(128, 30, 128)
    m = torch.randn(128, 31, 128)
    mask = torch.randn(128, 1, 31)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(8, d_model)
    attn = model(x, m, m,mask=mask)
    assert attn.shape == (128, 30, 128)
    print("Test passed!")
#test_multi_head4()


def test_multi_head_src(pad = 0):
     d_model = 128
     batchsize = 128
     seq_len = 31
     head = 8

     x = torch.randint(1, 20, (batchsize, seq_len, d_model), dtype=torch.float)
     print(f"x.shape = {x.shape}")# x.shape = torch.Size([128, 31, 128])

     src_mask = (torch.randint( 0, 2, (batchsize, seq_len)) != pad).unsqueeze(-2)
     print(f"src_mask.shape = {src_mask.shape}") # src_mask.shape = torch.Size([128, 1, 31])

     if src_mask != None:
          src_mask = src_mask.unsqueeze(1)
     print(f"src_mask.shape = {src_mask.shape}") # src_mask.shape = torch.Size([128, 1, 1, 31])


     d_k = d_model//head
     x = x.view(batchsize, -1, head, d_k)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 31, 8, 16])

     x = x.transpose(1,2)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 8, 31, 16])

     scores = torch.matmul(x, x.transpose(-2,-1))
     print(f"1 scores.shape = {scores.shape}") #  1 scores.shape = torch.Size([128, 8, 31, 31])
     if src_mask is not None:
          scores = scores.masked_fill(src_mask==0, -1e9)
          print(f"2 scores.shape = {scores.shape}") # 2 scores.shape = torch.Size([128, 8, 31, 31])

     p_attn = F.softmax(scores, dim = -1)
     print(f"1 p_attn.shape = {p_attn.shape}") #  1 p_attn.shape = torch.Size([128, 8, 31, 31])

     dropout = nn.Dropout(0.1)
     P_attn = dropout(p_attn)
     print(f"2 P_attn.shape = {P_attn.shape}") # 2 P_attn.shape = torch.Size([128, 8, 31, 31])
     attn = torch.matmul(P_attn, x)

     print(f"1  attn.shape = {attn.shape}") # 1  attn.shape = torch.Size([128, 8, 31, 16])

     attn = attn.transpose(1,2).contiguous().view(batchsize, -1, head*d_k)
     print(f"2  attn.shape = {attn.shape}") # 2  attn.shape = torch.Size([128, 31, 128])

     print("Test passed!")
#test_multi_head_src()


def test_multi_head_tgt(pad = 0):
     batchsize = 128
     d_model = 128
     seq_len = 30
     head = 8

     x = torch.randint(1, 20, (batchsize, seq_len, d_model), dtype=torch.float)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 30, 128])


     attn_shape = (1, seq_len, seq_len)
     sub_mask = torch.from_numpy(np.triu(np.ones(attn_shape), k=1) .astype('uint8'))==0
     # sub_mask.type
     
     tgt_mask = (torch.randint(0, 2, (batchsize, seq_len)) != pad).unsqueeze(-2)
     print(f"tgt_mask.shape = {tgt_mask.shape}") # tgt_mask.shape = torch.Size([128, 1, 30])

     tgt_mask = tgt_mask & sub_mask    
     print(f"tgt_mask.shape = {tgt_mask.shape}") # tgt_mask.shape = torch.Size([128, 30, 30])

     if tgt_mask != None:
          tgt_mask = tgt_mask.unsqueeze(1)
     print(f"tgt_mask.shape = {tgt_mask.shape}") # tgt_mask.shape = torch.Size([128, 1, 30, 30])


     d_k = d_model//head
     x = x.view(batchsize, -1, head, d_k)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 30, 8, 16])

     x = x.transpose(1,2)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 8, 30, 16])

     scores = torch.matmul(x, x.transpose(-2,-1))
     print(f"1 scores.shape = {scores.shape}") # 1 scores.shape = torch.Size([128, 8, 30, 30])
     if tgt_mask is not None:
          scores = scores.masked_fill(tgt_mask==0, -1e9)
          print(f"2 scores.shape = {scores.shape}") # 2 scores.shape = torch.Size([128, 8, 30, 30])


     attn = torch.matmul(scores, x)

     print(f"1  attn.shape = {attn.shape}") # 1  attn.shape = torch.Size([128, 8, 30, 16])

     attn = attn.transpose(1,2).contiguous().view(batchsize, -1, head*d_k)
     print(f"2  attn.shape = {attn.shape}") #  2  attn.shape = torch.Size([128, 30, 128])

     print("Test passed!")
#test_multi_head_tgt()


def test_multi_headSrcTgt(pad = 0):
     d_model = 128
     batchsize = 128
     seq_len_tgt = 30
     seq_len_src = 31
     head = 8

     x = torch.randint(1, 20, (batchsize, seq_len_tgt, d_model), dtype=torch.float)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 30, 128])

     src_mask = (torch.randint(0, 2, (batchsize, seq_len_src)) != pad).unsqueeze(-2)
     print(f"src_mask.shape = {src_mask.shape}") # src_mask.shape = torch.Size([128, 1, 31])

     if src_mask != None:
          src_mask = src_mask.unsqueeze(1)
     print(f"src_mask.shape = {src_mask.shape}") # src_mask.shape = torch.Size([128, 1, 1, 31])

     m =   torch.randint(1, 20, (batchsize, seq_len_src, d_model), dtype=torch.float)
     print(f"m.shape = {m.shape}") # m.shape = torch.Size([128, 31, 128])

     d_k = d_model//head
     x = x.view(batchsize, -1, head, d_k).transpose(1,2)
     print(f"x.shape = {x.shape}") # x.shape = torch.Size([128, 8, 30, 16])

     m = m.view(batchsize, -1, head, d_k).transpose(1,2)
     print(f"m.shape = {m.shape}") # m.shape = torch.Size([128, 8, 31, 16])

     scores = torch.matmul(x, m.transpose(-2,-1))
     print(f"1 scores.shape = {scores.shape}") # 1 scores.shape = torch.Size([128, 8, 30, 31])
     
     if src_mask is not None:
          scores = scores.masked_fill(src_mask==0, -1e9)
          print(f"2 scores.shape = {scores.shape}") # 2 scores.shape = torch.Size([128, 8, 30, 31])


     attn = torch.matmul(scores, m)
     print(f"1  attn.shape = {attn.shape}") # 1  attn.shape = torch.Size([128, 8, 30, 16])

     attn = attn.transpose(1,2).contiguous().view(batchsize, -1, head*d_k)
     print(f"2  attn.shape = {attn.shape}") # 2  attn.shape = torch.Size([128, 30, 128])

     print("Test passed!")
#test_multi_headSrcTgt()




# 以上测试说明 MultiHeadedAttention() 不改变x的shape





def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)  # h为多头头数，d-model为每个样本的特征数，NLP中是表示词向量的维度 (Transformer-base： d=512d=512，Transformer-big: d = 1024d=1024)。
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)   #d_ff为前向网络的神经节点数
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), # N为Encoder和Decoder层数
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(src_vocab, d_model), c(position)),
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# tmp_model = make_model(10, 10, 2)
# 查看模型结构：
#print(f"tmp_model = \n{tmp_model}")
#print(f"tmp_model.src_embed = \n{tmp_model.src_embed}")


"""
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
"""


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

"""

src = torch.tensor([[3, 5, 7, 0, 0], [2, 4, 6, 8, 0]])  # batch=2,seq_len=5
trg = torch.tensor([[2, 3, 4, 5, 0, 0], [3, 5, 6, 0, 0,0]])  # batch=2,seq_len=6

sample = Batch(src, trg)
print(f"sample.src = \n{sample.src}\nsample.trg = \n{sample.trg}")

print(f"sample.src_mask = \n{sample.src_mask}")
print(f"sample.src_mask.shape = \n{sample.src_mask.shape}")

print(f"sample.trg_mask = \n{sample.trg_mask}")
print(f"sample.trg_mask.shape = \n{sample.trg_mask.shape}")

print(f"sample.trg_mask = {sample.trg_mask}, sample.ntokens = {sample.ntokens}")

"""


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # print(f"out.shape = {out.shape}, batch.trg_y.shape = {batch.trg_y.shape}")
        # out.shape = torch.Size([30, 9, 512]), batch.trg_y.shape = torch.Size([30, 9])
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        # print(f"loss = {loss}")  # loss = 818.6756591796875
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



global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)





class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5) *
                              min(step**(-0.5), step * self.warmup**(-1.5)))


def get_std_opt(model):
    return NoamOpt(
        model.src_embed[0].d_model, 2, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                         eps=1e-9))




"""

opts = [
    NoamOpt(512, 1, 4000, None),
    NoamOpt(512, 1, 8000, None),
    NoamOpt(256, 1, 4000, None),
]
plt.plot(np.arange(1, 20000),
         [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])

"""



class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

"""
crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.4)
predict = torch.FloatTensor([
    [0, 0.2, 0.7, 0.1, 0],
    [0, 0.2, 0.7, 0.1, 0],
    [0, 0.2, 0.7, 0.1, 0],
])
v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
plt.imshow(crit.true_dist)
print(f"crit.true_dist = \n{crit.true_dist}")




crit = LabelSmoothing(5, 0, 0.1)


def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([
        [0, x / d, 1 / d, 1 / d, 1 / d],  # 概率分布，x 的值越大，标签 1 的概率越大
    ])
    #print(predict)
    return crit(
        Variable(predict.log()),
        Variable(torch.LongTensor([1])),  # 真实标签为 1
    ).item()


plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])


"""



# 生成随机数据
def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)




class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator  # 模型最后的输出层
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        # print(f"1  x.shape = {x.shape}, y.shape = {y.shape}")
        # x.shape = torch.Size([30, 9, 512]), y.shape = torch.Size([30, 9])
        x = self.generator(x)
        # print(f"2  x.shape = {x.shape}, y.shape = {y.shape}")
        # x.shape = torch.Size([30, 9, 11]), y.shape = torch.Size([30, 9])
        #print(x.contiguous().view(-1, x.size(-1)).shape)  # torch.Size([270, 11])
        #print(y.contiguous().view(-1).shape)  # torch.Size([270])
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))



for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask) # torch.Size([1, 10, 512])
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # tensor([[1]])
    for i in range(max_len - 1):
        out = model.decode(Variable(ys), memory, src_mask, Variable(subsequent_mask(ys.size(1)).type_as(src.data)))  #torch.Size([1, 3, 512])
        prob = model.generator(out[:, -1])  #torch.Size([1, 11])
        _, next_word = torch.max(prob, dim=1)   #
        next_word = next_word.item()      # 4
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


model.eval()
src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))



# 实战
from torchtext import data, datasets

if True:
    import spacy
    #spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de,
                     pad_token=BLANK_WORD)  # 定义预处理流程，分词、填充、
    TGT = data.Field(tokenize=tokenize_en,
                     init_token=BOS_WORD,
                     eos_token=EOS_WORD,
                     pad_token=BLANK_WORD)

    MAX_LEN = 100

    # 数据集
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'),
        fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
            vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2

    # 创建词汇表
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)



# 数据分批对训练速度很重要：需要拆分成均匀的批次，最小的填充

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train: # 训练模式，数据分批，然后打乱顺序

            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch): # batch first --> True
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)




# 使用 multi-gpu 加速训练速度：将单词生成拆分成块，便于并行处理


class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):

        total = 0.0

        # 将最终的线性输出层 并行 到多个 gpu中
        generator = nn.parallel.replicate(self.generator, devices=devices)

        # 将 transformer 的输出张量 并行 多个 gpu 中
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]

        # 将目标 并行 到多个 gpu 中
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # 将生成拆分成块？？
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):

            # 预测分布
            out_column = [[
                Variable(o[:, i:i + chunk_size].data,
                         requires_grad=self.opt is not None)
            ] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算损失
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # 损失求和并归一化
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # 反向传播
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # 反向传播整个模型
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab),
                               padding_idx=pad_idx,
                               smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True)
    valid_iter = MyIterator(val,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=False)
    model_par = nn.DataParallel(model, device_ids=devices)


if False:
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 2000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                         eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator,
                                      criterion,
                                      devices=devices,
                                      opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator,
                                             criterion,
                                             devices=devices,
                                             opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")





for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model,
                        src,
                        src_mask,
                        max_len=60,
                        start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    break


if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight



































































































































































































































































































