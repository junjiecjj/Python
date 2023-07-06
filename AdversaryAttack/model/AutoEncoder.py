#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:18:44 2023

@author: jack

https://zhuanlan.zhihu.com/p/116769890

https://blog.csdn.net/weixin_38739735/article/details/119013420

https://zhuanlan.zhihu.com/p/625085766

https://blog.csdn.net/winycg/article/details/90318371

https://www.bilibili.com/read/cv12946597

https://zhuanlan.zhihu.com/p/133207206

https://zhuanlan.zhihu.com/p/628604566


"""

import os, sys
import numpy as np
import torch
from torch import nn


sys.path.append("../")
# from . import common
from model  import common
#=============================================================================================================
#  AE based on mlp for MNIST
#=============================================================================================================

# https://zhuanlan.zhihu.com/p/116769890
class AED_mlp_MNIST(nn.Module):
    def __init__(self):
        super(AED_mlp_MNIST, self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()

        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return  encoded, decoded



#=============================================================================================================
#  AE based on cnn for MNIST
#=============================================================================================================


# https://blog.csdn.net/weixin_38739735/article/details/119013420
class Encoder_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_cnn_mnist, self).__init__()
        ### Convolutional p
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear p
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim),
            nn.Tanh()
        )
    def forward(self, x):
        # print(f"1 x.shape = {x.shape}")
        # torch.Size([25, 1, 28, 28])
        x = self.encoder_cnn(x)
        # print(f"2 x.shape = {x.shape}")
        # torch.Size([25, 32, 3, 3])
        x = self.flatten(x)
        # print(f"3 x.shape = {x.shape}")
        # torch.Size([25, 288])
        x = self.encoder_lin(x)
        # print(f"4 x.shape = {x.shape}")
        # torch.Size([25, 4])
        return x


class Decoder_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder_cnn_mnist, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,  padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x):
        # print(f"1 x.shape = {x.shape}")
        # 1 torch.Size([25, 4])
        x = self.decoder_lin(x)
        # print(f"2 x.shape = {x.shape}")
        # 2 x.shape = torch.Size([25, 288])
        x = self.unflatten(x)
        # print(f"3 x.shape = {x.shape}")
        # 3 x.shape = torch.Size([25, 32, 3, 3])
        x = self.decoder_conv(x)
        # print(f"4 x.shape = {x.shape}")
        # 4 x.shape = torch.Size([25, 1, 28, 28])

        x = torch.sigmoid(x)
        # x = torch.tanh(x)
        # print(f"5 x.shape = {x.shape}")
        # 5 x.shape = torch.Size([25, 1, 28, 28])
        return x

class AED_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim = 100, snr  = 3, quantize = True):
        super(AED_cnn_mnist, self).__init__()
        self.snr = snr
        self.quantize = quantize
        self.encoder = Encoder_cnn_mnist(encoded_space_dim)
        self.decoder = Decoder_cnn_mnist(encoded_space_dim)

    def forward(self, img, attack_vector = "" ):
        # print(f"img.shape = {img.shape}")
        encoded = self.encoder(img)
        # print(f"1 encoded.requires_grad = {encoded.requires_grad}")
        # print(f"0:    {encoded.min()}, {encoded.max()}, {encoded.mean()}")

        if self.quantize == True:
            encoded = common.Quantize(encoded)
        else:
            pass   # quatized = encoded

        ### semantic attack
        if attack_vector != "":
            encoded = encoded + attack_vector

        # print(f"0: {encoded.min()}, {encoded.max()}")

        Y =  common.Awgn(encoded, snr = self.snr)
        # print(f"2 encoded.requires_grad = {encoded.requires_grad}")

        decoded = self.decoder(Y)
        # print(f"3 decoded.requires_grad = {decoded.requires_grad}")
        return decoded

    def set_snr(self, snr):
        self.snr = snr

    def save(self, savedir, comp, snr, name = "AE_cnn_mnist"):
        save = os.path.join(savedir, f"{name}_comp={comp:.2f}_snr={snr:.0f}.pt")
        torch.save(self.model.state_dict(), save)
        return


# X = torch.randint(low = 0, high= 255, size = (128, 1, 28, 28)) * 1.0
# ae = AED_cnn_mnist(100, snr = None)
# y = ae(X)


# for param in ae.parameters():
#     print("param=%s, grad=%s" % (param.data.item(), param.grad.item()))

# #打印某一层的参数名
# for name in ae.state_dict():
#    print(name)
# #Then  I konw that the name of target layer is '1.weight'

# #schemem1(recommended)
# print(f"ae.state_dict()['encoder.encoder_cnn.0.weight'] = \n    {ae.state_dict()['encoder.encoder_cnn.0.weight']}")


# #打印每一层的参数名和参数值
# for name in ae.state_dict():
#     print(f" name = {name}  \n ae.state_dict()[{name}] = {ae.state_dict()[name]}")
#     # print(name)
#     # print(ae.state_dict()[name])



# #打印每一层的参数名和参数值
# params = list(ae.named_parameters())#get the index by debuging
# l = len(params)
# for i in range(l):
#     # print(params[i][0])              # name
#     # print(params[i][1].data)         # data
#     print(f" params[{i}][0] = {params[i][0]}, \n params[{i}][1].data = \n      {params[i][1].data}")



# #打印每一层的参数名和参数值
# params = {}#change the tpye of 'generator' into dict
# for name, param in ae.named_parameters():
#     params[name] = param.detach().cpu().numpy()
#     print(f"name = {name}, params[{name}] = \n{params[name]}")



# #打印每一层的参数名和参数值
# #schemem1(recommended)
# for name, param in ae.named_parameters():
#     # print(f"  name = {name}\n  param = \n    {param}")
#     print(f"  name = {name}\n  param.data = \n    {param.data}")



# #scheme4
# for layer in ae.modules():
#     if(isinstance(layer, nn.Conv2d)):
#         print(layer.weight)
