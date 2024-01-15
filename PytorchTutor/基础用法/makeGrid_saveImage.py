#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

保存图片主要使用save_image，定义如下：
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False)

注意：normalize=False代表只能将（0，1）的图片存储起来
     normalize=True将（0，255）的图片存储起来

"""

import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#内存分析工具
from memory_profiler import profile
import objgraph
import gc


#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)



def draw_images1(tmpout, generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (16, 10)):
    #generated_images = generated_images.reshape(examples, H, W)
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True
    # plt.ion()
    # for i in range(generated_images.shape[0]):
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(np.transpose(generated_images[i], (1,2,0)), cmap='gray', interpolation='none') # Greys   gray
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([] ) # #不显示y轴刻度值
            cnt += 1
    fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    plt.suptitle('Epoch: {}'.format(epoch, ), fontproperties=fontt,)

    out_fig = plt.gcf()
    out_fig.savefig(tmpout+"Generated_images_%d.png" % (epoch),  bbox_inches='tight')

    plt.show()
    # plt.close(fig)
    return




# 下载minist数据
dataset = datasets.MNIST(
    root='/home/jack/公共的/MLData',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=True
)

# 加载minist数据
dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True
)


#===================================================================================
# 保存图片
images, labels = next(iter(dataloader))
print(images.size())  # torch.Size([8, 1, 28, 28])


images = make_grid(images, nrow = 8, padding = 10 )
print(images.size())  # torch.Size([3, 84, 84])


savedir = "/home/jack/公共的/MLData/tmpout/"
os.makedirs(savedir, exist_ok = True)
#images = torch.randn(3, 314, 314)
save_image(images, savedir+'test.png',  )


#===================================================================================
# 其中如果tensor由很多小图片组成，则会自动调用make_grid()函数将小图片拼接为大图片再保存。

# 保存图片
# images, labels = next(iter(dataloader))
print(images.size())  # torch.Size([8, 1, 28, 28])


savedir = "/home/jack/公共的/MLData/tmpout/"
os.makedirs(savedir, exist_ok = True)
#images = torch.randn(3, 314, 314)
save_image(images, savedir+'test1.png', nrow = 8, padding = 10 )



















