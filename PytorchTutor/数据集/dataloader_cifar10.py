#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:30:45 2023

@author: jack
https://blog.csdn.net/QLeelq/article/details/121069095

"""


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import sys
import torch


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)


tmpout = "/home/jack/SemanticNoise_AdversarialAttack/tmpout/"

def draw_images( generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (10, 10)):
    #generated_images = generated_images.reshape(examples, H, W)
    fig = plt.figure(figsize = figsize, constrained_layout=True) #  constrained_layout=True
    # plt.ion()
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        #print(f"generated_images[i] = {generated_images[i].shape}")
        plt.imshow(np.transpose(generated_images[i], (1,2,0)), cmap='gray', interpolation='none') # Greys   gray
        plt.axis('off')

    fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    plt.suptitle('Epoch: {}'.format(epoch, ), fontproperties=fontt,)

    #plt.tight_layout(pad = 1.5 , h_pad=1, w_pad=0 )#  使得图像的四周边缘空白最小化
    # plt.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98, wspace=0.1, hspace=0)

    plt.savefig(tmpout+"Generated_images_%d.png" % (epoch),  bbox_inches='tight')
    # plt.suptitle('Epoch: {}, Iteration: {}'.format(epoch, iters), fontproperties=fontt, x=0.5, y=0.98,)
    # plt.savefig(self.args.tmpout+"Generated_images_%d_%d.png" % (epoch, iters),  bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    return


def draw_images1(tmpout, generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (10, 10)):
    #generated_images = generated_images.reshape(examples, H, W)
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True

    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt].permute(1,2,0)) # Greys   gray cmap='gray', interpolation='none'
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


def data_tf(x):
    print(f"{x.size}, {x}")
    x = x.resize((96, 96), 2)  # shape of x: (96, 96, 3)
    x = np.array(x, dtype='float32') / 255
    # print('shape of x:', np.shape(x))
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


root='/home/jack/公共的/MLData/CIFAR10'
# root='/home/jack/公共的/MLData/Mnist'
# root='/home/jack/公共的/MLData/FashionMNIST'


batch_size = 25
trans = []

resize = None

# if resize:
#     trans.append(torchvision.transforms.Resize(size = resize))

# trans.append( transforms.Resize(28) )
trans.append( transforms.ToTensor() )
# trans.append( transforms.Normalize([0.5], [0.5]) )


transform =  transforms.Compose(trans)

# MNIST  FashionMNIST    CIFAR10
trainset =  datasets.CIFAR10(root = root, # 表示 MNIST 数据的加载的目录
                                      train = True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download = True, # 表示是否自动下载 MNIST 数据集
                                      transform = transform) # 表示是否需要对数据进行预处理，none为不进行预处理

testset =  datasets.CIFAR10(root = root,
                                      train = False,
                                      download = True,
                                      transform = transform)

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 0

train_iter = DataLoader(trainset, batch_size=batch_size, shuffle = False, )
test_iter = DataLoader(testset, batch_size=batch_size, shuffle = False, )

print(f"len(trainset) = {len(trainset)}, len(testset) = {len(testset)}, len(train_iter) = {len(train_iter)}, len(test_iter) = {len(test_iter)}")
# len(trainset) = 50000, len(testset) = 10000, len(train_iter) = 2000, len(test_iter) = 400



print(f"trainset.data.shape = {trainset.data.shape}")           # trainset.data 的每个元素都是 0-255 的
print(f"len(trainset.targets) = {len(trainset.targets)}")       # list
print(f"trainset.data[0].shape = {trainset.data[0].shape}")     # trainset.data 的每个元素都是 0-255 的
print(f"trainset.targets[0] = {trainset.targets[0]}")
print(f"trainset[0][0].size() = {trainset[0][0].size()}")       # trainset[0][0] 的每个元素都是 0-1 的
print(f"trainset[0][1] = {trainset[0][1]}")
# trainset.data.shape = (50000, 32, 32, 3)
# len(trainset.targets) = 50000
# trainset.data[0].shape = (32, 32, 3)
# trainset.targets[0] = 6
# trainset[0][0].size() = torch.Size([3, 32, 32])
# trainset[0][1] = 6


print(f"train_iter.dataset.data.shape = {train_iter.dataset.data.shape}")
print(f"len(train_iter.dataset.targets) = {len(train_iter.dataset.targets)}")
print(f"train_iter.dataset.data[0].shape = {train_iter.dataset.data[0].shape}")                    # trainset.data 的每个元素都是 0-255 的
print(f"train_iter.dataset.targets[0] = {train_iter.dataset.targets[0]}")
print(f"train_iter.dataset[0][0].size() = {trainset[0][0].size()}")                        # trainset[0][0] 的每个元素都是 0-1 的
print(f"train_iter.dataset[0][1] = {trainset[0][1]}")
print(f"len(testset.targets) = {len(testset.targets)}")
# train_iter.dataset.data.shape = (50000, 32, 32, 3)
# len(train_iter.dataset.targets) = 50000
# train_iter.dataset.data[0].shape = (32, 32, 3)
# train_iter.dataset.targets[0] = 6
# train_iter.dataset[0][0].size() = torch.Size([3, 32, 32])
# train_iter.dataset[0][1] = 6
# len(testset.targets) = 10000




# print(trainset.train_data[0])
# plt.imshow(train_data.train_data[2].numpy(),cmap='Greys')
# plt.title('%i'%train_data.train_labels[2])
# plt.show()

for epoch, (X, y) in enumerate(train_iter):
    print(f"X.shape = {X.shape}, y.shape = {y.shape}, ")   # X的每个元素都是 0 - 1的.
    # draw_images1(tmpout, X,  epoch, 1, H = 32, W = 32, examples = 28,  dim = (5, 5), figsize = (10, 10))




#  cmap='Greys',  'gray'   'viridis'  Greys_r   Purples   binary  rainbow
# figsize=(10,10)
# dim=(5,5)

# # #  1
# fig = plt.figure(figsize=figsize)
# plt.tight_layout()
# for i in range(25):
#     plt.subplot(dim[0], dim[1], i+1,)
#     plt.imshow(X[i,0],  cmap = 'rainbow', interpolation='none', )
#     plt.axis('off')

# plt.show()
# # plt.savefig('Origin.png')


# #  2
# fig = plt.figure(figsize=figsize)
# plt.tight_layout()
# for i in range(25):
#     plt.subplot(dim[0], dim[1], i+1)
#     plt.imshow(X[i][0], cmap='gray', interpolation='none')#子显示
#     plt.axis('off')
# plt.show()










