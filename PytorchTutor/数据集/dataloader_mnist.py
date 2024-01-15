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
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import sys
import torch

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



def data_tf_mlp_mnist(x):
    ## 1
    # x = transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))

    ## 2
    x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

def data_tf_cnn_mnist_1(x):
    ## 1
    x = transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))

    # # 2
    # x = np.array(x, dtype='float32') / 255
    # # x = (x - 0.5) / 0.5
    # # x = x.reshape((-1,))  # (-1, 28*28)
    # x = x.reshape((1, 28, 28))  # ( 1, 28, 28)
    # x = torch.from_numpy(x)
    return x

def data_tf_cnn_mnist_batch(x):
    # ## 1
    # x = transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1, 1, 28, 28))

    # 2
    x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))  # (-1, 28*28)
    x = x.reshape((-1, 1, 28, 28))  # ( 1, 28, 28)
    x = torch.from_numpy(x)
    return x

root='/home/jack/公共的/MLData/'
tmpout = "/home/jack/SemanticNoise_AdversarialAttack/tmpout/"


batch_size = 25
trans = []

trans.append( transforms.ToTensor() )
# trans.append( transforms.Normalize([0.5], [0.5]) )
transform =  transforms.Compose(trans)

# data_tf = torchvision.transforms.Compose([ torchvision.transforms.Resize(28),
#                                         torchvision.transforms.ToTensor(),
#                                         torchvision.transforms.Normalize([0.5], [0.5])
#                                         ])

trainset =  datasets.MNIST(root = root, # 表示 MNIST 数据的加载的目录
                                      train = True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download = True, # 表示是否自动下载 MNIST 数据集
                                      transform = data_tf_cnn_mnist_1) # 表示是否需要对数据进行预处理，none为不进行预处理

testset =  datasets.MNIST(root = root, # 表示 MNIST 数据的加载的目录
                                      train = False,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download = True, # 表示是否自动下载 MNIST 数据集
                                      transform = data_tf_cnn_mnist_1) # 表示是否需要对数据进行预处理，none为不进行预处理
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 0

train_iter = DataLoader(trainset, batch_size=batch_size, shuffle = False,  )
test_iter = DataLoader(testset, batch_size=batch_size, shuffle = False,  )


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU.")

### (1)  data_tf_cnn_mnist_1
# trainset.data[0].shape
# Out[20]: torch.Size([28, 28])   0-255

# trainset[0][0].shape
# Out[21]: torch.Size([1, 28, 28])     0-1

### (2) transform
# trainset.data[0].shape
# Out[20]: torch.Size([28, 28])   0-255

# trainset[0][0].shape
# Out[21]: torch.Size([1, 28, 28])     0-1

### (3) data_tf_cnn_mnist_batch
# trainset.data[0].shape
# Out[34]: torch.Size([28, 28])    0-255

# trainset[0][0].shape
# Out[35]: torch.Size([1, 1, 28, 28])   0-1






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"len(trainset) = {len(trainset)}, len(train_iter) = {len(train_iter)}, ")
# batch_size = 25, len(trainset) = 60000, len(testset) = 10000, len(train_iter) = 2400, len(test_iter) = 400

import PIL
img = PIL.Image.fromarray(np.uint8(trainset.data[0]))




def data_inv_tf_mlp_mnist(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    recover_data = x * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = np.around(recover_data.detach().numpy() ).astype(np.uint8)
    return recover_data

# x.shape = (128, 1, 28, 28)
# recover_data = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_3D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    recover_data = x * 255
    recover_data = recover_data.reshape(( -1, 1, 28, 28))  # (-1, 28, 28)
    recover_data = np.around(recover_data.numpy()) # .astype(np.uint8)
    # recover_data =  recover_data.round() # .type(torch.uint8)
    return recover_data  #   (128, 1, 28, 28)

# x.shape = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_2D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    recover_data = x * 255
    recover_data = recover_data.reshape((-1, 28, 28))  # (-1, 28, 28)
    recover_data = np.around(recover_data.numpy()) # .astype(np.uint8)
    # recover_data =  recover_data.round() # .type(torch.uint8)

    return recover_data  # (128, 28, 28)




## plt.imshow()可以显示 numpy 也可以显示 tensor 数组.
def draw_images(tmpout, generated_images, labels, epoch,   dim = (5, 5), figsize = (10, 10)):
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            if len(generated_images[cnt].shape) == 2:
                axs[i, j].imshow(generated_images[cnt], cmap = 'Greys', interpolation='none') # Greys   gray
            elif len(generated_images[cnt].shape) == 3:
                if torch.is_tensor(generated_images):
                    axs[i, j].imshow(generated_images[cnt].permute(1,2,0), cmap = 'Greys', interpolation='none') # Greys   gray
                else:
                    axs[i, j].imshow(np.transpose(generated_images[cnt], (1,2,0)), cmap = 'Greys', interpolation='none') # Greys   gray
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([] ) # #不显示y轴刻度值
            font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18, 'color':'blue', }
            axs[i, j].set_title("label: {}".format(labels[cnt]),  fontdict = font1, )
            cnt += 1
    fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    plt.suptitle('Epoch: {}'.format(epoch, ), fontproperties=fontt,)

    out_fig = plt.gcf()
    out_fig.savefig(tmpout+"Generated_images_%d.png" % (epoch),  bbox_inches='tight')

    # plt.show()
    plt.close(fig)
    return


# use this general fun, images可以是tensor可以是numpy, 可以是(batchsize, 28, 28) 可以是(batchsize, 1/3, 28, 28)
def grid_imgsave(savedir, images, labels,  predlabs = '', dim = (4, 5), suptitle = '', basename = "raw_image"):
    rows = dim[0]
    cols = dim[1]
    if images.shape[0] != rows*cols:
        print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
        raise ValueError("img num and preset is inconsistent")
    figsize = (cols*2 , rows*2 + 1)
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True

    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            if len(images[cnt].shape) == 2:
                axs[i, j].imshow(images[cnt], cmap = 'Greys', interpolation='none') # Greys   gray
            elif len(images[cnt].shape) == 3:
                if torch.is_tensor(images):
                    axs[i, j].imshow(images[cnt].permute(1,2,0), cmap = 'Greys', interpolation='none') # Greys   gray
                else:
                    axs[i, j].imshow(np.transpose(images[cnt], (1,2,0)), cmap = 'Greys', interpolation='none') # Greys   gray
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([] ) # #不显示y轴刻度值
            font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18, 'color':'blue', }
            if predlabs != '':
                axs[i, j].set_title( r"$\mathrm{{label}}:{} \rightarrow {}$".format(labels[cnt], predlabs[cnt]),  fontdict = font1, )
            else:
                axs[i, j].set_title("label: {}".format(labels[cnt]),  fontdict = font1, )
            cnt += 1
    if suptitle != '':
        fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 22,   }
        plt.suptitle(suptitle, fontproperties=fontt,)

    out_fig = plt.gcf()
    out_fig.savefig( os.path.join(savedir, f"{basename}.png"),  bbox_inches='tight')
    plt.show()
    # plt.close(fig)
    return


for batch, (X, y) in enumerate(train_iter):
    X, y = X.to(device), y.to(device)
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")   # X的每个元素都是 0 - 1的.
    # draw_images(tmpout, data_inv_tf_cnn_mnist_batch_3D(X.cpu().detach()), y,  batch, dim = (5, 5),  figsize = (10, 10 + 1 ))
    # grid_imgsave(tmpout, data_inv_tf_cnn_mnist_batch_3D(X.cpu().detach()), y, dim = (5, 5), suptitle = 'ss', basename = "raw_image")
    X_hat_2D = data_inv_tf_cnn_mnist_batch_2D(X.cpu().detach())
    X_hat_3D = data_inv_tf_cnn_mnist_batch_3D(X.cpu().detach())
    print(f"X_hat_2D.shape = {X_hat_2D.shape}, X_hat_3D.shape = {X_hat_3D.shape}")
    print(f"{X.min()}, {X.max()}, {X_hat_2D.min()}, {X_hat_2D.max()},  {X_hat_3D.min()}, {X_hat_3D.max()}")
    if batch ==  0:
        break


##========================================================
# import PIL
# rows =  4
# cols = 5
# idx = np.arange(0, rows*cols, 1)
# labels = test_iter.dataset.targets[idx]
# # 真实图片
# real_image = test_iter.dataset.data[idx] #.numpy()
# for idx, (im, label) in enumerate(zip(real_image, labels)):
#     print(f"{idx}, {im.shape}, {label}")

#     im = PIL.Image.fromarray(im.numpy())
#     im.save(f"/home/jack/snap/{idx}_{label}.png")



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




##=====================================================================================

# print(f"trainset.train_data.size() = {trainset.train_data.size()}")            # trainset.train_data 的每个元素都是 0-255 的, 不随transform = 而变化
# print(f"trainset.test_data.size() = {trainset.test_data.size()}")             # trainset.train_data 的每个元素都是 0-255 的, 不随transform = 而变化
print(f"\ntrainset.data.size() = {trainset.data.size()}")                    # trainset.data 的每个元素都是 0-255 的, 不随 transform = 而变化, trainset.data是最深层次的原始数据, 不会改变的
# trainset.train_data.size() = torch.Size([60000, 28, 28])
# # trainset.test_data.size() = torch.Size([60000, 28, 28])
# # trainset.data.size() = torch.Size([60000, 28, 28])

# print(f"trainset.train_labels.size() = {trainset.train_labels.size()}")
# print(f"trainset.test_labels.size() = {trainset.test_labels.size()}")
# print(f"trainset.targets.size() = {trainset.targets.size()}")
# # trainset.train_labels.size() = torch.Size([60000])
# # trainset.test_labels.size() = torch.Size([60000])
# # trainset.targets.size() = torch.Size([60000])


# print(f"trainset.train_data[0].size() = {trainset.train_data[0].size()}")        # trainset.train_data 的每个元素都是 0-255 的
# print(f"trainset.test_data[0].size() = {trainset.test_data[0].size()}")         # trainset.test_data 的每个元素都是 0-255 的
print(f"trainset.data[0].size() = {trainset.data[0].size()}")                # trainset.data 的每个元素都是 0-255 的
print(f"trainset.data[0].max() = {trainset.data[0].max()}, trainset.data[0].min={trainset.data[0].min()}\n\n") # trainset.data[0] 的每个元素都是 0-255 的, 不随 transform = 而变化,
# # trainset.train_data[0].size() = torch.Size([28, 28])
# # trainset.test_data[0].size() = torch.Size([28, 28])
# # trainset.data[0].size() = torch.Size([28, 28])
# trainset.data[0].max() = 255, trainset.data[0].min=0

# print(f"trainset.train_labels[0] = {trainset.train_labels[0]}")
# print(f"trainset.test_labels[0] = {trainset.test_labels[0]}")
# print(f"trainset.targets[0] = {trainset.targets[0]}")
# # trainset.train_labels[0] = 5
# # trainset.test_labels[0] = 5
# # trainset.targets[0] = 5

print(f"trainset[0][0].size() = {trainset[0][0].size()}")                        # trainset[0][0] 的每个元素都是 0-1 的, 随transform = 而变化
print(f"trainset[0][0].max() = {trainset[0][0].max()}, trainset[0][0].min() = {trainset[0][0].min()}\n\n")
# print(f"trainset[0][1] = {trainset[0][1]}")
# # trainset[0][0].size() = torch.Size([1, 28, 28])
# # trainset[0][1] = 5


# print(f"train_iter.dataset.train_data.shape = {train_iter.dataset.train_data.shape}")
# print(f"train_iter.dataset.test_data.shape = {train_iter.dataset.test_data.shape}")
print(f"train_iter.dataset.data.size() = {train_iter.dataset.data.size()}")
# # train_iter.dataset.train_data.shape = torch.Size([60000, 28, 28])
# # train_iter.dataset.test_data.shape = torch.Size([60000, 28, 28])
# # train_iter.dataset.data.shape = torch.Size([60000, 28, 28])

# print(f"train_iter.dataset.train_labels.shape = {train_iter.dataset.train_labels.shape}")
# print(f"train_iter.dataset.test_labels.shape = {train_iter.dataset.test_labels.shape}")
# print(f"train_iter.dataset.targets.shape = {train_iter.dataset.targets.shape}")
# # train_iter.dataset.train_labels.shape = torch.Size([60000])
# # train_iter.dataset.test_labels.shape = torch.Size([60000])
# # train_iter.dataset.targets.shape = torch.Size([60000])

# print(f"train_iter.dataset.train_data[0].size() = {train_iter.dataset.train_data[0].size()}")        # train_iter.dataset.train_data 的每个元素都是 0-255 的
# print(f"train_iter.dataset.test_data[0].size() = {train_iter.dataset.test_data[0].size()}")          # train_iter.dataset.test_data 的每个元素都是 0-255 的
print(f"train_iter.dataset.data[0].size() = {train_iter.dataset.data[0].size()}")                    # train_iter.dataset.data 的每个元素都是 0-255 的
print(f"train_iter.dataset.data[0].max() = {train_iter.dataset.data[0].max()}, train_iter.dataset.data[0].min() = {train_iter.dataset.data[0].min()}\n\n")
# # train_iter.dataset.train_data[0].size() = torch.Size([28, 28])
# # train_iter.dataset.test_data[0].size() = torch.Size([28, 28])
# # train_iter.dataset.data[0].size() = torch.Size([28, 28])

# print(f"train_iter.dataset.train_labels[0] = {train_iter.dataset.train_labels[0]}")
# print(f"train_iter.dataset.test_labels[0] = {train_iter.dataset.test_labels[0]}")
# print(f"train_iter.dataset.targets[0] = {train_iter.dataset.targets[0]}")
# # train_iter.dataset.train_labels[0] = 5
# # train_iter.dataset.test_labels[0] = 5
# # train_iter.dataset.targets[0] = 5

print(f"train_iter.dataset[0][0].size() = {trainset[0][0].size()}")                        # trainset[0][0] 的每个元素都是 0-1 的
print(f"train_iter.dataset[0][0].max() = {train_iter.dataset[0][0].max()}, train_iter.dataset[0][0].min() = {train_iter.dataset[0][0].min()}")
# print(f"train_iter.dataset[0][1] = {trainset[0][1]}")
# # train_iter.dataset[0][0].size() = torch.Size([1, 28, 28])
# # train_iter.dataset[0][1] = 5





