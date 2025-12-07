#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:03:37 2023

@author: jack
"""

# 系统库
import math
import os, sys
import time
import datetime
import torch
# import torchvision
import numpy as np
# from scipy import stats
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator


#### 本项目自己编写的库
sys.path.append("..")

import MetricsLog

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype_np = lambda x, *args, **kwargs: x.astype(*args, **kwargs)
astype_tensor = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def try_gpu(i=0):
    """Return gpu device if exists, otherwise return cpu device."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


#============================================================================================================================
#                                                   data preporcess and recover
#============================================================================================================================

def data_inv_tf_mlp_mnist(x):
    """
    :param x:
    :return:
    """
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = np.around(recover_data.detach().cpu().numpy() ).astype(np.uint8)
    return recover_data


def data_tf_cnn_mnist_batch(x):
    # ## 1
    # x = torchvision.transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1, 1, 28, 28))

    ### 2
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))  # (-1, 28*28)
    x = x.reshape((-1, 1, 28, 28))  # ( 1, 28, 28)
    x = torch.from_numpy(x)
    return x



# x.shape = (128, 1, 28, 28)
# recover_data = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_3D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape(( -1, 1, 28, 28))  # (-1, 28, 28)
    # recover_data = np.around(recover_data.detach().numpy() ).astype(np.uint8)
    # recover_data =  recover_data.detach().type(torch.uint8)
    return recover_data  #   (128, 1, 28, 28)

# x.shape = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_2D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    # recover_data = x
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((-1, 28, 28))  # (-1, 28, 28)
    recover_data = np.around(recover_data.numpy()).astype(np.uint8)
    # recover_data =  recover_data.round().type(torch.uint8)
    return recover_data  # (128, 28, 28)


#============================================================================================================================
#                                                   计算正确率
#============================================================================================================================


"""
y.shape = torch.Size([12,])
y_hat.shape = torch.Size([12, 10])

y, y_hat是 tensor 或者 np 都可以
"""
def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    assert len(y.shape) == 1 and y.shape[0] == y_hat.shape[0]
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    if torch.is_tensor(y) and torch.is_tensor(y_hat):
        cmp = astype_tensor(y_hat, y.dtype) == y
        return float(reduce_sum(astype_tensor(cmp, y.dtype)))
    else:
        cmp = astype_np(y_hat, y.dtype) == y
        return float(reduce_sum(astype_np(cmp, y.dtype)))



# 计算在测试集上的正确率
def evaluate_accuracy_gpu(net, data_iter, device = None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    # Set the model to evaluation mode
    net.eval()
    if not device:
        device = next(net.parameters()).device
    # Accumulator has 2 parameters: (number of correct predictions, number of predictions)
    # print(f"device = {next(net.parameters()).device}")
    metric = MetricsLog.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # print(f"X.shape = {X.shape}, y.shape = {y.shape}, size(y) = {size(y)}/{y.size(0)}") # X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128]), size(y) = 128
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.size(0))  # size(y)
    return metric[0] / metric[1]


# 去归一化
def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def formatPrint2DArray(array):
    for linedata in array:
        for idx, coldata in enumerate(linedata):
            if idx == 0:
                print(f"{coldata:>8.3f}(dB)  ", end = ' ')
            else:
                print(f"{coldata:>8.3f}  ", end = ' ')
        print("\n", end = ' ')

    return
#============================================================================================================================
#                                                   定时器
#============================================================================================================================

class myTimer(object):
    def __init__(self, name = 'epoch'):
        self.acc = 0
        self.name = name
        self.timer = 0
        self.tic()
        self.start_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    def tic(self):  # time.time()函数返回自纪元以来经过的秒数。
        self.t0 = time.time()
        self.ts = self.t0

    # 返回从ts开始历经的秒数。
    def toc(self):
        diff = time.time() - self.ts
        self.ts = time.time()
        self.timer  += diff
        return diff

    def reset(self):
        self.ts = time.time()
        tmp = self.timer
        self.timer = 0
        return tmp

    def now(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # 从计时开始到现在的时间.
    def hold(self):
        return time.time() - self.t0



class Timer():
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        self.tik = time.time()
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def Sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


# tm = Timer()

# for epoch in range(10):
#     for j in range(3):
#         time.sleep(0.5)
#     tmp = tm.stop()
#     print(f"epoch = {epoch}, time:{tmp}/{tm.Sum()}")




#============================================================================================================================
#                                                   指标计算
#============================================================================================================================


def MSE_np_Batch(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    im, jm = np.float64(im), np.float64(jm)
    # X, Y = np.array(X).astype(np.float64), np.array(Y).astype(np.float64)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    return mse

def MSE_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)
    D = len(im.shape)
    if D < 2:
        raise ValueError('Input images must have >= 2D dimensions.')
    MSE = 0
    for i in range(im.shape[0]):
        MSE += MSE_np_Batch(im[i], jm[i], )

    avgmse = MSE/im.shape[0]
    return  avgmse, MSE, im.shape[0]


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr



def PSNR_np_Batch(im, jm, rgb_range = 255.0, cal_type = 'y'):
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)
    G = np.array( [65.481,   128.553,    24.966] )
    G = G.reshape(3, 1, 1)

    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        diff = diff * G
        diff = np.sum(diff, axis = -3) / rgb_range
    #print(f"diff.shape = {diff.shape}")
    mse = np.mean(diff**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = -10.0 * math.log10(mse)

    return psnr




def PSNR_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    D = len(im.shape)
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)

    PSNR = 0
    for i in range(im.shape[0]):
        if im[i].shape[0] == 1:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].shape[0] == 3:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = 'y')

    avgsnr = PSNR/im.shape[0]
    return  avgsnr, PSNR, im.shape[0]




#========================================================================================================================

def PSNR_torch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    maxp = max(im.max(), jm.max() )
    cr = torch.nn.MSELoss()
    mse = cr(jm, im)
    # out_np = X_hat.detach().numpy()
    psnr = 10 * np.log10(maxp ** 2 / mse.detach().numpy() ** 2)
    if psnr > 200:
        psnr = 200.0
    return  psnr

def PSNR_torch_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    # im, jm = torch.Tensor(im), torch.Tensor(jm)
    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        # gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs = [65.481, 128.553, 24.966]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1)
        diff = diff.mul(convert).sum(dim = -3) / rgb_range
    mse = diff.pow(2).mean().item()
    # print(f"mse = {mse}")
    if mse <= 1e-20:
        mse = 1e-20
    psnr =  -10.0 * math.log10( mse)

    return  psnr


def PSNR_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    # im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must 4-dimensions.')
    PSNR = 0
    for i in range(im.size(0)):
        if im[i].size(0) == 1:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].size(0) == 3:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = 'y')

    avgpsnr = PSNR/im.size(0)
    return  avgpsnr, PSNR, im.size(0)


def MSE_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    MSE = 0
    for i in range(im.size(0)):
        MSE += MSE_torch_Batch(im[i], jm[i] )

    avgmse = MSE/im.size(0)
    return  avgmse, MSE, im.size(0)



def MSE_torch_Batch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    diff = (im - jm)

    mse = diff.pow(2).mean().item()
    return mse

#============================================================================================================================
#
#============================================================================================================================

#  功能：将img每个像素点的至夹在[0,255]之间
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def prepare(device, precision, *Args):
    def _prepare(tensor):
        if  precision == 'half': tensor = tensor.half()
        return tensor.to( device)

    return [_prepare(a) for a in Args]





#============================================================================================================================
#
#============================================================================================================================


#============================================================================================================================
#                                                画图代码
#============================================================================================================================




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




















