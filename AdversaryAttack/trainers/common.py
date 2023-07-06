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
import numpy as np
from scipy import stats
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator




#### 本项目自己编写的库
sys.path.append("..")

from . import MetricsLog

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



# x.shape = (128, 1, 28, 28)
# recover_data = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_3D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    recover_data = x
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
    recover_data = x
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
#                                                   信源发生器
#============================================================================================================================

def uniform_sampling(n_sample, dim):
    # 均匀分布采样
    return np.random.uniform(0, 1, size=(n_sample, dim))


def normal_sampling(n_sample, dim):
    # normal gauss 分布采样
    return np.random.randn(n_sample, dim)


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


# X = np.ones((28, 28))*255
# Y = np.ones((28, 28))*254

# mse1 = ((X - Y)**2).mean()
# psnr1 = 10 * np.log10(255**2/mse1)


# x = (X/255.0 - 0.5)/0.5
# y = (Y/255.0 - 0.5)/0.5
# mse2 = ((x - y)**2).mean()
# psnr2 = 10 * np.log10(y.max()**2/mse2)


#============================================================================================================================
#                              AWGN: add noise
#============================================================================================================================


# 先将信号功率归一化，再计算噪声功率，再计算加躁的信号。
def AWGN(x, snr = 3 ):
    if snr == None:
        return x
    snr = 10**(snr / 10)
    noise_std = 1 / np.sqrt(snr)

    signal_power = x.type(torch.float32).pow(2).mean().sqrt()
    if signal_power > 1:
        print(f"signal_power = {signal_power}")
        x = torch.div(x, signal_power)

    x_output = x + torch.normal(0, noise_std, size = x.shape).to(x.device)
    return x_output


# x = torch.randint(low = 0, high = 10, size = (3, 4))
# print(f"x = {x}")

# y = AWGN(x, 3)
# print(f"x = {x}")
# print(f"y = {y}")


# 以实际信号功率计算噪声功率，再将信号加上噪声。
def Awgn(x, snr = 3):
    if snr == None:
        return x
    SNR = 10.0**(snr/10.0)
    # signal_power = ((x**2)*1.0).mean()
    signal_power = (x*1.0).pow(2).mean()
    noise_power = signal_power/SNR
    noise_std = torch.sqrt(noise_power)
    #print(f"x.shape = {x.shape}, signal_power = {signal_power}, noise_power={noise_power}, noise_std={noise_std}")

    noise = torch.normal(mean = 0, std = float(noise_std), size = x.shape)
    return x + noise.to(x.device)


# x = torch.randint(low = 0, high = 10, size = (3, 4))
# print(f"x = {x}")

# y = Awgn(x, 3)
# print(f"x = {x}")
# print(f"y = {y}")

#============================================================================================================================
#
#============================================================================================================================


#============================================================================================================================
#                                                画图代码
#============================================================================================================================


# 画出 分类正确率随攻击强度变换曲线图
def plotXY(X, Y, xlabel = "X", ylabel = "Y", title = "XY", legend = "Y vs. X", figsize = (10, 10), savepath = '~/tmp/', savename = 'hh'):
    fig, axs = plt.subplots(1, 1, figsize = figsize, constrained_layout=True )
    axs.plot(X, Y, color='b', linestyle='-', marker='*',markerfacecolor='b',markersize=12,  )

    ## 坐标轴的起始点
    axs.set_xlim(np.array(X).min() - 0.01, np.array(X).max() + 0.01 )   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
    axs.set_ylim(np.array(Y).min(),        np.array(Y).max() + 0.02 )         # ylim： 设置x、y坐标轴的起始点（从哪到哪）

    ## 设置坐标轴刻度
    axs.set_xticks(np.arange(0,  np.array(X).max() + 0.05, step = 0.1 ))         #  xticks： 设置坐标轴刻度的字体大小
    # axs.set_yticks(np.arange(0,  1.1,                      step = 0.2 ))          #  yticks： 设置坐标轴刻度的字体大小

    ## xlabel, ylabel 标签设置
    font3  = {'family':'Times New Roman','style':'normal','size':22}
    axs.set_xlabel(f"{xlabel}", fontproperties=font3)
    axs.set_ylabel(f"{ylabel}", fontproperties=font3)
    #axs.set_title(f"{title}",    fontproperties=font3)

    ## 设置 坐标轴的粗细
    axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    # ## 设置图例legend
    # font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
    # # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
    # legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
    # frame1 = legend1.get_frame()
    # frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    # labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
    # width——宽度：刻度线宽度（以磅为单位）。
    # 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
    axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=20, width=3, )
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(16) for label in labels]  # 刻度值字号

    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.93 , wspace=0.4, hspace=0.4)
    # 标题
    fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
    plt.suptitle(f"{title}", fontproperties=fontt, x = 0.5, y = 1,)

    out_fig = plt.gcf()
    out_fig.savefig(savepath + savename + '.pdf', format='pdf', bbox_inches = 'tight')
    out_fig.savefig(savepath + savename + '.eps', format='eps', bbox_inches = 'tight')

    plt.show()
    return


## 画出 原图和FGSM攻击后的图
def FGSM_draw_image( rows, cols, epsilons, examples, savepath = "", savename = "", suptitle = "", cmap = "Greys"): # gray
    cnt = 0
    fontx  = {'family':'Times New Roman','style':'normal','size' : 20, 'color':'blue', }
    fonty  = {'family':'Times New Roman','style':'normal','size' : 24, 'color':'blue', }
    # fonty  = {'family':'Times New Roman','style':'normal','size':14}
    ## plt.figure(figsize=(8, 10))
    if suptitle == '':
        fig, axs = plt.subplots(rows, cols, figsize = (cols * 2 , rows * 2 ), constrained_layout = True)
    else:
        fig, axs = plt.subplots(rows, cols, figsize = (cols * 2 , rows * 2 + 2), constrained_layout = True)
    for i in range(rows):
        for j in range(cols):
            cnt += 1
            orig, adv, ex = examples[i][j]
            # print(f"ex.shape = {ex.shape}")
            axs[i, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format( orig, adv ), fontdict = fontx) ## fontproperties=fontx
            axs[i, j].imshow(ex, cmap = cmap, interpolation='none')
            axs[i, j].set_xticks([])                                                  ## 不显示 x 轴刻度值
            axs[i, j].set_yticks([])                                                  ## 不显示 y 轴刻度值
            if j == 0:
                axs[i, j].set_ylabel("Eps: {}".format(epsilons[i]),  fontdict = fonty) ## fontproperties=fonty  fontsize = 14

    if suptitle != '':
        fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 22, }
        plt.suptitle(suptitle, fontproperties = fontt, x = 0.5, y = 0.99)
    plt.tight_layout()
    # plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1  , wspace = 0.2, hspace = 0.2)
    out_fig = plt.gcf()
    out_fig.savefig(savepath + savename + '.pdf', format='pdf', bbox_inches = 'tight')
    out_fig.savefig(savepath + savename + '.eps', format='eps', bbox_inches = 'tight')
    plt.show()
    return



# 画图 MNIST、FashionMnist或者Cifar10的网格图
def draw_images(tmpout, generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (16, 10)):
    #generated_images = generated_images.reshape(examples, H, W)
    fig = plt.figure(figsize = figsize, constrained_layout = True) #  constrained_layout=True
    # plt.ion()
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        #print(f"generated_images[i] = {generated_images[i].shape}")
        plt.imshow(generated_images[i].permute(1,2,0), cmap='gray', interpolation='none') # Greys   gray
        plt.axis('off')

    fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    plt.suptitle('Epoch: {}'.format(epoch, ), fontproperties=fontt,)

    out_fig = plt.gcf()
    out_fig.savefig(tmpout+"Generated_images_%d.png" % (epoch),  bbox_inches='tight')

    plt.show()
    # plt.close(fig)
    return

# 画图 MNIST、FashionMnist或者Cifar10的网格图
def draw_images1(tmpout, generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (16, 10)):
    #generated_images = generated_images.reshape(examples, H, W)
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True
    # plt.ion()
    # for i in range(generated_images.shape[0]):
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt].permute(1,2,0), cmap='gray', interpolation='none') # Greys   gray
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



# 画出高斯分布的生成曲线和实际曲线
def GAN_GeneGauss_plot(mean, std, generated_data, savepath, savename):
    x = np.arange(stats.norm(loc = mean, scale = std,).ppf(0.0001, ), stats.norm.ppf(0.9999, loc = mean, scale = std,), 0.1)
    pdf = stats.norm.pdf(x, loc = mean, scale = std,)

    fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True )

    counts, bins = np.histogram(generated_data, bins = 220, density = True)
    axs.stairs(counts, bins,  linestyle='-', linewidth = 2,  color = "blue", label = 'Generated Distribution',)
    # axs.hist(values, bins=220, density = True, color = "blue", label = 'Generated Distribution', )

    axs.plot(x, pdf, linestyle='-', linewidth = 2, color = 'r', label = r"$\mathrm{f(x)} = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$" )

    font3 = FontProperties(fname=fontpath + "simsun.ttf", size=18)
    font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18}
    legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号

    axs.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    #font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
    font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 24}
    #font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    axs.set_xlabel('Value', fontproperties = font3)
    axs.set_ylabel('Probility', fontproperties = font3)

    # fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    fontt  = {'family':'Times New Roman','style':'normal','size':26}
    #fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
    plt.suptitle('Histogram of Generated Distribution', fontproperties = fontt, )

    out_fig = plt.gcf()
    out_fig.savefig(savepath + savename + '.pdf', format='pdf', bbox_inches = 'tight')
    out_fig.savefig(savepath + savename + '.eps', format='eps', bbox_inches = 'tight')

    plt.show()

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



# 画出在指定压缩率和信噪比下训练 epoch 后的图片传输、恢复、分类效果
def R_SNR_epochImgs(savedir, model, classifier, dataloader, trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr, cols = 5, device = None, precision = "single"):
    model.eval()
    if not device:
        device = next(model.parameters()).device
    comparedir =  savedir + "/valiateImage"
    os.makedirs(comparedir, exist_ok = True)
    testloaderlen = dataloader.dataset.data.size(0)
    # 随机选 cols 张图片
    idx = np.random.randint(low = 0, high = testloaderlen, size = (cols, ))
    label = dataloader.dataset.targets[idx]
    # 原图
    real_image = dataloader.dataset.data[idx] #.numpy()

    with torch.no_grad():
        # 原图预处理
        test_data   =  data_tf_cnn_mnist_batch(real_image) # .view(-1, 1, 28, 28).type(torch.FloatTensor)/255.0
        test_data,  =  prepare(device, precision, test_data)
        # 原图预处理后分类
        labs_raw    = classifier(test_data).detach().cpu().argmax(axis = 1)
        # 传输
        im_result   = model(test_data)
        # 传输后分类
        labs_recover = classifier(im_result).detach().cpu().argmax(axis = 1)
        # 传输后图片的反预处理
        im_result =  data_inv_tf_cnn_mnist_batch_2D(im_result.detach().cpu() )

    rows =  2
    figsize = (cols*2 , rows*2)
    fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True) #  constrained_layout=True
    for j  in range(cols):
        axs[0, j].imshow(real_image[j], cmap='Greys')
        font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
        axs[0, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(label[j], labs_raw[j]),  fontdict = font1, )
        axs[0, j].set_xticks([] )  #  不显示x轴刻度值
        axs[0, j].set_yticks([] )  #  不显示y轴刻度值

        axs[1, j].imshow( im_result[j] , cmap='Greys')
        font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
        axs[1, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(label[j], labs_recover[j]),  fontdict = font1, )
        axs[1, j].set_xticks([] )  #  不显示x轴刻度值
        axs[1, j].set_yticks([] )  #  不显示y轴刻度值

        if j == 0:
            font = {'style': 'normal', 'size': 18, }
            axs[0, j].set_ylabel('Raw img', fontdict = font, labelpad = 8) # fontdict = font,
            axs[1, j].set_ylabel('Recovered img', fontdict = font, labelpad = 8)

    fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    fontt = {'style': 'normal', 'size': 20, }
    supt = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{epoch}}:{}, \mathrm{{PSNR}}_\mathrm{{train}}:{:.2f}\mathrm{{(dB)}}, \mathrm{{PSNR}}_\mathrm{{val}}:{:.2f}\mathrm{{(dB)}}$'.format(trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr)
    plt.suptitle(supt, fontproperties=fontt )

    out_fig = plt.gcf()
    out_fig.savefig(comparedir + f"/images_R={trainR:.1f}_trainSnr={trainSnr}(dB)_epoch={epoch}.png",  bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    return





















