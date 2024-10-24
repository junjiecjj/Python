#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:03:37 2023

@author: jack
"""

# 系统库
import math
import sys
import time
import datetime
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
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
sys.path.append("..")

from Trainer import MetricsLog

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
            #print(f"X.shape = {X.shape}, y.shape = {y.shape}, size(y) = {size(y)}")
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]



# 去归一化
def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



#============================================================================================================================
#                                                   定时器
#============================================================================================================================

class myTimer(object):
    def __init__(self, name = 'epoch'):
        self.acc = 0
        self.name = name
        self.timer = 0
        self.tic()

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
    # 均匀分布采样
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


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr



def PSNR_np_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
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

def MSE_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)
    D = len(im.shape)
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    MSE = 0
    for i in range(im.shape[0]):
        MSE += MSE_np_Batch(im[i], jm[i], )

    avgmse = MSE/im.shape[0]
    return  avgmse, MSE, im.shape[0]


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
def calc_metric(sr, hr, rgb_range, metrics, cal_type='y'):
    metric = []

    for met in metrics:
        if met == 'PSNR':
            psnr = PSNR_torch_Batch(sr, hr, rgb_range, cal_type='y')
        elif met == 'MSE':
            mse = MSE_torch_Batch(sr, hr,)
        else:
            m = 0
    metric.append(psnr)
    metric.append(mse)
    return torch.tensor(metric)



def PSNR_torch_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)

    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        # gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs = [65.481, 128.553, 24.966]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1)
        diff = diff.mul(convert).sum(dim = -3) / rgb_range
    mse = diff.pow(2).mean().item()
    # print(f"mse = {mse}")
    if mse <= 1e-10:
        mse = 1e-10
    psnr =  -10.0 * math.log10( mse)

    return  psnr


def PSNR_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    PSNR = 0
    for i in range(im.size(0)):
        if im[i].size(0) == 1:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].size(0) == 3:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = 'y')

    avgsnr = PSNR/im.size(0)
    return  avgsnr, PSNR, im.size(0)

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

    im, hr = torch.Tensor(im), torch.Tensor(jm)
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
#                                                   FGSM算法攻击代码
#============================================================================================================================
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # print(f"image.shape = {image.shape}, sign_data_grad.shape = {sign_data_grad.shape}, perturbed_image.shape = {perturbed_image.shape}")
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


#============================================================================================================================
#                                                画图代码
#============================================================================================================================


# 画出 分类正确率随攻击强度变换曲线图
def plotXY(X, Y, xlabel = "X", ylabel = "Y", title = "XY", legend = "Y vs. X", figsize = (10, 10), savepath = '~/tmp/', savename = 'hh'):
    fig, axs = plt.subplots(1, 1, figsize = figsize, constrained_layout=True )
    axs.plot(X, Y, color='b', linestyle='-', marker='*',markerfacecolor='b',markersize=12,  )

    ## 坐标轴的起始点
    axs.set_xlim(np.array(X).min() , np.array(X).max()+0.01)   # xlim: 设置x、y坐标轴的起始点（从哪到哪）
    axs.set_ylim(np.array(Y).min() , np.array(Y).max()+0.01) # ylim： 设置x、y坐标轴的起始点（从哪到哪）

    ## 设置坐标轴刻度
    axs.set_xticks(np.arange(0, .35, step=0.05))         #  xticks： 设置坐标轴刻度的字体大小
    axs.set_yticks(np.arange(0, 1.1, step=0.2))          #  yticks： 设置坐标轴刻度的字体大小

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
    plt.suptitle(f"{title}", fontproperties=fontt, x=0.5, y=1,)
    #

    out_fig = plt.gcf()
    out_fig.savefig(savepath + savename + '.pdf', format='pdf', bbox_inches = 'tight')
    out_fig.savefig(savepath + savename + '.eps', format='eps', bbox_inches = 'tight')

    plt.show()
    return


# 画出 原图和FGSM攻击后的图
def FGSM_draw_image(rows, cols, epsilons, examples, savepath, savename ):
    cnt = 0
    fontx  = {'family':'Times New Roman','style':'normal','size':14}
    fonty  = {'family':'Times New Roman','style':'normal','size':14}
    # plt.figure(figsize=(8, 10))
    fig, axs = plt.subplots(rows, cols, figsize=(6, 10), )
    for i in range(rows):
        for j in range(cols):
            cnt += 1
            orig, adv, ex = examples[i][j]
            axs[i, j].set_title("{} -> {}".format(orig, adv), fontproperties=fontx) # fontproperties=fontx
            axs[i, j].imshow(ex, cmap="gray")
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([]) # #不显示y轴刻度值
            if j == 0:
                axs[i, j].set_ylabel("Eps: {}".format(epsilons[i]),  fontproperties=fonty) # fontproperties=fonty  fontsize = 14

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1  , wspace=0.2, hspace=0.2)
    out_fig = plt.gcf()
    out_fig.savefig(savepath + savename + '.pdf', format='pdf', bbox_inches = 'tight')
    out_fig.savefig(savepath + savename + '.eps', format='eps', bbox_inches = 'tight')
    plt.show()
    return


# 画图 MNIST、FashionMnist或者Cifar10的网格图
def draw_images(tmpout, generated_images, epoch, iters, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (16, 10)):
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
            axs[i, j].imshow(np.transpose(generated_images[cnt], (1,2,0)), cmap='gray', interpolation='none') # Greys   gray
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
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 24}
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

































