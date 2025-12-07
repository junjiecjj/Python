#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 03:54:22 2025

@author: jack
"""


# 系统库
import math
import os, sys
import time
import datetime
import torch
import PIL
import numpy as np
# from scipy import stats
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#### 本项目自己编写的库
from Logs import Accumulator

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

def prepare(device, precision, *Args):
    def _prepare(tensor):
        if  precision == 'half': tensor = tensor.half()
        return tensor.to( device)

    return [_prepare(a) for a in Args]

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
#                              AWGN: add noise
#============================================================================================================================

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
    # plt.show()
    plt.close(fig)
    return


#%%

# 在指定压缩率和信噪比下训练间隔epoch时, 在验证集上的测试结果
def validate( args, model, classifier, dataloader, ):
    model.eval()
    classifier.eval()
    metric  = Accumulator(4)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X,             = prepare(args.device, args.precision, X)
            # 传输
            X_hat          = model(X)
            # 传输后分类
            y_hat          = classifier(X_hat).detach().cpu()
            X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
            # 计算准确率
            acc            = (y_hat.argmax(axis=1) == y).sum().item()
            # 不同方法计算 PSNR
            # batch_01_psnr  =  PSNR_torch(X , X_hat , )
            X              = data_inv_tf_cnn_mnist_batch_3D(X)
            X_hat          = data_inv_tf_cnn_mnist_batch_3D(X_hat)
            batch_avg_psnr = PSNR_torch_Batch(X, X_hat, )

            metric.add( batch_avg_psnr, acc, 1, X.size(0))
        # val_batch_01 = metric[0]/metric[3]
        val_batch    = metric[0]/metric[2]
        val_acc      = metric[1]/metric[3]
    return val_batch,  val_acc


# 画出在指定压缩率和信噪比下训练 epoch 后, 随机选中的10张图片传输、恢复、分类效果
def R_SNR_epochImgs(args, ckp, model, classifier, dataloader, trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr, cols = 5 ):
    model.eval()
    classifier.eval()
    comparedir = ckp.savedir + "/MiddleImage"
    os.makedirs(comparedir, exist_ok = True)
    testloaderlen = dataloader.dataset.data.size(0)
    # 随机选 cols 张图片
    idx = np.random.randint(low = 0, high = testloaderlen, size = (cols, ))
    y = dataloader.dataset.targets[idx]
    # 原图
    X = dataloader.dataset.data[idx]

    with torch.no_grad():
        # 原图预处理
        X_raw   =  data_tf_cnn_mnist_batch(X) # .view(-1, 1, 28, 28).type(torch.FloatTensor)/255.0
        # X_raw   = X_raw.to(args.device)
        X_raw, = prepare(args.device, args.precision, X_raw)
        # 原图预处理后分类
        y_raw    = classifier(X_raw).detach().cpu().argmax(axis = 1)
        # 传输
        X_transmit   = model(X_raw)
        # 传输后分类
        y_hat = classifier(X_transmit).detach().cpu().argmax(axis = 1)
        # 传输后图片的反预处理
        X_transmit = X_transmit.detach().cpu()
        X_transmit =  data_inv_tf_cnn_mnist_batch_2D(X_transmit)

    rows =  2
    figsize = (cols*2 , rows*2)
    fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True) #  constrained_layout=True
    for j  in range(cols):
        axs[0, j].imshow(X[j], cmap='Greys')
        font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
        axs[0, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(y[j], y_raw[j]),  fontdict = font1, )
        axs[0, j].set_xticks([] )  #  不显示x轴刻度值
        axs[0, j].set_yticks([] )  #  不显示y轴刻度值

        axs[1, j].imshow( X_transmit[j] , cmap='Greys')
        font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
        axs[1, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(y[j], y_hat[j]),  fontdict = font1, )
        axs[1, j].set_xticks([] )  #  不显示x轴刻度值
        axs[1, j].set_yticks([] )  #  不显示y轴刻度值

        if j == 0:
            font = {'style': 'normal', 'size': 18, }
            axs[0, j].set_ylabel('Raw img', fontdict = font, labelpad = 8) # fontdict = font,
            axs[1, j].set_ylabel('Recovered img', fontdict = font, labelpad = 8)

    fontt = {'style': 'normal', 'size': 20, }
    supt = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{epoch}}:{}, \mathrm{{PSNR}}_\mathrm{{train}}:{:.2f}\mathrm{{(dB)}}, \mathrm{{PSNR}}_\mathrm{{val}}:{:.2f}\mathrm{{(dB)}}$'.format(trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr)
    plt.suptitle(supt, fontproperties=fontt )

    out_fig = plt.gcf()
    out_fig.savefig(comparedir + f"/images_R={trainR:.1f}_trainSnr={trainSnr}(dB)_epoch={epoch}.png",  bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    return


## 保存指定压缩率和信噪比下训练完后, 画出在所有测试信噪比下的 10张 图片传输、恢复、分类示例.
def R_SNR_valImgs(ckp, args, model, classifier, dataloader, trainR = 0.1, tra_snr = 2, snrlist = np.arange(-2, 10, 2) ):
    model.eval()
    classifier.eval()
    savedir = os.path.join( ckp.testResdir, f"Images_compr={trainR:.1f}_trainSnr={tra_snr}(dB)" )
    os.makedirs(savedir, exist_ok = True)
    rows =  4
    cols = 5
    # 固定的选前几张图片
    idx = np.arange( rows*cols )
    y = dataloader.dataset.targets[idx]
    # 原图
    X  = dataloader.dataset.data[idx]
    # 原图的预处理
    X_raw   = data_tf_cnn_mnist_batch(X)
    X_raw,  = prepare(args.device, args.precision, X_raw)
    # 原图预处理后分类
    y_raw    = classifier(X_raw).detach().cpu().argmax(axis = 1)
    raw_dir  = os.path.join(ckp.testResdir, "raw_image")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok = True)
        for idx, (im, label) in enumerate(zip(X, y)):
            im = PIL.Image.fromarray(im.numpy())
            im.save(os.path.join(raw_dir, f"{idx}_{label}.png"))
        grid_imgsave(raw_dir, X, y, predlabs = y_raw, dim = (rows, cols), suptitle = "Raw images", basename = "raw_grid_images")

    # 开始遍历测试信噪比
    with torch.no_grad():
        for snr in snrlist:
            subdir = os.path.join(savedir, f"testSNR={snr}(dB)")
            os.makedirs(subdir, exist_ok = True)
            model.set_snr(snr)
            # 传输
            X_trans = model(X_raw)
            # 传输后分类
            y_hat = classifier(X_trans).detach().cpu().argmax(axis = 1)
            # 自编码器恢复的图片
            X_trans = X_trans.detach().cpu()
            X_trans = data_inv_tf_cnn_mnist_batch_2D(X_trans)
            for idx, (im, label) in enumerate(zip(X_trans, y)):
                im = PIL.Image.fromarray(im )
                im.save(os.path.join(subdir, f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)_{idx}_{label}.png"))
            a = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr, snr)
            bs = f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)"
            grid_imgsave(subdir, X_trans, y, predlabs = y_hat, dim = (rows, cols), suptitle = a, basename = "grid_images_" + bs )
    return

# 在指定压缩率和信噪比下训练完所有的epoch后, 在测试集上的指标统计
def test_R_snr(ckp, testRecoder, args, model, classifier, dataloader, compr, tasnr, SNRlist = np.arange(-2, 10, 2), ):
    tm = myTimer()
    model.eval()
    classifier.eval()
    ckp.write_log(f"#=============== 开始在 压缩率:{compr:.1f}, 信噪比:{tasnr}(dB)下测试, 开始时刻: {tm.start_str} ================\n")
    ckp.write_log("  {:>12}  {:>12}  {:>12} ".format("测试信噪比", "acc", "avg_batch" ))
    print( f"    压缩率:{compr:.1f}, 信噪比: {tasnr} (dB), 测试集:")
    # 增加 指定压缩率和信噪比下训练的模型的测试条目
    testRecoder.add_item(compr, tasnr,)
    # with torch.no_grad():
    for snr in SNRlist:
        model.set_snr(snr)
        # 增加条目下的 一个测试信噪比
        testRecoder.add_snr(compr, tasnr, snr)
        metric             = Accumulator(4)
        for batch, (X, y) in enumerate(dataloader):
            X,             = prepare(args.device, args.precision, X)
            X_hat          = model(X)
            # 传输后分类
            y_hat          = classifier(X_hat).detach().cpu()
            # 计算准确率
            X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
            acc            = (y_hat.argmax(axis=1) == y).sum().item()
            # batch_01_psnr  = PSNR_torch(X, X_hat, )
            X              = data_inv_tf_cnn_mnist_batch_3D(X)
            X_hat          = data_inv_tf_cnn_mnist_batch_3D(X_hat)
            batch_avg_psnr = PSNR_torch_Batch(X, X_hat, )
            metric.add(acc,  batch_avg_psnr, 1, X.size(0))
        acc          = metric[0]/metric[3]
        avg_batch    = metric[1]/metric[2]

        met = torch.tensor([acc, avg_batch ])
        testRecoder.assign(compr, tasnr, met)
        ckp.write_log(f"  {snr:>10}, {acc:>12.3f}, {avg_batch:>12.3f} ")
        print( f"  {snr:>10}(dB), {acc:>12.3f}, {avg_batch:>12.3f} ")
    testRecoder.save(ckp.testResdir,)
    testRecoder.plot_inonefig1x2(ckp.testResdir, metric_str = ['acc', 'batch_PSNR', ], tra_compr = compr, tra_snr = tasnr,)
    return























