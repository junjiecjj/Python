
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""

import os, sys
# import math
# import datetime
# import imageio
# import cv2
# import skimage
# import glob
import scipy
import numpy as np
import torch
import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm


#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



# 初始化随机数种子
def set_random_seed(seed = 999999,):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def set_printoption(precision = 3):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(
        precision = precision,    # 精度，保留小数点后几位，默认4
        threshold = 1000,
        edgeitems = 3,
        linewidth = 150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile = None,
        sci_mode = False  # 用科学技术法显示数据，默认True
    )


def Initial(args):
    theta_true = np.random.randn(args.D, 1)
    theta0 = np.zeros((args.D, 1))
    X = {}
    Y = {}
    for client in range(args.num_of_clients):
        X[f"client{client:d}"] = np.random.randn(args.local_ds, args.D)
        noise = np.random.normal(loc = 0.0, scale = args.sigma, size = (args.local_ds, 1))
        Y[f"client{client:d}"] = X[f"client{client:d}"] @ theta_true + noise
    frac = {}
    tot_data_size = np.sum([len(val) for key, val in X.items()])
    for key, val in X.items():
        frac[key] = len(val)/tot_data_size
    return  theta_true, theta0, X, Y, frac


def optimal_linear_reg(X, Y, frac):
    XTX = np.sum([data.T @ data for client, data in X.items()], axis = 0)
    XTY = np.sum([data.T @ Y[client] for client, data in X.items()], axis = 0)
    theta_optim = np.linalg.inv(XTX) @ XTY
    optim_Fw = np.sum([0.5 * frac[client] * np.linalg.norm(data @ theta_optim - Y[client], ord = 2 )**2/len(data) for client, data in X.items()])
    return theta_optim, optim_Fw


def mess_stastic(args, comm_round, savedir, message_lst):
    allmes = np.array(message_lst).flatten()
    fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)

    lb = f"{args.case},E = {args.local_up if args.case != 'gradient' else 1}, {args.channel}, {"SNR="+args.SNR+"(dB)" if args.channel != 'erf' else ''},round={comm_round}"
    # count, bins, ignored = axs.hist(allmes, density=True, bins='auto', histtype='stepfilled', alpha=0.5, facecolor = "#0099FF", label= lb, zorder = 4)

    mu  = allmes.mean()
    std = allmes.std()
    X = np.linspace(allmes.min(), allmes.max(), 100)
    N_pdf = scipy.stats.norm.pdf(X, loc = mu, scale = std)
    axs.plot(X, N_pdf, c = 'b', lw = 2, label = f"N({mu}, {std**2})")

    ## 直方图和核密度估计图
    sns.histplot(allmes, kde = True,  ax=axs, stat = "density", bins = 50, color='skyblue')
    axs.axvline(x=mu, color = 'r', linestyle = '--')
    axs.axvline(x=mu + std, color = 'r', linestyle = '--')
    axs.axvline(x=mu - std, color = 'r', linestyle = '--')

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
    # legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    # frame1 = legend1.get_frame()
    # frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
    axs.set_xlabel('Value', fontdict = font, )
    axs.set_ylabel('Density', fontdict = font, )
    axs.set_title(f"{lb}", fontdict = font,  )

    axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

    # 显示图形
    out_fig = plt.gcf()
    out_fig.savefig(savedir + f'_round{comm_round}.eps', bbox_inches='tight', pad_inches=0,)
    plt.show()
    return










































































































































































