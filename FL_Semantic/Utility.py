
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

import os
# import sys
# import math
import datetime
# import imageio
# import cv2
# import skimage
# import glob
import random
import numpy as np
import torch

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

# matplotlib.use('WXagg')
import matplotlib.pyplot as plt

import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
# from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#### 本项目自己编写的库
# sys.path.append("..")


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



# 初始化随机数种子
def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
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


## 功能：
class checkpoint():
    def __init__(self, args ):
        print( "\n#================================ checkpoint 开始准备 =======================================\n")
        self.args = args
        self.ok = True
        self.n_processes = 8

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.savedir = os.path.join(args.save_path, f"{self.now}_{args.modelUse}")
        os.makedirs(self.savedir, exist_ok = True)

        print(f"训练结果保存目录 = {self.savedir} ")

        # 模型参数保存的目录
        # self.modelSaveDir = os.path.join(args.ModelSave, f"{self.now}_Model_{args.modelUse}")
        self.modelSaveDir = self.args.ModelSave
        os.makedirs(self.modelSaveDir, exist_ok = True)

        # open_type = 'a' if os.path.exists(self.getSavePath('trainLog.txt')) else 'w'
        # self.log_file = open(self.getSavePath('trainLog.txt'), open_type)
        self.writeArgsLog(self.getSavePath('argsConfig.txt'))

        # Prepare test dir and so on:
        self.testResdir = os.path.join(self.savedir, "test_results")
        os.makedirs(self.testResdir, exist_ok=True)
        print(f"测试结果保存目录 = {self.testResdir} ")

        self.cdf_pdf = os.path.join(self.savedir, "cdf_pdf")
        os.makedirs(self.cdf_pdf, exist_ok=True)
        print(f"cdf, pdf 结果保存目录 = {self.cdf_pdf} ")

        print( "\n#================================ checkpoint 准备完毕 =======================================\n")
        return

    def writeArgsLog(self, filename, open_type = 'w'):
        with open(filename, open_type) as f:
            f.write('====================================================================================\n')
            f.write(self.now + '\n')
            f.write('====================================================================================\n\n')

            f.write("###############################################################################\n")
            f.write("################################  args  #######################################\n")
            f.write("###############################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}\n")
            f.write("\n################################ args end  ##################################\n")
        return


    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)

    # 写日志
    def write_log(self, log, train = False ):
        if train == True:
            logfile = self.getSavePath('trainLog.txt')
        else:
            logfile = self.get_testSavepath('testLog.txt')
        with open(logfile, 'a+') as f:
            f.write(log + '\n')
        return

    # 写日志
    def write_attacklog(self, log ):
        logfile = self.getSavePath('AttackLog.txt')
        with open(logfile, 'a+') as f:
            f.write(log + '\n')
        return

# >>> 测试相关函数
    # 初始化测试结果目录
    def get_testSavepath(self, *subdir):
        return os.path.join(self.testResdir, *subdir)


# <<< 测试相关函数

def localUpdateCDF1(round_idx, client, para_w, round_cdf_pdf):
    # params_float = np.empty((0, 0))
    data = torch.Tensor()

    for key, val in para_w.items():
        if 'float' in str(val.dtype):
            data = torch.cat((data, val.detach().clone().cpu().flatten()))

    # mean = data.mean()
    # var = data.var()
    # print(f"    round {round_idx}, {client}, {data.numel()}, mean = {mean}, var = {var}")
    fig, axs = plt.subplots(2,1, figsize=(8, 10), constrained_layout=True)
    # torch.save(data, "/home/jack/snap/test.pt")
    ##======================= subfig 1 ================================================
    weights = np.ones_like(data)/float(len(data))
    re = axs[0].hist(data, bins = 500, density=True, histtype='bar', color='yellowgreen', alpha=0.75, label = 'pdf')
    torch.save(re[:2], f"{round_cdf_pdf}/round={round_idx}_{client}_pdf.pt")
    # print(f"   {re[0].min()}, {re[0].max()}, {re[1].min()}, {re[1].max()}")
    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    #font1  = {'family':'Times New Roman','style':'normal','size':17}
    axs[0].set_xlabel(r'值', fontproperties=font1)
    axs[0].set_ylabel(r'概率', fontproperties=font1)
    font1  = {'family':'Times New Roman','style':'normal','size':22}
    axs[0].set_title('PDF', fontproperties=font1)
    axs[0].grid()

    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
    #font1  = {'family':'Times New Roman','style':'normal','size':22}
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
    # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
    legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, framealpha=1)
    # frame1 = legend1.get_frame()
    # frame1.set_alpha(1)

    # labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
    # width——宽度：刻度线宽度（以磅为单位）。
    # 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
    axs[0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, )
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号


    axs[0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    ##======================= subfig 1 ================================================
    # 我们只需更改直方图的图像类型，令histtype=‘step’，就会画出一条曲线来（Figure3，实际上就是将直方柱并在一起，除边界外颜色透明），类似于累积分布曲线。这时，我们就能很好地观察到不同数据分布曲线间的差异。
    #cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
    re1 = axs[1].hist(data, bins = 500, density=True, histtype='step', facecolor='red', alpha=0.75, cumulative=True, rwidth=0.8, label = 'CDF')
    torch.save(re1[:2], f"{round_cdf_pdf}/round={round_idx}_{client}_cdf.pt")
    font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
    #font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
    #font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

    axs[1].set_xlabel(r'值', fontproperties = font2)
    axs[1].set_ylabel(r'累计概率', fontproperties = font2)
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
    axs[1].set_title('CDF', fontproperties = font2)
    axs[1].grid()

    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
    #font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
    legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2, framealpha=1 )
    # frame1 = legend1.get_frame()
    # frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明

    # x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
    # axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
    axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号

    axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    name = f"/round={round_idx}_{client}.eps"
    savedir =   round_cdf_pdf + name
    out_fig = plt.gcf()
    out_fig .savefig(savedir)
    #out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
    plt.close()
    return




def localUpdateCDF(round_idx, client, para_w, round_cdf_pdf):
    # params_float = np.empty((0, 0))
    data = torch.Tensor()

    for key, val in para_w.items():
        if 'float' in str(val.dtype):
            data = torch.cat((data, val.detach().clone().cpu().flatten()))

    torch.save(data, f"{round_cdf_pdf}/round={round_idx}_{client}.pt")

    return







































































































































































