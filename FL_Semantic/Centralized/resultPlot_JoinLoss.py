
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

"""

import os
import sys


from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import numpy as np
import imageio
import torch.nn as nn
import torch

from transformers import optimization
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

import glob
import socket, getpass , os
import numpy as np

# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')

# 本项目自己编写的库
# from option import args
sys.path.append("..")
# from  ColorPrint import ColoPrint
# color =  ColoPrint()



fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)


jpeg_comp     =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  ]
jpeg_SNR      =  [ -2, -1, 0,  1,  2 , 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40]


jpeg_res = np.array([
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 21.906,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 21.906, 27.145, 31.447,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 23.617, 29.329, 34.721, 39.869,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.486, 17.314, 22.020, 29.329, 35.846, 41.591, 45.330,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 13.954, 21.787, 23.956, 25.708, 27.215, 34.790, 41.591, 45.671, 46.674,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 15.291, 22.363, 24.593, 26.403, 28.142, 29.859, 31.576, 39.869, 45.330, 46.674, 46.733,  ],
    [ 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 10.466, 11.255, 21.545, 24.282, 26.337, 28.209, 30.121, 32.047, 33.983, 35.911, 43.789, 46.573, 46.732, 46.733,  ]])


# 功能：
class ResultPlot():
    def __init__(self, ):
        self.mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        self.color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
        self.alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

        self.CompRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  ]
        # self.CompRate = np.around(self.CompRate , decimals = 1, )
        self.SNRtrain = [1, 3, 10, 20 ]

        self.SNRtest = np.arange(-2, 21, 1)
        self.SNRtest = np.append(self.SNRtest, 25)
        self.SNRtest = np.append(self.SNRtest, 30)
        self.SNRtest = np.append(self.SNRtest, 35)
        self.SNRtest = np.append(self.SNRtest, 40)
        self.SNRtest.sort()

        self.rootdir = f"{user_home}/SemanticNoise_AdversarialAttack/results/"
        self.home = f"{user_home}"
        self.savedir = os.path.join(self.home, 'SemanticNoise_AdversarialAttack/Figures')
        os.makedirs(self.savedir, exist_ok=True)

        # 2023-06-08-13:30:19_AE_cnn_classify_mnist_R_noiseless
        # 2023-06-08-13:38:21_AE_noqua_cnn_classify_joinloss_mnist_R_noiseless
        date1 = "2023-06-08-13:38:21"
        data1 = "MNIST"
        name1 = f"_AE_noqua_cnn_classify_joinloss_{data1.lower()}_R_noiseless"
        self.res_R_noiseless = torch.load(os.path.join(self.rootdir, date1 + name1, "test_results/TesRecorder_TeMetricLog.pt"))

        # 2023-06-09-21:39:56_AE_noqua_cnn_classify_joinloss_mnist_R_SNR
        date2 = "2023-06-09-21:39:56"
        data2 = "MNIST"
        name2 = f"_AE_noqua_cnn_classify_joinloss_{data2.lower()}_R_SNR"
        self.res_R_SNR = torch.load(os.path.join(self.rootdir, date2 + name2, "test_results/TesRecorder_TeMetricLog.pt"))

        return

    # 画出 JPEG 和 JSCC (包括在无噪训练和指定信噪比 tra_test_snr 训练) 在指定测试信噪比 tra_test_snr, 指定压缩率 Rlist下训练完后,  PSNR 随 压缩率 Rlist 的曲线. 每条曲线对应一个 tra_test_snr
    def FixedSNRtraintest_PSNRorAcc_vs_R(self, tra_test_snr = [ 3,  20, ], Rlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], PSNR = True, col_idx = 3):
        width = 6
        high  = 5
        fig, axs = plt.subplots(1, 1, figsize=(width, high), constrained_layout = True)# constrained_layout=True
        ## JEPG
        if PSNR == True:
            for i, snr in enumerate(tra_test_snr):
                if snr in jpeg_SNR:
                    col = jpeg_SNR.index(snr)
                    psnr_jpg = []
                    R = []
                    for j, cr in enumerate(Rlist):
                        if cr in jpeg_comp:
                            R.append(cr)
                            raw = jpeg_comp.index(cr)
                            psnr_jpg.append(jpeg_res[raw, col])
                    if R != [] and psnr_jpg != [] and len(R) ==len(psnr_jpg):
                        axs.plot(R, psnr_jpg, label = f"JPEG, {snr} dB", color = 'b',linestyle = '-', marker = self.mark[i], markersize = 8,)
                    else: pass

        ## R, noiseless
        for i, snr in enumerate(tra_test_snr):
            if snr in self.SNRtest:
                row = np.where(self.SNRtest == snr)[0][0]
                psnr_less = []
                R = []
                for cr in Rlist:
                    if cr in self.CompRate:
                        tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain=noiseless(dB)"
                        psnr_less.append(self.res_R_noiseless[tmps][row, col_idx])
                        R.append(cr)
                if R != [] and psnr_less != [] and len(R) ==len(psnr_less):
                    lb = r"$\mathrm{{SNR}}_\mathrm{{train}}:\mathrm{{noiseless}}, \mathrm{{SNR}}_\mathrm{{test}}={} \mathrm{{dB}}$".format(snr)
                    # lb = "noiseless"
                    axs.plot(R, psnr_less, label = lb, color = 'r', linestyle = '-', marker = self.mark[i], markersize = 8,)
                else: pass

        ## R, SNR
        for i, snr in enumerate(tra_test_snr):
            if (snr in self.SNRtest) and (snr in self.SNRtrain):
                row = np.where(self.SNRtest == snr)[0][0]
                psnr_snr = []
                R = []
                for cr in Rlist:
                    if cr in self.CompRate:
                        R.append(cr)
                        tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain={snr}(dB)"
                        psnr_snr.append(self.res_R_SNR[tmps][row, col_idx])
                if R != [] and psnr_snr != [] and len(R) ==len(psnr_snr):
                    lb = r"$\mathrm{{SNR}}_\mathrm{{train}}={} \mathrm{{dB}}, \mathrm{{SNR}}_\mathrm{{test}}={} \mathrm{{dB}}$".format(snr, snr)
                    axs.plot(R, psnr_snr, label = lb, color = 'g', linestyle = '-', marker = self.mark[i], markersize = 8,)
                else:pass

        # label
        font = {'family':'Times New Roman','style':'normal','size':14, }
        axs.set_xlabel("Compression Rate", fontproperties=font)
        if PSNR == True:
            axs.set_ylabel( "PSNR (dB)", fontproperties = font )# , fontdict = font1
        else:
            axs.set_ylabel( "Test Accuracy (%)", fontproperties = font )# , fontdict = font1

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':8, }
        legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(1) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(1)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(1)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(1)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top=True, right = True, labelsize = 16, width = 2 )
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(12) for label in labels] #刻度值字号

        axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
        # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
        x_major_locator=MultipleLocator(0.1)
        axs.xaxis.set_major_locator(x_major_locator)
        # y_major_locator=MultipleLocator(0.1)
        # axs.yaxis.set_major_locator(y_major_locator)

        # fontt = {'family':'Times New Roman','style':'normal','size':16}
        # if self.title != '':
        #     plt.suptitle(self.title, fontproperties = fontt, )
        out_fig = plt.gcf()
        if PSNR == True:
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsR)_SNRTrain=({'_'.join([str(R) for R in tra_test_snr])})_R=({'-'.join([str(R) for R in Rlist])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsR)_SNRTrain=({'_'.join([str(R) for R in tra_test_snr])})_R=({'-'.join([str(R) for R in Rlist])})_Plot.png") )
        else:
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsR)_SNRTrain=({'_'.join([str(R) for R in tra_test_snr])})_R=({'-'.join([str(R) for R in Rlist])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsR)_SNRTrain=({'_'.join([str(R) for R in tra_test_snr])})_R=({'-'.join([str(R) for R in Rlist])})_Plot.png") )
        # plt.show()
        plt.close()
        return

    # 画出 JPEG 和 JSCC (包括在无噪训练和指定信噪比训练) 在指定压缩率 R 和 训练信噪比 trainsnr 下训练完后, PSNR 随 测试信噪比 testsnrlist 的曲线. 每条曲线对应一个 trainsnr
    def FixR_PSNRorAcc_vs_SNRtest(self,  R = 0.2, trainsnr = [3, 10], testsnrlist = jpeg_SNR,  PSNR = True, col_idx = 3, savename = ""):
        width = 6
        high  = 5
        fig, axs = plt.subplots(1, 1, figsize=(width, high),  constrained_layout = True) # constrained_layout=True
        ## JEPG
        if PSNR == True:
            if R in jpeg_comp:
                raw = jpeg_comp.index(R)
                psnr_jpg = []
                snrlist = []
                for i, snr in enumerate(testsnrlist):
                    if snr in jpeg_SNR:
                        snrlist.append(snr)
                        psnr_jpg.append(jpeg_res[raw, jpeg_SNR.index(snr)])
                if psnr_jpg != [] and snrlist != [] and len(psnr_jpg) == len(snrlist):
                    axs.plot(snrlist, psnr_jpg, label = "JPEG", color = 'b',linestyle = '-', marker = self.mark[0], markersize = 6, )
                else: pass

        ## R, noiseless
        if R in self.CompRate:
            tmps = f"TestMetrics:Compr={R:.1f},SNRtrain=noiseless(dB)"
            psnr_less = []
            snrlist = []
            for i, snr in enumerate(testsnrlist):
                if snr in self.SNRtest:
                    snrlist.append(snr)
                    row = np.where(self.SNRtest == snr)[0][0]
                    psnr_less.append(self.res_R_noiseless[tmps][row, col_idx])
            if psnr_less != [] and snrlist != [] and len(psnr_less) == len(snrlist):
                lb = r"$\mathrm{{SNR}}_\mathrm{{train}}=\mathrm{{noiseless}}$"
                # lb = "noiseless"
                axs.plot(snrlist, psnr_less, label = lb, color = 'r',linestyle = '-', marker = self.mark[1], markersize = 6, )
            else: pass

        ## R, SNR
        if R in self.CompRate:
            for i, trasnr in enumerate(trainsnr):
                print(f"{trasnr}")
                if  (trasnr in self.SNRtest) and (trasnr in self.SNRtrain):
                    tmps = f"TestMetrics:Compr={R:.1f},SNRtrain={trasnr}(dB)"
                    psnr_snr = []
                    snrlist = []
                    for tesnr in testsnrlist:
                        if tesnr in self.SNRtest:
                            snrlist.append(tesnr)
                            row = np.where(self.SNRtest == tesnr)[0][0]
                            psnr_snr.append(self.res_R_SNR[tmps][row, col_idx])
                    if psnr_snr != [] and snrlist != [] and len(psnr_snr) == len(snrlist):
                        lb = r"$\mathrm{{SNR}}_\mathrm{{train}}={} \mathrm{{dB}}$".format(trasnr)
                        axs.plot(snrlist, psnr_snr, label = lb, color = 'g',linestyle = '-', marker = self.mark[i], markersize = 6, )

        font = {'family':'Times New Roman','style':'normal','size':12, }
        axs.set_xlabel(r"$\mathrm{SNR}_\mathrm{test}$", fontproperties=font)
        if PSNR == True:
            axs.set_ylabel( "PSNR (dB)", fontproperties = font )# , fontdict = font1
        else:
            axs.set_ylabel( "Test Accuracy (%)", fontproperties = font )# , fontdict = font1
        #plt.title(label, fontproperties=font)

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':12, }
        legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(1) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(1)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(1)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(1)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 16, width = 3)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(12) for label in labels] #刻度值字号

        # axs.set_ylim(0.0, 1.001)  #拉开坐标轴范围显示投影
        x_major_locator=MultipleLocator(5)
        axs.xaxis.set_major_locator(x_major_locator)


        title = f"Compression Rate = {R}"
        if  title != '':
            fontt = {'family':'Times New Roman','style':'normal','size':16}
            plt.suptitle(title, fontproperties = fontt, )
        out_fig = plt.gcf()
        if PSNR == True:
            # out_fig.savefig( os.path.join(self.savedir, f"JPEG_JSCC(PSNRvsSNR_test)_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsSNRtest)_R={R:.1f}_SNRTrain=({'_'.join([str(R) for R in trainsnr])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsSNRtest)_R={R:.1f}_SNRTrain=({'_'.join([str(R) for R in trainsnr])})_Plot.png") )
        else:
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsSNRtest)_R={R:.1f}_SNRTrain=({'_'.join([str(R) for R in trainsnr])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsSNRtest)_R={R:.1f}_SNRTrain=({'_'.join([str(R) for R in trainsnr])})_Plot.png") )
        # plt.show()
        plt.close()
        return


    # 画出 JPEG 和 JSCC (包括在无噪训练和指定信噪比 trainsnr 下训练) 在指定信噪比 trainsnr 下训练，不同压缩率 Rlist 下的 PSNR 随 测试信噪比 testsnrlist 的曲线. 每条曲线对应一个 Rlist.
    def FixSNRtrain_PSNRorAcc_vs_SNRtest(self, trainsnr = 3, Rlist = [0.2, 0.8], testsnrlist = jpeg_SNR,  PSNR = True,  col_idx = 3, savename = "PSNRorAcc_vs_SNRtest"):
        width = 6
        high  = 5
        fig, axs = plt.subplots(1, 1, figsize=(width, high),  constrained_layout = True)# constrained_layout=True

        ## JEPG
        if  PSNR == True:
            for idx, cr in enumerate(Rlist):
                if cr in jpeg_comp:
                    raw = jpeg_comp.index(cr)
                    psnr_jpg = []
                    snrlist = []
                    for i, snr in enumerate(testsnrlist):
                        if snr in jpeg_SNR:
                            snrlist.append(snr)
                            psnr_jpg.append(jpeg_res[raw, jpeg_SNR.index(snr)])
                    if snrlist != [] and psnr_jpg != [] and len(snrlist) == len(psnr_jpg):
                        axs.plot(snrlist, psnr_jpg, label = f"JPEG, R={cr:.1f}", color = 'b', linestyle = '-', marker = self.mark[idx], markersize = 6,)
                    else: pass

        ## R, noiseless
        for idx, cr in enumerate(Rlist):
            if cr in self.CompRate:
                tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain=noiseless(dB)"
                psnr_less = []
                snrlist = []
                for i, snr in enumerate(testsnrlist):
                    if snr in self.SNRtest:
                        snrlist.append(snr)
                        row = np.where(self.SNRtest == snr)[0][0]
                        psnr_less.append(self.res_R_noiseless[tmps][row, col_idx])
                if snrlist != [] and psnr_less != [] and len(snrlist) == len(psnr_less):
                    lb = r"$\mathrm{{SNR}}_\mathrm{{train}}=\mathrm{{noiseless}},\mathrm{{R}}={}$".format(cr)
                    # lb = "noiseless"
                    axs.plot(snrlist, psnr_less, label = lb, color = 'r', linestyle = '-', marker = self.mark[idx], markersize = 6,)
                else: pass

        ## R, SNR
        if (trainsnr in self.SNRtest) and (trainsnr in self.SNRtrain):
            for idx, cr in enumerate(Rlist):
                if cr in self.CompRate:
                    tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain={trainsnr}(dB)"
                    psnr_snr = []
                    snrlist = []
                    for tesnr in testsnrlist:
                        if tesnr in self.SNRtest:
                            snrlist.append(tesnr)
                            row = np.where(self.SNRtest == tesnr)[0][0]
                            psnr_snr.append(self.res_R_SNR[tmps][row, col_idx])
                    if psnr_snr != [] and snrlist != [] and len(psnr_snr) == len(snrlist):
                        lb = r"$\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{dB}}, \mathrm{{R}}={}$".format(trainsnr, cr)
                        axs.plot(snrlist, psnr_snr, label = lb, color = 'g', linestyle = '-', marker = self.mark[idx], markersize = 6,)
                    else: pass

        font = {'family':'Times New Roman','style':'normal','size':12, }
        axs.set_xlabel(r"$\mathrm{SNR}_\mathrm{test}$", fontproperties=font)
        if PSNR == True:
            axs.set_ylabel( "PSNR (dB)", fontproperties = font )# , fontdict = font1
        else:
            axs.set_ylabel( "Test Accuracy (%)", fontproperties = font )# , fontdict = font1
        #plt.title(label, fontproperties=font)

        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
        font1 = {'family':'Times New Roman','style':'normal','size':12, }
        legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none')  # 设置图例legend背景透明

        axs.spines['bottom'].set_linewidth(1.5) ###设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(1.5)   ###设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(1.5)    ###设置上部坐标轴的粗细

        axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 16, width = 3)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(12) for label in labels] #刻度值字号

        # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
        x_major_locator=MultipleLocator(5)
        axs.xaxis.set_major_locator(x_major_locator)


        # title = r"$\mathrm{{SNR}}_\mathrm{{train}}:{} \mathrm{{dB}}$".format(trainsnr)
        # if  title != '':
        #     fontt = {'family':'Times New Roman','style':'normal','size':16}
        #     plt.suptitle(title, fontproperties = fontt, )
        out_fig = plt.gcf()

        # out_fig.savefig( os.path.join(self.savedir, f"JPEG_JSCC(PSNRvsSNR_test)_Plot.eps") )
        if PSNR == True:
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsSNRtest)_SNRTrain={trainsnr}_R=({'_'.join([str(R) for R in Rlist])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(PSNRvsSNRtest)_SNRTrain={trainsnr}_R=({'_'.join([str(R) for R in Rlist])})_Plot.png") )
        else:
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsSNRtest)_SNRTrain={trainsnr}_R=({'_'.join([str(R) for R in Rlist])})_Plot.eps") )
            out_fig.savefig( os.path.join(self.savedir, f"(ACCvsSNRtest)_SNRTrain={trainsnr}_R=({'_'.join([str(R) for R in Rlist])})_Plot.png") )
        # plt.show()
        plt.close()
        return



# def main():
pl = ResultPlot( ) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'
snrtest = np.arange(-2, 21, 1)
# snrtest = np.append(snrtest, 24)
# snrtest = np.append(snrtest, 25)
# snrtest = np.append(snrtest, 28)
# snrtest = np.append(snrtest, 30)
# snrtest = np.append(snrtest, 32)
# snrtest = np.append(snrtest, 35)
# snrtest = np.append(snrtest, 40)
snrtest.sort()
##===================== 1 ======================================
# pl.FixedSNRtraintest_PSNRorAcc_vs_R( tra_test_snr = [ 3, 10, 20,  ], Rlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ],  col_idx = 3 )

# ##===================== 2 ======================================
R = 0.9
trainSNR = [ 1, 3, 10, 20, 30]
# pl.FixR_PSNRorAcc_vs_SNRtest(R = R, trainsnr = trainSNR, testsnrlist = snrtest, col_idx = 3,  )


# ##===================== 3 ======================================

trainSNR = 3
rlist = [0.3, 0.9]

pl.FixSNRtrain_PSNRorAcc_vs_SNRtest(trainsnr = trainSNR, Rlist = rlist, testsnrlist = snrtest, col_idx = 3, )

# ##===================== 4  分类准确率 ======================================
# pl.FixedSNRtraintest_PSNRorAcc_vs_R( tra_test_snr = [1, 3, 10, 20, ], Rlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  PSNR = False,  col_idx = 1)

# ##===================== 5  分类准确率 ======================================

# pl.FixR_PSNRorAcc_vs_SNRtest(R = 0.6, trainsnr = [ 3, 10 ], testsnrlist = snrtest,  PSNR = False,  col_idx = 1)

# ##===================== 6  分类准确率======================================

# pl.FixSNRtrain_PSNRorAcc_vs_SNRtest(trainsnr = 10, Rlist = [0.2, 0.8], testsnrlist = snrtest,  PSNR = False,  col_idx = 1)

    # return


# mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
# color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]

def Compare_JSACshaoshuo(col_idx = 1, metrics = "Acc"):
    snrtest = np.arange(-2, 21, 1)
    snrtest = np.append(snrtest, 25)
    snrtest = np.append(snrtest, 30)
    snrtest = np.append(snrtest, 35)
    snrtest = np.append(snrtest, 40)
    snrtest.sort()

    CompRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  ]

    JSAC = torch.load("/home/jack/SemanticNoise_AdversarialAttack/results/2023-06-08-13:38:21_AE_noqua_cnn_classify_joinloss_mnist_R_noiseless/test_results/TesRecorder_TeMetricLog.pt")
    myres = torch.load("/home/jack/SemanticNoise_AdversarialAttack/results/2023-06-08-13:30:19_AE_cnn_classify_mnist_R_noiseless/test_results/TesRecorder_TeMetricLog.pt")

    width = 6
    high  = 5
    fig, axs = plt.subplots(1, 1, figsize=(width, high),  constrained_layout = True)# constrained_layout=True

    ## R, noiseless
    Rlist = [0.1, 0.3, 0.5, 0.9]
    for idx, cr in enumerate(Rlist):
        if cr in  CompRate:
            tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain=noiseless(dB)"
            lb = r"$\mathrm{{Join Loss}}, \mathrm{{R}}={}$".format(cr)
            axs.plot(snrtest, JSAC[tmps][:, col_idx], label = lb, color = 'b', linestyle = '-', marker =  mark[idx], markersize = 6,)

    for idx, cr in enumerate(Rlist):
        if cr in  CompRate:
            tmps = f"TestMetrics:Compr={cr:.1f},SNRtrain=noiseless(dB)"
            lb = r"$\mathrm{{MSE Loss}}, \mathrm{{R}}={}$".format(cr)
            axs.plot(snrtest, myres[tmps][:, col_idx], label = lb, color = '#FF6347', linestyle = '-', marker =  mark[idx], markersize = 6,)



    font = {'family':'Times New Roman','style':'normal','size':12, }
    axs.set_xlabel(r"$\mathrm{SNR}_\mathrm{test}$", fontproperties=font)
    if metrics == "Acc":
        axs.set_ylabel( "Test Accuracy (%)", fontproperties = font )# , fontdict = font1
    else:
        axs.set_ylabel( "PSNR (dB)", fontproperties = font )# , fontdict = font1
    #plt.title(label, fontproperties=font)

    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
    # font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
    font1 = {'family':'Times New Roman','style':'normal','size':12, }
    legend1 = axs.legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = font1,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    axs.spines['bottom'].set_linewidth(1.5) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(1.5)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(1.5)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(1.5)    ###设置上部坐标轴的粗细

    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 16, width = 3)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(12) for label in labels] #刻度值字号

    # axs.set_xlim(0.05, 0.94)  #拉开坐标轴范围显示投影
    x_major_locator=MultipleLocator(5)
    axs.xaxis.set_major_locator(x_major_locator)

    title = r"$\mathrm{{SNR}}_\mathrm{{train}}:\mathrm{{noiseless}} $"
    if  title != '':
        fontt = {'family':'Times New Roman','style':'normal','size':16}
        plt.suptitle(title, fontproperties = fontt, )
    out_fig = plt.gcf()

    savedir = os.path.join(user_home, 'SemanticNoise_AdversarialAttack/Figures')
    if metrics == "Acc":
        out_fig.savefig( os.path.join( savedir, "compare_AccVsSNRtest_Plot.eps") )
    else:
        out_fig.savefig( os.path.join( savedir, "compare_PSNRVsSNRtest_Plot.eps") )
    # plt.show()
    plt.close()

    return


# if __name__ == '__main__':
#     main()
#     # Compare_JSACshaoshuo(col_idx = 1, metrics = "Acc")
#     # Compare_JSACshaoshuo(col_idx = 2, metrics = "PSNR")














