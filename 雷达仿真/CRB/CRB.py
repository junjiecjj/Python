#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:41:19 2025

@author: jack

https://blog.csdn.net/weixin_43270276/article/details/124647286

https://blog.csdn.net/qq_42432868/article/details/129804574?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-129804574-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-129804574-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=8

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18

#%%  parameters
pi = np.pi
derad = pi/180             # 角度->弧度
M = 4;                     # 阵元个数
K = 1;                     # 信源数目
N = 1;                     # 快拍数
delay = 0.1*pi;              # 待估计的延时，可以由测向信号产生
snr = np.arange(10,32,2)     # 遍历信噪比范围
trail = 10000;               # 尝试次数
d = np.arange(M)             # 线阵间隔
esti_music = np.zeros(trail);
Len_SNR = snr.size
VAREsti_snr = np.zeros(Len_SNR)
VARCalc_snr = np.zeros(Len_SNR)

for index_snr in range(Len_SNR):
    snr_current = snr[index_snr]
    A = np.exp(-1j * d * delay)          #  方向矢量，复数形式

    for index_trail in range(trail):
        S = np.random.randn(K, N);              # 信源信号，入射信号，不相干即可，也可以用正弦替代
        X = A@S                                 # 构造接收信号
        X1 = awgn(X, snr_current, 'measured');  # 引入高斯白噪声，此时的SNR为对数形式
        Rxx = X1*X1'/N;                         # 标准MUSIC算法，计算协方差矩阵
        [EV,D] = eig(Rxx);                      # 特征值分解
        EVA = diag(D)';                         # 将特征值矩阵对角线提取并转为一行
        [EVA,I] = sort(EVA);                    # 将特征值排序 从小到大
        EV = fliplr(EV(:,I));                   # 对应特征矢量排序
        iang = -pi:0.01*pi:pi;                  # 延时遍历范围
        for index = 1:length(iang)
            angle_input = iang(index);
            phim = angle_input;
            a = exp(-j*d*phim).';
            En = EV(:,K+1:M);                   #  取矩阵的第M+1到N列组成噪声子空间
            Pmusic(index) = 1/(a'*En*En'*a);    # 行程MUSIC谱
        end
        Pmusic = abs(Pmusic);
        [y x] = max(Pmusic);                     # 单目标，求最大值即可
        esti_music(index_trail) = iang(x(1));   # 寻找最大值对应延时

    VAREsti_snr(index_snr) = (sum((esti_music-delay).^2)/(trail));       # 计算方差
    SNR_Linear = 10^(snr_current/10);                                    # 将对数SNR转换成线性格式
    VARCalc_snr(index_snr) = 6/(N*M*(M-1)*(M+1)*SNR_Linear);

end

figure(1)                                                        # 线性CRLB作图，近似为1/x函数
plot(snr,VAREsti_snr,'b')
hold on
plot(snr,VARCalc_snr,'r')
hold off

figure(2)                                                        # 对数CRLB作图，近似斜率为负的线性函数
plot(snr,10*log10(VAREsti_snr),'b')
hold on
plot(snr,10*log10(VARCalc_snr),'r')
hold off
































































