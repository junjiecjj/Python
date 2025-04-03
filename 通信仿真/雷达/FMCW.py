#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack
匹配滤波（脉冲压缩） matlab代码，亲测可用:
https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6

脉冲压缩原理以及实验代码详解:
https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10

基于Python的FMCW雷达工作原理仿真（附代码）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485426&idx=1&sn=ad1d302e2177b037778ee9e6d405ec33&chksm=c11f0a67f66883717c39bd6deab5a184182dec6192c517967d68b85b25cafe286607cfee1f7d&scene=21#wechat_redirect

雷达仿真 | 调频连续雷达 (FMCW)仿真Python代码
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247489978&idx=1&sn=8ed933fc275af974846a6f3be00f05d8&chksm=c06f60d3542de596d383a4c3322e1e0f33bd6853ad0962f49be14baaff7c1126ed51d0912525&mpshare=1&scene=1&srcid=0323bPMO68rvEFUU8aa5QnQY&sharer_shareinfo=cf22eefea6d212ac1867dcdaee6a8788&sharer_shareinfo_first=cf22eefea6d212ac1867dcdaee6a8788&exportkey=n_ChQIAhIQrzTnyknZZ2pOw7YOXO15HBKfAgIE97dBBAEAAAAAAKc4N%2B%2BIx4AAAAAOpnltbLcz9gKNyK89dVj0bIgzCpeuPa34D1Ov6V3ZVNbFSz830ZSINdOhiMO4Uw3qKUZFF%2FImjJO464ckbuOZkdSe4h1DJcnocX0ZxNrUBDOpDrKjOASUS8g8h3qrKw38eqEqDov7zgh7O9awFsoWefnY9rAKjSSjR2lhrmRH6icJX1x97e90jc%2FWoOgVyyTbCDDG8uDHbot7VmRc572NQq5ztzDZrGerQDeD%2BJ7%2BZrNugOG0ZauOW%2FkfU36c8T7oc3xiHMNI4imMMqMFS7UEPlluvQR%2FQaLpP1%2B9T8dm58YFWYOji4dCBTENOtiiLeOpPF4l71R1NrLA3OBDCfCKsI7%2BGmtxu%2FBD&acctmode=0&pass_ticket=l3Xl3zfrRyJIluhuYJTPnj02ELo%2F%2Fw4SEt9eaw9t0FoT7Ao94AINqNgjZ5nk%2FjXv&wx_header=0#rd

分享 | 调频连续波 (FMCW) 毫米波雷达目标检测Python仿真:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247491272&idx=1&sn=8c816033438a549fdaeb20e51b154896&chksm=c11f135df6689a4bb0528639e9f437c86e941ef816e78f2b9a935f568db8be529f85234cafd5&token=134337482&lang=zh_CN#rd

干货：FMCW雷达系统信号处理建模与仿真（含MATLAB代码）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247484183&idx=1&sn=fbf605f11d510343c5beda9bdc5c32a4&chksm=c11f0e82f66887941da61052bbfeee2fa37227e7a94d8ebb23fc1e9cc88a357f95ff4a10d012&scene=21#wechat_redirect

第4讲 -- 线性调频连续波LFMCW测量原理：测距、测速、测角:
https://zhuanlan.zhihu.com/p/687473210

利用Python实现FMCW雷达的距离多普勒估计(2D-FFT, 距离FFT，速度FFT)
https://blog.csdn.net/caigen0001/article/details/108815569
干货 | 利用MATLAB实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s?__biz=MzI2NzE1MTU3OQ==&mid=2649214285&idx=1&sn=241742b17b557c433ac7f5010758cd0f&chksm=f2905cf9c5e7d5efc16e84cab389ac24c5561a73d27fb57ca4d0bf72004f19af92b013fbd33b&scene=21#wechat_redirect
干货 | 利用Python实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s/X8uYol6cWoWAX6aUeR7S2A

雷达初学者必读 | 毫米波雷达信号处理入门教程:
https://blog.csdn.net/qq_35844208/article/details/127122316
毫米波雷达：信号处理:
https://zhuanlan.zhihu.com/p/524371087?utm_source=wechat_session&utm_medium=social&s_r=0
雷达原理 | 用MATLAB信号处理是如何解算目标的距离和速度信息的？
https://zhuanlan.zhihu.com/p/422798513

[解疑][TI]TI毫米波雷达系列（三）：调频连续波雷达回波信号3DFFT处理原理（测距、测速、测角）
https://blog.csdn.net/qq_35605018/article/details/108816709
回波3DFFT处理（测距、测速、测角）
https://blog.csdn.net/nuaahz/article/details/90719605
雷达信号处理之FMCW 3D-FFT原理（附带MATLAB仿真程序）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485771&idx=1&sn=8e269280b663226160227aec22806c3e&chksm=c11f04def6688dc8e20c2e92bed6bc4547bf107f87b77bfff29f2c66434fdb5702333184ee1d&scene=178&cur_album_id=2442863581802381317#rd

雷达入门课系列文章（1）| 基于MATLAB的雷达信号处理实验教程
https://zhuanlan.zhihu.com/p/567656893

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%


#%%


#%%
import numpy as np
import matplotlib.pyplot as plt

# ====== 参数设置 ======
c = 3e8             # 光速 (m/s)
fc = 77e9           # 载波频率 (Hz)
B = 300e6           # 带宽 (Hz)
Tc = 50e-6          # Chirp 时长 (s)
N_chirps = 64       # 每帧的 Chirp 数



#%%


#%%


#%%


#%%





































