#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 22:04:31 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
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
np.random.seed(42)
def drawpp(m, wop):
    """绘制阵列响应图"""
    thetas = np.arange(-90, 91)
    tm = thetas * np.pi / 180
    am = np.exp(-1j * np.pi * np.arange(0, m).reshape(-1, 1) * np.sin(tm))
    A = np.abs(wop.conj().T @ am)  # 阵列响应
    A = A / np.max(A)

    # 极坐标图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(tm, A.flatten())
    ax.set_title('归一化阵列响应幅值极坐标图，八阵元，信噪比20db')
    plt.show()

    # 直角坐标图
    A_db = 10 * np.log10(A.flatten())  # 对数图
    plt.figure()
    plt.plot(thetas, A_db)
    plt.title('八阵元，信噪比20db')
    plt.xlabel('入射角/度')
    plt.ylabel('归一化 A=10*log10(A)')
    plt.grid(True)
    plt.axis([-90, 90, -35, 0])

    # 标记特定角度
    plt.axvline(x=-45, color='r', linestyle='-')
    plt.axvline(x=30, color='r', linestyle='-')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.axvline(x=60, color='r', linestyle='-')
    plt.show()

def get_B(m, theta):
    """用于产生阻塞矩阵，采用正交法构造阻塞矩阵"""
    u0 = 0.5 * np.sin(theta[0])  # 假设阵元间距为半个波长
    a0 = np.exp(-1j * 2 * np.pi * np.arange(0, m) * u0)
    u = u0 + np.arange(1, m)
    B = np.exp(-1j * 2 * np.pi * np.arange(0, m).reshape(-1, 1) * u)
    Bm = B.conj().T  # M-1*M 的矩阵
    return Bm

def get_Rxx(s_rec, N, p, m):
    """计算协方差矩阵"""
    Rxx = np.zeros((m, m), dtype=complex)
    for i in range(N):
        x = s_rec[:, i]  # 获取第i个快拍
        R = np.outer(x.conj(), x)  # x' * x
        Rxx = Rxx + R
    Rxx = Rxx / N
    return Rxx

def get_s_rec(s, m, p, theta):
    """用于产生经过阵元后的信号数据"""
    A = np.zeros((m, p), dtype=complex)
    j = 1j

    # 阵元间距为半个波长
    wi = np.pi * np.sin(theta)
    A = np.exp(-1j * wi.reshape(-1, 1) * np.arange(0, m))  # 阵列流型
    s_rec = A.T @ s

    # 添加噪声，SNR=10 db
    signal_power = np.mean(np.abs(s_rec)**2)
    noise_power = signal_power / (10**(10/10))  # SNR = 10 dB
    noise = np.sqrt(noise_power/2) * (np.random.randn(*s_rec.shape) + 1j * np.random.randn(*s_rec.shape))
    s_rec = s_rec + noise

    return s_rec

def to_get_s(w, N, p):
    """生成信号"""
    s = np.zeros((p, N), dtype=complex)
    for i in range(p):
        s[i, :] = np.exp(1j * w[i] * np.arange(1, N+1))  # 复指数信号，假设信道增益为1
    return s

def main():
    # 参数设置
    m = 8  # array阵元
    p = 4  # signal number信号数
    N = 3000  # recursive number 迭代次数 或快拍数

    theta = np.array([30, 0, -45, 60]) * np.pi / 180  # DOA 30为期望信号方向
    j = 1j
    w = np.array([0.01, 0.2, 0.3, 0.4]) * np.pi  # frequency for each signal.各个信号的数字频率

    # 生成信号
    s = to_get_s(w, N, p)
    s_rec = get_s_rec(s, m, p, theta)
    S = s_rec  # output date matrix .m*N 的阵元输出数据矩阵

    # 自适应调节权
    y = S  # input data of GSC
    ad = np.exp(-1j * np.pi * np.arange(0, m).reshape(-1, 1) * np.sin(theta[0]))  # steering vector in the direction of expected. 期望信号方向导向矢量
    c = 10  # condition 波束形成条件
    C = ad.conj().T
    Wc = C.conj().T @ np.linalg.inv(C @ C.conj().T) * c  # main path weight 主通道固定权

    wa = np.zeros((m-1, 1), dtype=complex)  # auxiliary path 辅助通道自适应权
    B = get_B(m, theta)  # get Block Matrix 得到阻塞矩阵
    u = 0.000001

    Zc = np.zeros(N, dtype=complex)
    Za = np.zeros(N, dtype=complex)
    Z = np.zeros(N, dtype=complex)

    for k in range(N):
        yb = B @ y[:, k]  # m-1*1 的列向量
        Zc[k] = Wc.conj().T @ y[:, k]
        Za[k] = wa[:, 0].conj().T @ yb
        Z[k] = Zc[k] - Za[k]
        wa = wa - u * Z[k] * yb.conj().reshape(-1, 1)

    # 主通道
    wop = Wc
    print("主通道权向量:")
    print(wop)
    drawpp(m, wop)

    # 辅助通道
    wop = B.conj().T @ wa
    print("辅助通道权向量:")
    print(wop)
    drawpp(m, wop)

    # 总的阵列响应
    wop = Wc - B.conj().T @ wa
    print("总权向量:")
    print(wop)
    drawpp(m, wop)

if __name__ == "__main__":
    main()
