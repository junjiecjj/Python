#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:03:24 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
K = 6  # 天线数量
snr = -2  # 信噪比
theta = np.array([10, 16, 20])  # 波达方向
Sample = np.array([10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 300, 400, 600, 900, 1200])  # 快拍数
Ntrial = 200

j = 1j  # 虚数单位

# ----------------------------------------------------------------------- #
Ndoa = len(theta)
Nsample = len(Sample)
pdf_MDL = np.zeros((Nsample, Ndoa + 2))
Num_ref = np.arange(0, Ndoa + 2)

for nNsample in range(Nsample):
    number_dEVD = np.zeros(Ndoa + 2)

    T = Sample[nNsample]

    for nTrial in range(Ntrial):
        # ==================================================================
        # 生成信号
        source_power = 10 ** (snr / 10)

        # 生成源信号
        source_wave = (np.random.randn(Ndoa, T) + j * np.random.randn(Ndoa, T)) * np.sqrt(0.5)
        st = np.sqrt(source_power) * source_wave

        # 生成噪声
        nt = (np.random.randn(K, T) + j * np.random.randn(K, T)) * np.sqrt(0.5)

        # 阵列流型矩阵
        A = np.exp(1j * np.pi * np.arange(K)[:, np.newaxis] * np.sin(theta ))

        # 接收信号
        xt = A @ st + nt

        # ======================= MDL方法 ==============================
        Rx = xt @ xt.T.conj() / T
        U, s, Vh = np.linalg.svd(Rx)

        a = np.zeros(K)
        for m in range(K):
            # 取第m+1到第K个特征值
            negv = s[m:]
            # 计算几何平均
            if len(negv) > 0:
                geometric_mean = np.exp(np.sum(np.log(negv)) / len(negv))
                arithmetic_mean = np.mean(negv)
                Tsph = arithmetic_mean / geometric_mean
                # MDL准则
                a[m] = T * (K - m) * np.log(Tsph) + 0.5 * m * (2 * K - m) * np.log(T)
            else:
                a[m] = np.inf

        dEVD = np.argmin(a)

        # 统计检测结果
        if dEVD < len(number_dEVD):
            number_dEVD[dEVD] += 1

    pdf_MDL[nNsample, :] = number_dEVD / Ntrial
    print(f"快拍数 {T}: 检测到{Ndoa}个源的概率 = {pdf_MDL[nNsample, Ndoa]:.4f}")

# ============================================
# 绘图
plt.figure()
plt.semilogx(Sample, pdf_MDL[:, Ndoa], 'b:*', linewidth=2, markersize=8)
plt.ylabel('Probability of Detection')
plt.xlabel('Number of Snapshots')
plt.xlim(Sample[0], Sample[-1])
plt.ylim(0, 1)
plt.grid(True)
plt.xticks([10, 100, 1000], ['10^1', '10^2', '10^3'])
plt.tight_layout()
plt.show()










