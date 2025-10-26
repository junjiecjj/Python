#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:46:34 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import imageio
import os

# 认知雷达基础
# 作者: Dr Anum Pirkani

# 版权所有 (c) 2025 Anum Pirkani
# 保留所有权利。
# 在此授予许可，无需书面协议，无需许可费或版税费，即可使用、复制、
# 修改和分发此代码（源文件）及其文档用于任何目的，前提是
# 版权声明完整地出现在此代码的所有副本中，并且此代码的来源
# 应在任何使用此代码报告研究的出版物中得到承认。

np.random.seed(18)

NumRangeBins  = 101
NumDopplerBins = 64  # 修正了变量名拼写
NumScans      = 64

# 初始化状态
class Car:
    def __init__(self):
        self.InitialRange = 100
        self.DopplerBin = 40
        self.SNRdB = 10
        self.Velocity = -0.80
        self.VelocityDrift = 0
        self.hr0 = 2.2  # 目标形状（高斯分布中的距离、多普勒标准差）
        self.hd0 = 2.0

class Pedestrian:
    def __init__(self):
        self.InitialRange = 30
        self.DopplerBin = 9
        self.SNRdB = 8
        self.GaitFreq = 0.22
        self.SidebandW = 5
        self.Velocity = -0.20
        self.hr0 = 2.0  # 目标形状
        self.hd0 = 1.7

Car = Car()
Pedestrian = Pedestrian()

# 背景噪声和干扰
NoiseMu = 0.4      # 均值（线性）
ClutterScale = 0.55  # 瑞利分布
InterfProb = 0.1    # 每次扫描的干扰概率
InterfPower = 8     # 干扰强度
InterfWidth = [4, 18]  # 干扰宽度范围

# CA-CFAR
class CFAR:
    def __init__(self):
        self.TrainR = 6
        self.TrainD = 6
        self.GuardR = 2
        self.GuardD = 2
        self.k = 4.2

CFAR = CFAR()

# 认知适应
class Alpha:
    def __init__(self):
        self.power = 0.08  # 注意力功率平滑
        self.focus = 0.06  # 带宽聚焦平滑

Alpha = Alpha()
MinPower = 0.12
NoisePerScan = 0.18

# 带宽聚焦参数
BWTotal = 1.0
FocusHalfWinR = 9
FocusHalfWinD = 9
Kcenters = 4

# 耦合强度
GainBWToSNR = 0.25    # SNR乘数: 1 + gain*(eff_local - 0.5)
GainBWToClutter = 0.25  # 杂波方差减少: (1 - gain*(effBW - 0.5))
GainBWToHrange = 0.40  # 距离窄化: hr0 / (1 + gain*(eff_local - 0.5))

PowerMapRD = np.ones((NumRangeBins, NumDopplerBins))  # 当注意力 >= min_power时
BWFocusRD = np.zeros((NumRangeBins, NumDopplerBins))  # 注意力聚焦
DwellRD = np.ones((NumRangeBins, NumDopplerBins))     # 驻留（积分时间偏置）

DetLog = np.zeros((NumScans, NumRangeBins, NumDopplerBins), dtype=bool)
SigLog = np.zeros((NumScans, NumRangeBins, NumDopplerBins))

TP = 0; FP = 0; TN = 0; FN = 0

# 当前状态
CarRange = Car.InitialRange
CarDoppler = Car.DopplerBin
PedRange = Pedestrian.InitialRange
PedDoppler = Pedestrian.DopplerBin

# 创建图形 - 修复了窗口设置问题
fig = plt.figure(figsize=(12, 9))
try:
    # 尝试设置窗口位置，如果失败则忽略
    fig.canvas.manager.window.geometry('80x60+1200+900')
except:
    pass  # 在某些后端中可能无法设置窗口位置

gifFilename = 'CognitiveRadar.gif'
gifDelay = 0.08
FrameCount = 0

# 辅助函数
def gaussPatch2D(Nr, Nd, r0, d0, hr, hd):
    """在(r0,d0)处居中的归一化2D高斯斑块"""
    rr, dd = np.mgrid[0:Nr, 0:Nd]
    W = np.exp(-((rr - r0)**2) / (2 * hr**2) - ((dd - d0)**2) / (2 * hd**2))
    W = W / (np.max(W) + np.finfo(float).eps)
    return W

def fusedFocus2D(Nr, Nd, centers, hr, hd):
    """来自多个(r,d)中心的聚焦图"""
    F = np.zeros((Nr, Nd))
    for k in range(len(centers)):
        F = np.maximum(F, gaussPatch2D(Nr, Nd, centers[k, 0], centers[k, 1], hr, hd))
    if np.max(F) > 0:
        F = F / np.max(F)
    return F

def localMean(A, r0, d0, win):
    """在方形(2*win+1)^2窗口中的局部均值"""
    Nr, Nd = A.shape
    r1 = max(0, r0 - win); r2 = min(Nr, r0 + win + 1)
    d1 = max(0, d0 - win); d2 = min(Nd, d0 + win + 1)
    patch = A[r1:r2, d1:d2]
    return np.mean(patch)

def CA_CFAR_AP(X, CFAR, sz):
    """CA-CFAR检测器"""
    Nr, Nd = sz
    det = np.zeros((Nr, Nd), dtype=bool)

    Tr = CFAR.TrainR
    Td = CFAR.TrainD
    Gr = CFAR.GuardR
    Gd = CFAR.GuardD
    k = CFAR.k

    # 积分图
    Sp = np.zeros((Nr+1, Nd+1))
    Sp[1:, 1:] = np.cumsum(np.cumsum(X, axis=0), axis=1)

    def rectsum(a, b, c, d):
        """计算矩形区域的和"""
        a = max(0, min(Nr-1, a))
        b = max(0, min(Nr-1, b))
        c = max(0, min(Nd-1, c))
        d = max(0, min(Nd-1, d))
        if a > b or c > d:
            return 0
        return Sp[b+1, d+1] - Sp[a, d+1] - Sp[b+1, c] + Sp[a, c]

    for r in range(Nr):
        for d in range(Nd):
            # 完整（训练+保护+CUT）窗口
            r1 = r - (Tr + Gr)
            r2 = r + (Tr + Gr)
            d1 = d - (Td + Gd)
            d2 = d + (Td + Gd)

            # 保护+CUT
            g1 = r - Gr
            g2 = r + Gr
            h1 = d - Gd
            h2 = d + Gd

            totSum = rectsum(r1, r2, d1, d2)
            rr1 = max(0, r1); rr2 = min(Nr-1, r2)
            dd1 = max(0, d1); dd2 = min(Nd-1, d2)
            totNum = (rr2 - rr1 + 1) * (dd2 - dd1 + 1)

            guardSum = rectsum(g1, g2, h1, h2)
            gg1 = max(0, g1); gg2 = min(Nr-1, g2)
            hh1 = max(0, h1); hh2 = min(Nd-1, h2)
            guardNum = (gg2 - gg1 + 1) * (hh2 - hh1 + 1)

            refNum = totNum - guardNum
            if refNum <= 0:
                det[r, d] = False
                continue

            refMean = (totSum - guardSum) / refNum
            thr = refMean * (1 + 0.25 * k)
            det[r, d] = X[r, d] > thr

    return det

def mat2gray(A):
    """将矩阵归一化到[0,1]范围"""
    A_min = np.min(A)
    A_max = np.max(A)
    if A_max == A_min:
        return np.ones_like(A)
    return (A - A_min) / (A_max - A_min)

# 主循环
frames = []  # 用于存储GIF帧

for scan in range(NumScans):
    EffectiveBW = BWTotal * (0.5 + 0.5 * BWFocusRD)

    Noise = np.maximum(0, NoiseMu + 0.15 * np.random.randn(NumRangeBins, NumDopplerBins))

    Clutter = ClutterScale * np.abs((np.random.randn(NumRangeBins, NumDopplerBins) +
                                    1j * np.random.randn(NumRangeBins, NumDopplerBins)) / np.sqrt(2))
    Clutter = Clutter * (1.0 - GainBWToClutter * (EffectiveBW - 0.5))
    RD = Noise + Clutter

    # 干扰
    if np.random.rand() < InterfProb:
        if np.random.rand() < 0.5:
            rC = np.random.randint(8, NumRangeBins-8)
            width = np.random.randint(InterfWidth[0], InterfWidth[1]+1)
            rIdx = slice(max(0, rC-width), min(NumRangeBins, rC+width+1))
            RD[rIdx, :] = RD[rIdx, :] + InterfPower * (0.5 + 0.5 * np.random.rand())
        else:
            dC = np.random.randint(8, NumDopplerBins-8)
            width = np.random.randint(InterfWidth[0], InterfWidth[1]+1)
            dIdx = slice(max(0, dC-width), min(NumDopplerBins, dC+width+1))
            RD[:, dIdx] = RD[:, dIdx] + InterfPower * (0.5 + 0.5 * np.random.rand())

    # 更新目标状态
    CarRange = CarRange + Car.Velocity
    CarDoppler = CarDoppler + Car.VelocityDrift
    PedRange = PedRange + Pedestrian.Velocity
    CarRange = max(4, min(NumRangeBins-4, CarRange))
    PedRange = max(4, min(NumRangeBins-4, PedRange))

    # 在每个目标中心周围的BW采样
    win = 2
    CarRngIdx, CarDopIdx = round(CarRange), round(CarDoppler)
    PedRngIdx, PedDopIdx = round(PedRange), round(PedDoppler)

    CarEffLocal = localMean(EffectiveBW, CarRngIdx, CarDopIdx, win)
    PedEffLocal = localMean(EffectiveBW, PedRngIdx, PedDopIdx, win)

    CarSNRBoost = 1 + GainBWToSNR * (CarEffLocal - 0.5)
    PedSNRBoost = 1 + GainBWToSNR * (PedEffLocal - 0.5)

    CarHREff = Car.hr0 / (1 + GainBWToHrange * (CarEffLocal - 0.5))
    PedHREff = Pedestrian.hr0 / (1 + GainBWToHrange * (PedEffLocal - 0.5))
    CarHREff = max(0.8, CarHREff)
    PedHREff = max(0.7, PedHREff)

    # 汽车
    SNRlinCar = 10**((Car.SNRdB * CarSNRBoost) / 10)
    car_patch = gaussPatch2D(NumRangeBins, NumDopplerBins, CarRngIdx, CarDopIdx, CarHREff, Car.hd0)
    RD = RD + SNRlinCar * 0.9 * car_patch * PowerMapRD * (0.6 + 0.4 * DwellRD)

    # 行人
    DopCenter = PedDoppler + round(3.5 * np.sin(2 * np.pi * Pedestrian.GaitFreq * scan))
    SideAmp = 0.65 + 0.35 * np.sin(2 * np.pi * Pedestrian.GaitFreq * scan + np.pi/5)  # 时变扩展
    SNRlinPed = 10**((Pedestrian.SNRdB * PedSNRBoost) / 10)
    ped_patch = gaussPatch2D(NumRangeBins, NumDopplerBins, PedRngIdx, DopCenter, PedHREff, Pedestrian.hd0)
    RD = RD + SNRlinPed * 0.7 * ped_patch * PowerMapRD * (0.6 + 0.4 * DwellRD)

    for sb in range(1, Pedestrian.SidebandW + 1):
        w = SideAmp * np.exp(-0.5 * (sb / (Pedestrian.SidebandW/1.8))**2)
        RD = RD + SNRlinPed * 0.18 * w * gaussPatch2D(NumRangeBins, NumDopplerBins, PedRngIdx, DopCenter + sb, max(0.9, PedHREff), Pedestrian.hd0) * PowerMapRD
        RD = RD + SNRlinPed * 0.18 * w * gaussPatch2D(NumRangeBins, NumDopplerBins, PedRngIdx, DopCenter - sb, max(0.9, PedHREff), Pedestrian.hd0) * PowerMapRD

    RD = RD * (0.9 + 0.2 * EffectiveBW)

    SigLog[scan, :, :] = RD

    # CA-CFAR
    Detections = CA_CFAR_AP(RD, CFAR, (NumRangeBins, NumDopplerBins))
    DetLog[scan, :, :] = Detections

    # 来自CFAR的显著性（当前+最近），平滑
    RecentWin = slice(max(0, scan-5), scan+1)
    DetRecent = np.sum(DetLog[RecentWin, :, :], axis=0) > 0
    Saliency = gaussian_filter(Detections.astype(float) + 0.5 * DetRecent.astype(float), sigma=2.0)

    # 从显著性中选择前K个聚焦中心
    idxSort = np.argsort(Saliency.ravel())[::-1]
    idxSort = idxSort[:min(Kcenters, len(idxSort))]
    rC, dC = np.unravel_index(idxSort, (NumRangeBins, NumDopplerBins))
    centers = np.column_stack((rC, dC))

    if np.max(Saliency) < 0.25 and scan < 5:
        centers = np.array([[CarRngIdx, CarDopIdx], [PedRngIdx, DopCenter]])

    # 更新BW聚焦
    LocalF = fusedFocus2D(NumRangeBins, NumDopplerBins, centers, FocusHalfWinR, FocusHalfWinD)
    BWFocusRD = (1 - Alpha.focus) * BWFocusRD + Alpha.focus * LocalF

    # 注意力功率图，仅来自检测的奖励
    Reward = mat2gray(Detections.astype(float))  # [0..1]
    PowerMapRD = (1 - Alpha.power) * PowerMapRD + Alpha.power * (1 + Reward)
    PowerMapRD = np.maximum(PowerMapRD, MinPower)
    PowerMapRD = PowerMapRD / np.max(PowerMapRD)
    PowerMapRD = MinPower + (1 - MinPower) * PowerMapRD

    dwellBase = mat2gray(PowerMapRD + 0.6 * Saliency)
    DwellRD = 0.25 + 0.75 * dwellBase
    if np.random.rand() < NoisePerScan:
        DwellRD = np.maximum(DwellRD, np.random.rand(NumRangeBins, NumDopplerBins))

    # 真实目标位置
    GT = np.zeros((NumRangeBins, NumDopplerBins), dtype=bool)
    GT[CarRngIdx, CarDopIdx] = True
    GT[PedRngIdx, DopCenter] = True
    GTd = convolve2d(GT, np.ones((3, 3)), mode='same') > 0

    TP += np.sum(Detections & GTd)
    FP += np.sum(Detections & ~GTd)
    TN += np.sum(~Detections & ~GTd)
    FN += np.sum(~Detections & GTd)

    # 绘图
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.imshow(RD, origin='lower', aspect='auto')
    plt.title(f'Scan {scan+1}: Range Doppler')
    plt.xlabel('Doppler Bin')
    plt.ylabel('Range Bin')
    plt.colorbar()
    plt.set_cmap('jet')

    plt.subplot(2, 2, 2)
    plt.imshow(Detections, origin='lower', aspect='auto')
    plt.title('CFAR Detections')
    plt.xlabel('Doppler Bin')
    plt.ylabel('Range Bin')
    plt.colorbar()
    plt.set_cmap('gray')

    plt.subplot(2, 2, 3)
    plt.imshow(EffectiveBW, origin='lower', aspect='auto')
    plt.title('Cognitive Focus')
    plt.xlabel('Doppler Bin')
    plt.ylabel('Range Bin')
    plt.colorbar()
    plt.set_cmap('jet')

    plt.subplot(2, 2, 4)
    plt.imshow(PowerMapRD, origin='lower', aspect='auto')
    plt.title('Attention Power Map')
    plt.xlabel('Doppler Bin')
    plt.ylabel('Range Bin')
    plt.colorbar()
    plt.set_cmap('jet')
    plt.clim(0.8, 1)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

    # 保存当前帧到内存
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

# 保存GIF
print("正在生成GIF...")
imageio.mimsave(gifFilename, frames, duration=gifDelay, loop=0)
print(f"GIF已保存为: {gifFilename}")

P_D = TP / (TP + FN + np.finfo(float).eps)
P_FA = FP / (FP + TN + np.finfo(float).eps)
print(f'P_D: {P_D:.3f}')
print(f'P_FA: {P_FA:.3f}')

plt.show()
