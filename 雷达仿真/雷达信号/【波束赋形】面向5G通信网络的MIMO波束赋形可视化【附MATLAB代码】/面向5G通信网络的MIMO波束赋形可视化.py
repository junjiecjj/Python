#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 08:41:25 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247504602&idx=3&sn=0bbd408624d4a40cc2d5620dea395e6c&chksm=c26724ece8e9066d5f019a4f67a0b4bb37f39b85628ea384ae79d0cfda90acff83e66a77e8e5&mpshare=1&scene=1&srcid=0604jiXbcJNp3SD6xuooZb0A&sharer_shareinfo=97ed50bb3cf25d6d2e1bc30e78162774&sharer_shareinfo_first=97ed50bb3cf25d6d2e1bc30e78162774&exportkey=n_ChQIAhIQDM%2BFub8MMZH%2Fu2bzhJIt4xKfAgIE97dBBAEAAAAAABREMoIZ%2BQoAAAAOpnltbLcz9gKNyK89dVj0YoivGUKc5XxbwK2o%2BRT0qlX%2FEN4q8hCDf83%2BVtJjO2sNU9vQCAnsKehlgKlwK6qU9ZCFvlHH5vBqk56QPPKWYMGMKyBU9KHGls39tndr0FZUj3JRgKg7taEp5uTAdJU1y5xeFl0dZHPQVt%2FZrKYnpPiNKlxaPrBlzovMAnnxipIf7plrYpkKweh1yBQ87XiNQhk5sidQxQ%2BGwzRMUbBX5%2BpMzBFOjKxsF0MsLa%2FkIH6l3510d1DIrhX6ajnoFjffsuDv%2BjZs6O%2FQvfYfQn12Un4KGcWl5QbbQZPzD5xPrFLN6j0o1Mrd15PBD2ElrziuEUD2DguFRATl&acctmode=0&pass_ticket=SVI22YdMOfTX5uPwSMGb9MflpDte2DTqI3s5xPMRV3eUvgZ89twln3MnOpqIgVWS&wx_header=0&poc_token=HJIXa2ijD4JaWVP_YivLAs0E4xHMTiDyJjhc1avq


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings

# 设置随机种子
np.random.seed(42)

# 1. 设置环境
print("设置环境...")

# 2. 定义天线布局
numAntennas = 8
bsPosition = np.array([0, 0])
lambda_val = 1
d = lambda_val / 2
antennaPositions = np.arange(numAntennas) * d

# 3. 定义设备及其最小间隔
areaSize = 100
numDevices = 25
minSeparation = 10  # 设备间最小距离
devicePositions = np.zeros((numDevices, 2))

# 生成具有最小间隔的设备位置
for i in range(numDevices):
    attempts = 0
    while attempts < 100:
        pos = areaSize * (np.random.rand(2) - 0.5)
        if i == 0:
            devicePositions[i, :] = pos
            break

        # 检查与所有先前设备的距离
        distances = np.sqrt(np.sum((devicePositions[:i, :] - pos)**2, axis=1))
        if np.all(distances >= minSeparation):
            devicePositions[i, :] = pos
            break

        attempts += 1

    if attempts >= 100:
        raise ValueError(f'Could not place device {i+1} with minimum separation of {minSeparation}. Try reducing minSeparation or areaSize.')

# 4. 用户选择多通信
while True:
    try:
        numCommunications = int(input('Enter number of communications (1 to 10): '))
        if 1 <= numCommunications <= 10:
            break
        else:
            print('Invalid input. Try again.')
    except ValueError:
        print('Invalid input. Try again.')

commDevices = [None] * numCommunications
# 为最多10个通信定义颜色
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]

for comm in range(numCommunications):
    print(f'Communication {comm+1}:')

    max_devices = min(numAntennas - 1, numDevices)
    while True:
        try:
            numDevInComm = int(input(f'Enter number of devices for communication {comm+1} (2 to {max_devices}): '))
            if 2 <= numDevInComm <= max_devices:
                break
            else:
                print(f'Invalid input. Must be between 2 and {max_devices}.')
        except ValueError:
            print('Invalid input. Try again.')

    devices = np.zeros(numDevInComm, dtype=int)
    selectedDevices = []

    for prev_comm in range(comm):
        if commDevices[prev_comm] is not None:
            selectedDevices.extend(commDevices[prev_comm])

    for i in range(numDevInComm):
        while True:
            try:
                dev = int(input(f'Enter device number {i+1} for communication {comm+1} (1 to {numDevices}): '))
                if 1 <= dev <= numDevices and dev not in selectedDevices and dev not in devices[:i]:
                    devices[i] = dev
                    selectedDevices.append(dev)
                    break
                else:
                    print('Invalid or already selected device. Try again.')
            except ValueError:
                print('Invalid input. Try again.')

    commDevices[comm] = devices

# 5. 仿真设置
SNR = 30  # 增加的SNR
noiseVar = 10**(-SNR/10)
numIterations = 5
receivedPowerMIMO = [None] * numCommunications
receivedPowerSISO = [None] * numCommunications
sinrValues = [None] * numCommunications

for comm in range(numCommunications):
    numStreams = len(commDevices[comm])
    receivedPowerMIMO[comm] = np.zeros((numStreams, numIterations))
    receivedPowerSISO[comm] = np.zeros((numStreams, numIterations))
    sinrValues[comm] = np.zeros((numStreams, numIterations))

# 路径损耗模型参数
PL0 = 30  # 参考距离d0处的路径损耗(dB)
d0 = 1    # 参考距离(米)
pathLossExponent = 3  # 环境相关
shadowingStdDev = 1   # 进一步减少以提高稳定性

# 6. 可视化设置
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MIMO Beamforming Visualization')
movementScale = 5
numAnimationSteps = 20

# 极坐标子图设置
ax2 = plt.subplot(122, polar=True)

for iter in range(numIterations):
    # 设备移动（带最小间隔的随机游走）
    for i in range(numDevices):
        attempts = 0
        while attempts < 100:
            newPos = devicePositions[i, :] + movementScale * (np.random.rand(2) - 0.5)
            newPos = np.clip(newPos, -areaSize/2, areaSize/2)

            # 检查与其他所有设备的距离
            other_indices = list(range(numDevices))
            other_indices.pop(i)
            otherPositions = devicePositions[other_indices, :]
            distances = np.sqrt(np.sum((otherPositions - newPos)**2, axis=1))

            if np.all(distances >= minSeparation):
                devicePositions[i, :] = newPos
                break

            attempts += 1

        if attempts >= 100:
            warnings.warn(f'Could not move device {i+1} while maintaining minimum separation.')

    # 左子图：设备布局和数据流
    ax1.clear()
    ax1.set_xlim([-areaSize/2, areaSize/2])
    ax1.set_ylim([-areaSize/2, areaSize/2])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Iteration {iter+1}/{numIterations} - Device Layout')
    ax1.grid(True)
    ax1.plot(bsPosition[0], bsPosition[1], 'ks', markersize=10, linewidth=2)
    ax1.plot(devicePositions[:, 0], devicePositions[:, 1], 'bo', markersize=8)

    # 处理每个通信
    totalWeights = np.zeros((numAntennas, 0), dtype=complex)
    senderIndices = np.zeros(numCommunications, dtype=int)

    for comm in range(numCommunications):
        selectedDevices = commDevices[comm]
        numStreams = len(selectedDevices)
        angles = np.zeros(numStreams)
        pathGains = np.zeros(numStreams)

        # 计算角度和路径增益（用于信道缩放）
        for i in range(numStreams):
            devPos = devicePositions[selectedDevices[i] - 1, :] - bsPosition
            angles[i] = np.degrees(np.arctan2(devPos[1], devPos[0]))
            dist = np.linalg.norm(devPos)
            PL_dB = PL0 + 10 * pathLossExponent * np.log10(dist / d0) + np.random.randn() * shadowingStdDev
            pathGains[i] = 10**(-PL_dB / 10)

        # 带路径损耗缩放的MIMO信道 (numStreams x numAntennas)
        condThreshold = 50  # 降低阈值以获得更好的条件
        H = (np.random.randn(numStreams, numAntennas) + 1j * np.random.randn(numStreams, numAntennas)) / np.sqrt(2)

        for i in range(numStreams):
            H[i, :] = np.sqrt(pathGains[i]) * H[i, :]

        # 如果条件数太高，重新生成H
        attempts = 0
        maxAttempts = 20  # 增加尝试次数

        while np.linalg.cond(H) > condThreshold and attempts < maxAttempts:
            H = (np.random.randn(numStreams, numAntennas) + 1j * np.random.randn(numStreams, numAntennas)) / np.sqrt(2)
            for i in range(numStreams):
                H[i, :] = np.sqrt(pathGains[i]) * H[i, :]
            attempts += 1

        if attempts >= maxAttempts:
            warnings.warn(f'Could not generate well-conditioned channel matrix for Comm {comm+1} after {maxAttempts} attempts.')

        # MMSE波束成形（预编码矩阵）
        W = np.linalg.pinv(H.conj().T @ H + noiseVar * np.eye(numAntennas)) @ H.conj().T

        # 归一化W的列以确保公平的功率分配
        for i in range(numStreams):
            W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

        weights = W  # numAntennas x numStreams

        # 符号向量（例如，为简单起见全为1）
        s = np.ones(numStreams, dtype=complex)  # numStreams x 1

        # 发射信号: x = W * s
        x = weights @ s  # numAntennas x 1

        # 接收信号: y = H * x + noise
        noise = np.sqrt(noiseVar) * (np.random.randn(numStreams) + 1j * np.random.randn(numStreams))
        receivedSignal = H @ x + noise  # numStreams x 1

        # 功率和SINR
        for i in range(numStreams):
            signalPower = np.abs(receivedSignal[i])**2  # 每个流的标量功率
            interference = np.sum(np.abs(receivedSignal)**2) - signalPower  # 总干扰
            sinrValues[comm][i, iter] = 10 * np.log10(signalPower / (interference + noiseVar))
            receivedPowerMIMO[comm][i, iter] = signalPower

        # 带路径损耗的SISO基线
        for i in range(numStreams):
            hSISO = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            hSISO = np.sqrt(pathGains[i]) * hSISO
            sisoNoise = np.sqrt(noiseVar) * (np.random.randn() + 1j * np.random.randn())
            sisoSignal = hSISO * 1 + sisoNoise  # 单个符号
            receivedPowerSISO[comm][i, iter] = np.abs(sisoSignal)**2

        # 用通信特定颜色绘制选定的设备
        ax1.plot(devicePositions[selectedDevices-1, 0], devicePositions[selectedDevices-1, 1],
                colors[comm] + '*', markersize=12, linewidth=2)

        # 为此通信选择随机发送者
        senderIndices[comm] = np.random.randint(numStreams)

        # 累积权重用于波束模式可视化
        totalWeights = np.hstack([totalWeights, weights])

    # 所有通信的同步可视化
    # 阶段1：动画显示所有发送者到基站的路径
    flows = [None] * numCommunications

    for comm in range(numCommunications):
        selectedDevices = commDevices[comm]
        senderIdx = senderIndices[comm]
        senderPos = devicePositions[selectedDevices[senderIdx] - 1, :]
        ax1.plot([senderPos[0], bsPosition[0]], [senderPos[1], bsPosition[1]],
                colors[comm] + '--', linewidth=1.5)
        flows[comm] = ax1.plot(senderPos[0], senderPos[1], colors[comm] + '^', markersize=8)[0]

    for step in range(numAnimationSteps):
        t = step / numAnimationSteps
        for comm in range(numCommunications):
            selectedDevices = commDevices[comm]
            senderIdx = senderIndices[comm]
            senderPos = devicePositions[selectedDevices[senderIdx] - 1, :]
            x = senderPos[0] + t * (bsPosition[0] - senderPos[0])
            y = senderPos[1] + t * (bsPosition[1] - senderPos[1])
            flows[comm].set_data(x, y)

        plt.pause(0.01)

    # 阶段2：动画显示基站到所有接收者的路径（所有通信）
    flows = [None] * numCommunications

    for comm in range(numCommunications):
        selectedDevices = commDevices[comm]
        senderIdx = senderIndices[comm]
        receiver_flows = []

        for i in range(len(selectedDevices)):
            if i != senderIdx:
                receiverPos = devicePositions[selectedDevices[i] - 1, :]
                ax1.plot([bsPosition[0], receiverPos[0]], [bsPosition[1], receiverPos[1]],
                        colors[comm] + '--', linewidth=1.5)
                flow = ax1.plot(bsPosition[0], bsPosition[1], colors[comm] + '^', markersize=8)[0]
                receiver_flows.append(flow)

        flows[comm] = receiver_flows

    for step in range(numAnimationSteps):
        t = step / numAnimationSteps
        for comm in range(numCommunications):
            selectedDevices = commDevices[comm]
            senderIdx = senderIndices[comm]

            for i, flow in enumerate(flows[comm]):
                receiver_idx = i if i < senderIdx else i + 1
                receiverPos = devicePositions[selectedDevices[receiver_idx] - 1, :]
                x = bsPosition[0] + t * (receiverPos[0] - bsPosition[0])
                y = bsPosition[1] + t * (receiverPos[1] - bsPosition[1])
                flow.set_data(x, y)

        plt.pause(0.01)

    # 右子图：辐射模式
    ax2.clear()
    thetaSweep = np.linspace(-np.pi, np.pi, 360)

    if totalWeights.size > 0:
        totalWeight = np.sum(totalWeights, axis=1)
        beamPattern = np.zeros(len(thetaSweep))

        for t, theta in enumerate(thetaSweep):
            sv = np.exp(-1j * 2 * np.pi * d * np.arange(numAntennas) * np.sin(theta) / lambda_val)
            beamPattern[t] = np.abs(sv.conj() @ totalWeight)

        beamPattern = 20 * np.log10(beamPattern / np.max(beamPattern))
        beamPattern[beamPattern < -30] = -30
        ax2.plot(thetaSweep, beamPattern, 'r', linewidth=1.5)

    ax2.set_title('Beam Pattern of Antenna Array')
    ax2.set_rlim([-30, 0])

    plt.pause(0.2)

# 最终绘图和指标
plt.figure(figsize=(10, 6))
plt.title('MIMO + Beamforming vs SISO by Communication')
barWidth = 0.4
avgPowerMIMOComm = np.zeros(numCommunications)
avgPowerSISOComm = np.zeros(numCommunications)

for comm in range(numCommunications):
    avgPowerMIMO = np.mean(receivedPowerMIMO[comm], axis=1)
    avgPowerSISO = np.mean(receivedPowerSISO[comm], axis=1)
    avgPowerMIMOComm[comm] = np.mean(avgPowerMIMO)
    avgPowerSISOComm[comm] = np.mean(avgPowerSISO)

x = np.arange(1, numCommunications + 1)
plt.bar(x - barWidth/2, 10*np.log10(avgPowerMIMOComm), barWidth, color='r', label='MIMO + Beamforming')
plt.bar(x + barWidth/2, 10*np.log10(avgPowerSISOComm), barWidth, color='gray', label='SISO')

# 确定哪个系统性能更好并添加注释
for comm in range(numCommunications):
    mimoPower = 10*np.log10(avgPowerMIMOComm[comm])
    sisoPower = 10*np.log10(avgPowerSISOComm[comm])

    if mimoPower > sisoPower:
        betterSystem = 'MIMO better'
        yPos = mimoPower + 1
    else:
        betterSystem = 'SISO better'
        yPos = sisoPower + 1

    plt.text(comm + 1, yPos, betterSystem, ha='center', fontsize=8, color='k')

plt.xlabel('Communication')
plt.ylabel('Avg Received Power (dB)')
plt.legend()
plt.xticks(x, [f'Comm {i+1}' for i in range(numCommunications)])
plt.grid(True)

# 控制台输出
print('Communication Metrics:')
print(f'SNR: {SNR:.2f} dB')

for comm in range(numCommunications):
    selectedDevices = commDevices[comm]
    avgSINR = np.mean(sinrValues[comm], axis=1)
    print(f'Metrics for Communication {comm+1}:')

    for i in range(len(selectedDevices)):
        print(f'Device {selectedDevices[i]} - Final MIMO Power: {10*np.log10(receivedPowerMIMO[comm][i, -1]):.2f} dB, Avg SINR: {avgSINR[i]:.2f} dB')
        print(f'          Final SISO Power: {10*np.log10(receivedPowerSISO[comm][i, -1]):.2f} dB')

# 打印每个通信的性能比较
print('Performance Comparison:')
for comm in range(numCommunications):
    mimoPower = 10*np.log10(avgPowerMIMOComm[comm])
    sisoPower = 10*np.log10(avgPowerSISOComm[comm])

    if mimoPower > sisoPower:
        print(f'Communication {comm+1}: MIMO performs better with avg power {mimoPower:.2f} dB vs SISO {sisoPower:.2f} dB')
    else:
        print(f'Communication {comm+1}: SISO performs better with avg power {sisoPower:.2f} dB vs MIMO {mimoPower:.2f} dB')

plt.tight_layout()
plt.show()

























