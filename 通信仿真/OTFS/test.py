

# https://zhuanlan.zhihu.com/p/521009340

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c
import scipy.special as sp

# 物理常数和参数设置
c0 = c  # 光速
fc = 30e9  # 载波频率
lambda_ = c0 / fc  # 波长
N = 64  # 子载波数量
M = 16  # 符号数量
delta_f = 15e3 * 2**6  # 子载波间隔
T = 1 / delta_f  # 符号持续时间
Tcp = T / 4  # 循环前缀持续时间
Ts = T + Tcp  # 总符号持续时间

qam = 4  # 4-QAM调制

# 生成传输数据
data = np.random.randint(0, qam, (N, M))

# QAM调制 - 手动实现4-QAM
def qammod_4qam(data, unit_avg_power=True):
    # 4-QAM映射: 0->(1+1j), 1->(1-1j), 2->(-1+1j), 3->(-1-1j)
    mapping = {
        0: 1 + 1j,
        1: 1 - 1j,
        2: -1 + 1j,
        3: -1 - 1j
    }

    modulated = np.vectorize(mapping.get)(data)

    if unit_avg_power:
        # 归一化到单位平均功率
        modulated = modulated / np.sqrt(2)

    return modulated

TxData = qammod_4qam(data, True)

# 目标参数
target_pos = 60  # 目标距离
target_delay = 2 * target_pos / c0  # 距离到时延转换
target_speed = 20  # 目标速度
# 多普勒频移计算
target_dop = 2 * target_speed / lambda_

SNR_dB = 5
SNR = 10**(SNR_dB/10)

# 接收信号模拟
RxData = np.zeros((N, M), dtype=complex)
for kSubcarrier in range(N):
    for mSymbol in range(M):
        # 信道效应：时延和多普勒
        phase_shift = np.exp(-1j * 2 * np.pi * fc * target_delay) * \
                     np.exp(-1j * 2 * np.pi * kSubcarrier * delta_f * target_delay)

        # 添加高斯噪声
        noise = np.sqrt(1/2) * (np.random.randn() + 1j * np.random.randn())

        RxData[kSubcarrier, mSymbol] = np.sqrt(SNR) * TxData[kSubcarrier, mSymbol] * phase_shift + noise

# 移除发射数据信息
dividerArray = RxData / TxData

# MUSIC算法
nTargets = 1
Rxxd = dividerArray @ dividerArray.conj().T / M

# 特征值分解
distanceEigen, Vd = np.linalg.eig(Rxxd)

# 排序特征值和特征向量
sorted_indices = np.argsort(distanceEigen)[::-1]
distanceEigenDiag = distanceEigen[sorted_indices]
Vd = Vd[:, sorted_indices]

# 噪声子空间
distanceEigenMatNoise = Vd[:, nTargets:]

# MUSIC谱估计
omegaDistance = np.arange(0, 2 * np.pi + np.pi/100, np.pi/100)
SP = np.zeros(len(omegaDistance), dtype=complex)
nIndex = np.arange(0, N)

for index, omega in enumerate(omegaDistance):
    omegaVector = np.exp(-1j * nIndex * omega)
    denominator = omegaVector.conj().T @ (distanceEigenMatNoise @ distanceEigenMatNoise.conj().T) @ omegaVector
    SP[index] = (omegaVector.conj().T @ omegaVector) / denominator

SP = np.abs(SP)
SPmax = np.max(SP)
SP_dB = 10 * np.log10(SP / SPmax)
distanceIndex = omegaDistance * c0 / (2 * np.pi * 2 * delta_f)

plt.figure(figsize=(12, 8))
plt.plot(distanceIndex, SP_dB, label='MUSIC')

# 周期图/FFT估计
NPer = 16 * N
normalizedPower = np.abs(np.fft.ifft(dividerArray, NPer, axis=0))
mean_normalizedPower = np.mean(normalizedPower, axis=1)
mean_normalizedPower = mean_normalizedPower / np.max(mean_normalizedPower)
mean_normalizedPower_dB = 10 * np.log10(mean_normalizedPower)

rangeIndex = np.arange(0, NPer) * c0 / (2 * delta_f * NPer)

plt.plot(rangeIndex, mean_normalizedPower_dB, label='periodogram')
plt.grid(True)
plt.xlabel('Range [m]')
plt.ylabel('Normalized Range Profile [dB]')
plt.legend()
plt.title('Range Estimation Comparison')
plt.tight_layout()
plt.show()

# 估计距离
rangeEstimation = np.argmax(mean_normalizedPower_dB)
distanceE = rangeEstimation * c0 / (2 * delta_f * NPer)
print(f"Estimated distance: {distanceE:.2f} m")
print(f"True distance: {target_pos} m")
