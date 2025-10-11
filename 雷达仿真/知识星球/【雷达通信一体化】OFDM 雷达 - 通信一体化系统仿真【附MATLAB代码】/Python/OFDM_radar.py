

# https://zhuanlan.zhihu.com/p/521009340

import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.constants import c
import commpy
from Modulations import modulator

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

# 生成传输数据
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
TxData = modem.modulate(bits).reshape(N, M)

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
        phase_shift = np.exp(-1j * 2 * np.pi * target_dop * target_delay) * np.exp(1j * 2 * np.pi * mSymbol * Ts * target_dop) * np.exp(-1j * 2 * np.pi * kSubcarrier * delta_f * target_delay)
        # 添加高斯噪声
        noise = np.sqrt(1/2) * (np.random.randn() + 1j * np.random.randn())
        RxData[kSubcarrier, mSymbol] = np.sqrt(SNR) * TxData[kSubcarrier, mSymbol] * phase_shift + noise

# 移除发射数据信息
dividerArray = RxData / TxData

################### MUSIC算法
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

#################### 周期图/FFT估计
NPer = 16 * N
normalizedPower = np.abs(np.fft.ifft(dividerArray, NPer, axis=0))
mean_normalizedPower = np.mean(normalizedPower, axis=1)
mean_normalizedPower = mean_normalizedPower / np.max(mean_normalizedPower)
mean_normalizedPower_dB = 10 * np.log10(mean_normalizedPower)

rangeIndex = np.arange(0, NPer) * c0 / (2 * delta_f * NPer)

################### Plot
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)
axs.plot(distanceIndex, SP_dB, label='MUSIC')
axs.plot(rangeIndex, mean_normalizedPower_dB, label='periodogram')
axs.set_xlabel('Range [m]')
axs.set_ylabel('Normalized Range Profile [dB]')
axs.legend()

plt.title('Range Estimation Comparison')
plt.show()
plt.close('all')


# 估计距离
rangeEstimation = np.argmax(mean_normalizedPower_dB)
distanceE = rangeEstimation * c0 / (2 * delta_f * NPer)
print(f"Estimated distance: {distanceE:.2f} m")
print(f"True distance: {target_pos} m")





