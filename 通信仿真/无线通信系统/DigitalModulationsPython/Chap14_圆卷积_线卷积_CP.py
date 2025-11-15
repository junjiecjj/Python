#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 17:23:38 2025

@author: jack
"""
import scipy
import numpy as np
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


#%% 实现与MATLAB cconv完全一致的圆卷积
import numpy as np

def cconv(a, b, n=None):
    """
    实现与MATLAB cconv完全一致的圆卷积
    参数:
        a, b: 输入复数数组
        n: 输出长度 (None表示默认长度len(a)+len(b)-1)
    返回:
        圆卷积结果 (复数数组)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # 默认输出长度
    if n is None:
        n = len(a) + len(b) - 1
    # 线性卷积
    linear_conv = np.convolve(a, b, mode='full')
    # 处理不同n的情况
    if n <= 0:
        return np.array([], dtype=complex)
    result = np.zeros(n, dtype=complex)
    if n <= len(linear_conv):
        # n <= M+N-1: 重叠相加
        for k in range(n):
            # 收集所有k + m*n位置的元素
            idx = np.arange(k, len(linear_conv), n)
            result[k] = np.sum(linear_conv[idx])
    else:
        # n > M+N-1: 补零
        result[:len(linear_conv)] = linear_conv

    return result

# 测试用例
a = np.array([1+1j, 2-2j, 3+0.7j])
b = np.array([2-1j, 1-2j, 4+1.7j, 1.1-9j, 6.8-4.4j, 6.7+4j])

# 对应MATLAB的输出
c = cconv(a, b)
c1 = cconv(a, b, 5)
c2 = cconv(a, b, 10)
c3 = cconv(a, b, 2)

print(f"\n\nc = {c}")
print(f"c1 = {c1}")
print(f"c2 = {c2}")
print(f"c3 = {c3}")


# 对应MATLAB的输出
C = cconv(b, a)
C1 = cconv(b, a, 5)
C2 = cconv(b, a, 10)
C3 = cconv(b, a, 2)

print(f"\n\nC = {C}")
print(f"C1 = {C1}")
print(f"C2 = {C2}")
print(f"C3 = {C3}")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 生成 循环矩阵
# def CirculantMatric(gen, row):
#      if type(gen) == list:
#           col = len(gen)
#      elif type(gen) == np.ndarray:
#           col = gen.size
#      row = col

#      mat = np.zeros((row, col), dtype = gen.dtype)
#      mat[:, 0] = gen
#      for i in range(1, row):
#           mat[:,i] = np.roll(gen, i)
#      return mat

# def circularConvolve(h, s, N):
#     if h.size < N:
#         h = np.hstack((h, np.zeros(N-h.size)))
#     col = N
#     row = s.size
#     H = np.zeros((row, col), dtype = s.dtype)
#     H[:, 0] = h
#     for i in range(1, row):
#           H[:,i] = np.roll(h, i)
#     res = H @ s
#     return res

##>>>
# generateVec =  [1 , 2  , 3 , 4  ]
# X = np.array(generateVec)
# L = len(generateVec)
# A = CirculantMatric(X, L)
# A1 = scipy.linalg.circulant(X)

##>>>
h = np.array([-0.4878, -1.5351, 0.2355])
s = np.array([-0.0155, 2.5770, 1.9238, -0.0629, -0.8105, 0.6727, -1.5924, -0.8007])
N = s.size

lin_s_h = scipy.signal.convolve(h, s)
cir_s_h = cconv(h, s, N)
print(f"lin_s_h = \n    {lin_s_h}\ncir_s_h = \n    {cir_s_h}")

## 加了循环前缀后线卷积的部分结果等于圆卷积。
Ncp = h.size - 1 # len(CP) >= len(h) - 1
s_cp = np.hstack((s[-Ncp:], s))
lin_scp = scipy.signal.convolve(h, s_cp)
r = lin_scp[Ncp:Ncp+N]
print(f" r = \n    {r}\n cir_s_h = \n    {cir_s_h}")  # cir_s_h == r

# 14.2.3 Verifying DFT property， 时域的圆卷积等于频域相乘.
R = scipy.fft.fft(r, N)
H = scipy.fft.fft(h, N)
S = scipy.fft.fft(s, N)

print(f"R = {R}")
print(f"H*S = {H*S}")
# r1 == r
r1 = scipy.fft.ifft(S*H)         # 频域相乘后ifft等于r
print(f"r1 = \n{r1}\ncir_s_h = \n{cir_s_h}")


#%%


#%% 验证《从微积分到5G》Chap13.Eq(13.1), Page 247

def genH(h, Nx,):
    Nh = h.size
    H = np.zeros((Nx+Nh-1, Nx),  dtype= complex )
    h = np.pad(h, (0, Nx - 1))
    for j in range(Nx):
        H[:,j] = np.roll(h, j)
    return H

def convMatrix(h, N):  #
    """
    Construct the convolution matrix of size (L+N-1)x N from the
    input matrix h of size L. (see chapter 1)
    Parameters:
        h : numpy vector of length L
        N : scalar value
    Returns:
        H : convolution matrix of size (L+N-1)xN
    """
    col = np.hstack((h, np.zeros(N-1)))
    row = np.hstack((h[0], np.zeros(N-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H

def CutFoldAdd(x, L):
    out = np.zeros(L, dtype = x.dtype)
    if x.size % L == 0:
        Int = x.size//L
        for i in range(Int):
            out += x[i*L:(i+1)*L]
    else:
        pad = L - x.size % L
        Int = x.size//L + 1
        for i in range(Int-1):
            out += x[i*L:(i+1)*L]
        out += np.pad(x[(Int-1)*L:Int*L], (0, pad))
    return out

# MOD_TYPE = "qam"
# Order = 16
# modem, Es, bps = modulator(MOD_TYPE, Order)
# Constellation = modem.constellation/np.sqrt(Es)
# AvgEnergy = np.mean(np.abs(Constellation)**2)

Nh = h.size
Nx = s.size
# h = np.sqrt(1/2) * (np.random.randn(Nh) + 1j * np.random.randn(Nh))
H = genH(h, Nx,)

# d = np.random.randint(Order, size = Nx)
# x = Constellation[d]
sigma2 = 0.1
z = np.sqrt(sigma2/2) * (np.random.randn(H.shape[0]) + 1j * np.random.randn(H.shape[0]))

## 线卷积, y == lin_s_h
y = H @ s # + z
print(f"y = {y}")

## 把y, h, x切成Nx的长度然后累加起来
h1 = np.pad(h, (0, Nx - 1))
h_tilde = CutFoldAdd(h1, Nx)
y_tilde = CutFoldAdd(y, Nx)
# z_tilde = CutFoldAdd(z, Nx)
H_tilde = scipy.linalg.circulant(h_tilde)

## 圆卷积。 y1 == y_tilde 得到验证, 仔细一想，这是肯定成立的， 因为仅仅只是行之间的线性相加，没有理由不成立。
y1 = H_tilde @ s # + z_tilde

## y1 == r, 这也说明，关于OFDM: (1) 在发送方加CP ; (2) 在接收方做切分然后累加转为圆卷积。两者是等效的。


#%% <A Dual-Functional Sensing-Communication Waveform Design Based on OFDM, Guanding Yu>

# 下面是OFDM中IFFT -> +cp -> H -> -cp -> FFT的等效过程
h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629, ])
s = np.fft.ifft(S) # IFFT
N = s.size
L = h.size
cir_s_h = cconv(h, s, N)  #   circular conv

H = convMatrix(h, N)
y = H @ s                 # linear conv

lenCP = L - 1
Acp = np.block([[np.zeros((lenCP, N-lenCP)), np.eye(lenCP)], [np.eye(N)]])
s_cp = Acp @ s                    # add CP

H_cp = convMatrix(h, s_cp.size)
y_cp = H_cp @ s_cp                #  pass freq selected channel

y_remo_cp = y_cp[lenCP:lenCP + N] # receiver, remove cp, == cir_s_h

H_cp1 = convMatrix(h, s_cp.size)[lenCP:lenCP + N, :]
y_remo_cp1 = H_cp1 @ s_cp        #  pass freq selected channel + remove cp

F = scipy.linalg.dft(N)/np.sqrt(N)
FH = F.conj().T

Diag = F @ H_cp1 @ Acp @ FH  # Eq.(3): F@T(h)@A@FH is diagonal such that the data is parallelly transmitted over different subcarriers, and thus the ISI is avoided.

CirH = H_cp1 @ Acp
print(f"h = {h}\nCirH = \n{CirH}") # H --> CirH, 将拓普利兹矩阵变为循环阵, 到这里，从离散信号角度完美的对应OFDM的理论


U, s, VH = scipy.linalg.svd(H_cp1)

V = VH.conj().T

Vs = V[:, -lenCP:]


H_cp1 @ Vs # Proof of Theorem 1


Vc = V[:, :S.size]
Vs = V[:, -lenCP:]

Vc @ Vc.conj().T + Vs @ Vs.conj().T  #  == I

x_oc = np.random.randn(H_cp1.shape[1])
s_os = np.random.randn(h.size - 1)
x_os = Vs @ s_os
x = x_oc + x_os  # Eq.(5)

s_s = s_os + Vs.conj().T @ x_oc
x_s = Vs @ s_s
x_c = Vc @ Vc.conj().T @ x_oc
x1 = x_c + x_s    # Eq.(9)

x_c @ x_s.conj().T # == 0,


# Eq.(11)
a = np.linalg.norm(x_oc)**2
b = np.linalg.norm(x_c)**2 + np.linalg.norm(Vs.conj().T @ x_oc)**2




#%%




#%%




#%%












































































