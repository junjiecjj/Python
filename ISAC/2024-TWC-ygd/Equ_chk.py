#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:51:52 2025

@author: jack
"""

import numpy as np



def generate_U_n(n, M):
    """
    正确生成矩阵 U_n，其第 n 条次对角线为 1，其余元素为 0

    参数:
    n: int - 对角线索引
        n > 0: 上对角线
        n = 0: 主对角线
        n < 0: 下对角线
    M: int - 矩阵维度 (M x M)

    返回:
    U_n: numpy.ndarray - (M, M) 维矩阵
    """
    # 创建 M x M 的零矩阵
    U_n = np.zeros((M, M),)

    # 填充第 n 条对角线
    # 对于 n >= 0，填充上对角线
    # 对于 n < 0，填充下对角线
    for i in range(M):
        j = i + n
        if 0 <= j < M:
            U_n[i, j] = 1

    return U_n

np.random.seed(42)


N = 6
n = 2
Un = generate_U_n(n, N, )
U_n = generate_U_n(-n, N, )

xc = np.random.randn(N) + 1j * np.random.randn(N)
xs = np.random.randn(N) + 1j * np.random.randn(N)


# a_{-n} = a_n^*
an = xc.conj().T @ Un @ xc + xs.conj().T @ Un @ xs
a_n = xc.conj().T @ U_n @ xc + xs.conj().T @ U_n @ xs

# b_{-n} = c_n^*
b_n = xc.conj().T @ U_n @ xs
cn = xs.conj().T @ Un @ xc

# c_{-n} = b_n^*
bn = xc.conj().T @ Un @ xs
c_n = xs.conj().T @ U_n @ xc


#%% Eq.(26), Lemma.2
import numpy as np
import matplotlib.pyplot as plt

"""
验证引理2：通过频域方法计算周期序列的互相关
"""
# 设置参数
N = 64  # 序列长度

# 生成两个周期序列
np.random.seed(42)
x1_P = np.random.randn(N) + 1j * np.random.randn(N)  # 复数序列
x2_P = np.random.randn(N) + 1j * np.random.randn(N)

print("序列长度 N =", N)
print("x1_P 形状:", x1_P.shape)
print("x2_P 形状:", x2_P.shape)

# 方法1：直接计算互相关（时域方法）
def direct_cross_correlation(x1, x2):
    N = len(x1)
    r_direct = np.zeros(N, dtype=complex)

    for k in range(N):
        # 周期序列的互相关

        for n in range(N):
            r_direct[k] += np.conj(x1[n]) * x2[(n + k) % N]

    return r_direct

# 方法2：使用引理的频域方法
def lemma_cross_correlation(x1, x2):
    # 计算DFT
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)

    # 取第一个DFT的共轭，然后与第二个DFT进行Hadamard积
    product = np.conj(X1) * X2

    # 计算IDFT得到互相关
    r_lemma = np.fft.ifft(product)

    return r_lemma

# 计算两种方法的互相关
r_direct = direct_cross_correlation(x1_P, x2_P)
r_lemma = lemma_cross_correlation(x1_P, x2_P)

# 计算误差
error = np.abs(r_direct - r_lemma)
max_error = np.max(error)
mean_error = np.mean(error)

print(f"\n最大误差: {max_error:.2e}")
print(f"平均误差: {mean_error:.2e}")

# 可视化结果
plt.figure(figsize=(12, 8))

# 绘制实部比较
plt.subplot(2, 2, 1)
plt.plot(np.real(r_direct), 'b-', label='直接方法', linewidth=2)
plt.plot(np.real(r_lemma), 'r--', label='引理方法', linewidth=1.5)
plt.title('互相关实部比较')
plt.xlabel('延迟 k')
plt.ylabel('实部')
plt.legend()
plt.grid(True)

# 绘制虚部比较
plt.subplot(2, 2, 2)
plt.plot(np.imag(r_direct), 'b-', label='直接方法', linewidth=2)
plt.plot(np.imag(r_lemma), 'r--', label='引理方法', linewidth=1.5)
plt.title('互相关虚部比较')
plt.xlabel('延迟 k')
plt.ylabel('虚部')
plt.legend()
plt.grid(True)

# 绘制幅度比较
plt.subplot(2, 2, 3)
plt.plot(np.abs(r_direct), 'b-', label='直接方法', linewidth=2)
plt.plot(np.abs(r_lemma), 'r--', label='引理方法', linewidth=1.5)
plt.title('互相关幅度比较')
plt.xlabel('延迟 k')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

# 绘制误差
plt.subplot(2, 2, 4)
plt.plot(error, 'g-', linewidth=2)
plt.title('两种方法的误差')
plt.xlabel('延迟 k')
plt.ylabel('绝对误差')
plt.grid(True)

plt.tight_layout()
plt.show()

# 验证引理的正确性
if max_error < 1e-10:
    print("\n✅ 验证成功：引理2正确！两种方法结果一致。")
else:
    print("\n⚠️  验证失败：两种方法结果存在显著差异。")

# 额外测试：验证互相关的性质
print("\n" + "="*50)
print("额外测试：验证互相关的对称性")

# 交换序列顺序
r_swapped = np.fft.ifft(np.conj(np.fft.fft(x2_P)) * np.fft.fft(x1_P))

# 检查 r_xy[k] = r_yx^*[-k] 的性质
r_expected = np.conj(np.roll(r_swapped[::-1], 1))

symmetry_error = np.max(np.abs(r_direct - r_expected))
print(f"对称性验证误差: {symmetry_error:.2e}")

if symmetry_error < 1e-10:
    print("✅ 互相关对称性验证成功！")
else:
    print("⚠️ 互相关对称性验证失败！")

    # return r_direct, r_lemma, error, x1_P, x2_P

# # 运行验证
# if __name__ == "__main__":
#     r_direct, r_lemma, error, x1_P, x2_P = verify_cross_correlation_lemma()





















