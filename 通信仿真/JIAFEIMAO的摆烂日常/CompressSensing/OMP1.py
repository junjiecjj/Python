#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:47:22 2024

@author: jack
"""

# 实验(压缩感知在不同实验条件下的重建效果分析)



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.linear_model import OrthogonalMatchingPursuit
from matplotlib import rcParams, font_manager

def find_chinese_font():
    font_list = font_manager.fontManager.ttflist
    for font in font_list:
        if "SimHei" in font.name:
            return font.fname
        elif "SimSun" in font.name:
            return font.fname
    return None

font_path = find_chinese_font()
if font_path:
    my_font = font_manager.FontProperties(fname=font_path)
else:
    print("未找到中文字体")

rcParams['axes.unicode_minus'] = False
# rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'


def generate_sparse_signal(N, K):
    signal = np.zeros(N)
    non_zero_indices = np.random.choice(N, K, replace=False)
    signal[non_zero_indices] = np.random.randn(K)
    return signal

N = 128
K = 10
sparse_signal = generate_sparse_signal(N, K)

Ms = [50, 80, 100]
bound_conditions = [None, (0, None)]
measurement_matrices = ['Gaussian', 'Bernoulli']
algorithms = ['L1', 'OMP']

plt.figure(figsize = (32, 4*len(Ms)), constrained_layout = True)
plot_idx = 1
for idx, M in enumerate(Ms):
    # for M in Ms_part:
    for bounds in bound_conditions:
        for matrix_type in measurement_matrices:
            for algo in algorithms:
                if matrix_type == 'Gaussian':
                    phi = np.random.randn(M, N)
                elif matrix_type == 'Bernoulli':
                    phi = np.random.choice([-1, 1], size=(M, N))
                y = phi @ sparse_signal

                if algo == 'L1':
                    c = np.ones(N)
                    A_eq = phi
                    b_eq = y
                    bounds_l1 = [(0, None) if bounds else (None, None) for _ in range(N)]
                    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds_l1, method='highs')
                    if result.success:
                        reconstructed_signal = result.x
                    else:
                        reconstructed_signal = None
                elif algo == 'OMP':
                    omp = OrthogonalMatchingPursuit(n_nonzero_coefs = K)
                    omp.fit(phi, y)
                    reconstructed_signal = omp.coef_ if omp.coef_ is not None else None
                plt.subplot(len(Ms), len(bound_conditions) * len(measurement_matrices) * len(algorithms), plot_idx)
                plt.plot(sparse_signal, label='原始稀疏信号', color='blue', linewidth=1.5)
                if reconstructed_signal is not None:
                    plt.plot(reconstructed_signal, label='重建信号', linestyle='--', color='red', linewidth=1.5)
                title_text = f'M={M}, {matrix_type}, {algo}, Bound={bounds}'
                plt.title(title_text, fontsize=14, )
                plt.xticks([])
                plt.yticks([])
                if plot_idx == 1:
                    plt.legend(fontsize=12, prop=my_font, loc='upper right', )
                plot_idx += 1
plt.suptitle( '压缩感知不同实验条件下的重建效果对比', fontproperties=my_font, fontsize=22 )
plt.show()























