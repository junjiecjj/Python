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
algorithms = ['L1 minimization', 'OMP']

for idx, Ms_part in enumerate([Ms[:2], Ms[2:]]):
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plot_idx = 1

    for M in Ms_part:
        for bounds in bound_conditions:
            for matrix_type in measurement_matrices:
                for algo in algorithms:


                    if matrix_type == 'Gaussian':
                        phi = np.random.randn(M, N)
                    elif matrix_type == 'Bernoulli':
                        phi = np.random.choice([-1, 1], size=(M, N))
                    y = phi @ sparse_signal

                    if algo == 'L1 minimization':
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
                        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K)
                        omp.fit(phi, y)
                        reconstructed_signal = omp.coef_ if omp.coef_ is not None else None
                    plt.subplot(len(Ms_part), len(bound_conditions) * len(measurement_matrices) * len(algorithms), plot_idx)
                    plt.plot(sparse_signal, label='原始稀疏信号', color='blue', linewidth=1.5)
                    if reconstructed_signal is not None:
                        plt.plot(reconstructed_signal, label='重建信号', linestyle='--', color='red', linewidth=1.5)
                    title_text = f'M={M}, {matrix_type}, {algo}, Bound={bounds}'
                    plt.title(title_text, fontsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    if plot_idx == 1:
                        plt.legend(fontsize=8, loc='upper right')
                    plot_idx += 1
    plt.suptitle(f'压缩感知不同实验条件下的重建效果对比 - 第 {idx+1} 组', fontsize=20)
    plt.subplots_adjust(top=0.88, hspace=0.4)
    plt.show()
