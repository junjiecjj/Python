#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:10:18 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics import mean_squared_error
import seaborn as sns
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

def generate_synthetic_data(n_samples = 100, n_features = 20, n_components = 10, noise_level = 0.1):
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        indices = np.random.choice(n_features, n_components, replace=False)
        X[i, indices] = np.random.randn(n_components)
    noise = noise_level * np.random.randn(n_samples, n_features)
    X_noisy = X + noise
    return X_noisy, X

def dictionary_learning_algorithms(X, n_components = 15):
    mini_batch_dict = MiniBatchDictionaryLearning(n_components = n_components, alpha = 1, max_iter = 500, random_state = 42)
    D_minibatch = mini_batch_dict.fit(X).components_
    X_minibatch = mini_batch_dict.transform(X) @ D_minibatch
    error_minibatch = mean_squared_error(X, X_minibatch)

    ksvd_dict = MiniBatchDictionaryLearning(n_components = n_components, alpha = 1, max_iter = 1000, random_state = 42)
    D_ksvd = ksvd_dict.fit(X).components_
    X_ksvd = ksvd_dict.transform(X) @ D_ksvd
    error_ksvd = mean_squared_error(X, X_ksvd)

    mod_dict = MiniBatchDictionaryLearning(n_components = n_components, alpha = 0.5, max_iter = 700, random_state = 42)
    D_mod = mod_dict.fit(X).components_
    X_mod = mod_dict.transform(X) @ D_mod
    error_mod = mean_squared_error(X, X_mod)

    return {
        "Mini-batch Dictionary Learning": (X_minibatch, error_minibatch, D_minibatch, 500, 1),
        "K-SVD": (X_ksvd, error_ksvd, D_ksvd, 1000, 1),
        "MOD": (X_mod, error_mod, D_mod, 700, 0.5)
    }

n_samples = 100
n_features = 50
n_components = 15

X_noisy, X_true = generate_synthetic_data(n_samples, n_features, n_components)
results = dictionary_learning_algorithms(X_noisy, n_components)

colors = sns.color_palette("Set2", 3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout = True)
algorithm_names = ["Mini-batch Dictionary Learning", "K-SVD", "MOD"]

for i, (name, (X_rec, error, D, max_iter, alpha)) in enumerate(results.items()):
    axes[i].text(0.5, 1.1, f"参数: max_iter={max_iter}, alpha={alpha}", ha='center', va='bottom', transform=axes[i].transAxes, fontsize=10, color="darkblue")
    axes[i].plot(X_true[0], label="原始信号", color="black", linestyle="--", linewidth=1.5)
    axes[i].plot(X_rec[0], label=f"{name} 重构 (误差: {error:.2f})", color=colors[i], linewidth=1.5)
    axes[i].legend(loc="upper right", fontsize=10, frameon=False)
    axes[i].set_title(f"{name} 重构效果", fontsize=12, weight='bold')
    axes[i].set_xlabel("特征维度", fontsize=11)
    axes[i].set_ylabel("信号幅度", fontsize=11)
    axes[i].tick_params(axis='both', which='major', labelsize=10)

plt.suptitle("字典学习算法比较", fontsize=16, weight='bold')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()














