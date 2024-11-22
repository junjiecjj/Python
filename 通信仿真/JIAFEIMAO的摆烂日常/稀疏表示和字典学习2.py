#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:11:53 2024

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

def generate_synthetic_data(n_samples=100, n_features=20, n_components=10, noise_level=0.1):
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        indices = np.random.choice(n_features, n_components, replace=False)
        X[i, indices] = np.random.randn(n_components)
    noise = noise_level * np.random.randn(n_samples, n_features)
    X_noisy = X + noise
    return X_noisy, X

def dictionary_learning_algorithms(X, n_components=15):
    mini_batch_dict = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, max_iter=500, random_state=42)
    D_minibatch = mini_batch_dict.fit(X).components_
    X_minibatch = mini_batch_dict.transform(X) @ D_minibatch
    error_minibatch = mean_squared_error(X, X_minibatch)

    ksvd_dict = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, max_iter=1000, random_state=42)
    D_ksvd = ksvd_dict.fit(X).components_
    X_ksvd = ksvd_dict.transform(X) @ D_ksvd
    error_ksvd = mean_squared_error(X, X_ksvd)

    mod_dict = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.5, max_iter=700, random_state=42)
    D_mod = mod_dict.fit(X).components_
    X_mod = mod_dict.transform(X) @ D_mod
    error_mod = mean_squared_error(X, X_mod)

    return {
        "Mini-batch Dictionary Learning": (X_minibatch, error_minibatch, D_minibatch),
        "K-SVD": (X_ksvd, error_ksvd, D_ksvd),
        "MOD": (X_mod, error_mod, D_mod)
    }

def run_multiple_experiments(n_experiments=10, n_samples=100, n_features=50, n_components=15):
    errors = {
        "Mini-batch Dictionary Learning": [],
        "K-SVD": [],
        "MOD": []
    }

    for _ in range(n_experiments):
        X_noisy, X_true = generate_synthetic_data(n_samples, n_features, n_components)
        results = dictionary_learning_algorithms(X_noisy, n_components)

        for name, (X_rec, error, D) in results.items():
            errors[name].append(error)
    avg_errors = {name: np.mean(errors[name]) for name in errors}
    return errors, avg_errors

n_experiments = 10
n_samples = 100
n_features = 50
n_components = 15

errors, avg_errors = run_multiple_experiments(n_experiments, n_samples, n_features, n_components)

print("各算法平均重构误差:")
for name, error in avg_errors.items():
    print(f"{name}: 平均误差 = {error:.4f}")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout = True)
sns.boxplot(data=list(errors.values()), palette="coolwarm", ax=axes[0], linewidth=1.5)
axes[0].set_xticklabels(list(errors.keys()), rotation=15, fontsize=12)
axes[0].set_ylabel("重构误差", fontsize=13, weight='bold')
axes[0].set_title("字典学习算法重构误差分布", fontsize=14, weight='bold')
axes[0].tick_params(axis='y', labelsize=12)

axes[1].bar(avg_errors.keys(), avg_errors.values(), color=sns.color_palette("coolwarm", 3), edgecolor='black', linewidth=1)
axes[1].set_ylabel("平均重构误差", fontsize=13, weight='bold')
axes[1].set_title("字典学习算法平均重构误差", fontsize=14, weight='bold')
axes[1].tick_params(axis='x', labelsize=12, rotation=15)
axes[1].tick_params(axis='y', labelsize=12)

plt.suptitle("字典学习算法误差分析", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
for ax in axes:
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()







