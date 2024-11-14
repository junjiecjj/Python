#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:32:32 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247486365&idx=1&sn=b975e8533dbcc411633980c930b2e21c&chksm=c098401a7537e6fd74f0b7e296f525d18dee126e2a5903da7af9dbdb9bb5aeeba2e45cae03cf&mpshare=1&scene=1&srcid=1114UibbWz4PxG2P9GCCpvls&sharer_shareinfo=34ebc8f58e7e3004be2ddc9619e6aa08&sharer_shareinfo_first=34ebc8f58e7e3004be2ddc9619e6aa08&exportkey=n_ChQIAhIQlfBkBNMAVjvrZ4NYIbHRnRKfAgIE97dBBAEAAAAAAOKJFA4c2KcAAAAOpnltbLcz9gKNyK89dVj0nn1FrqtOkm0ieNtTI%2FArNugCAcHUmm%2F61Ufx52sPxemOZy%2Bg0PKNke1WfWiSi3I53iMJiXZ%2BJhJBvnVsELOaFmDlMOrSASEtQg4hlJgFXwltioH%2FK%2FJ8WwdX4VR3JqZkoN7nzVP%2B6wrU0SCrQwV0Y4%2FwZ57BLe2w7oPlnTAmyOTHjCDNertI6Zf%2Fx%2FXTXEve7GVShbpGxETvq5EAZzkU3VZwVOHmwpY7e9MR45LZNrtB5F%2FkhyvnOjrD9nbtYr3LA6XILA7iC975C8PuiadhpY5J9oqKqJZsnvPg7gTPAd55w%2F7Ibeh8qhyhiK0lfxAKHtzzeReYwVi5&acctmode=0&pass_ticket=MmcQZQgH2%2FU1voMMyM8x5lhqrcf2I45%2FPi9c9HsahE%2BdlPOzC%2F5VE%2FZ95BcYrFwa&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.sparse import rand as sparse_rand
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
np.random.seed(42)
n_samples, n_features = 1000, 2000
density_levels = [0.01, 0.05, 0.1]
noise_levels = np.linspace(0.1, 1.0, 20)
ridge_alpha = 0.01
sbl_alpha = 1e-6

def sparsity(coef, threshold=1e-3):
    return np.sum(np.abs(coef) > threshold) / len(coef)
results = []

for density in density_levels:
    X = sparse_rand(n_samples, n_features, density=density, format='csr').toarray()
    true_coef = np.zeros(n_features)
    non_zero_indices = np.random.choice(n_features, 30, replace=False)
    true_coef[non_zero_indices] = np.random.normal(loc=0, scale=1, size=30)

    noise_std = 0.05
    y = X @ true_coef + norm.rvs(scale=noise_std, size=n_samples)

    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X, y)
    sbl_solution = np.linalg.solve(X.T @ X + sbl_alpha * np.eye(n_features), X.T @ y)

    results.append({
        'density': density,
        'Ridge_sparsity': sparsity(ridge.coef_),
        'SBL_sparsity': sparsity(sbl_solution),
        'Ridge_L1': np.sum(np.abs(ridge.coef_)),
        'SBL_L1': np.sum(np.abs(sbl_solution)),
        'Ridge_L2': np.sqrt(np.sum(ridge.coef_**2)),
        'SBL_L2': np.sqrt(np.sum(sbl_solution**2)),
        'ridge_mse_errors': [], 'sbl_mse_errors': [],
        'ridge_mae_errors': [], 'sbl_mae_errors': [],
        'ridge_r2_scores': [], 'sbl_r2_scores': [],
        'ridge_exp_var_scores': [], 'sbl_exp_var_scores': [],
        'ridge_rmse_errors': [], 'sbl_rmse_errors': [],
        'ridge_max_errors': [], 'sbl_max_errors': []
    })

    for noise in noise_levels:
        y_noisy = X @ true_coef + norm.rvs(scale=noise, size=n_samples)
        ridge.fit(X, y_noisy)
        sbl_solution_noisy = np.linalg.solve(X.T @ X + sbl_alpha * np.eye(n_features), X.T @ y_noisy)

        results[-1]['ridge_mse_errors'].append(mean_squared_error(ridge.coef_, true_coef))
        results[-1]['sbl_mse_errors'].append(mean_squared_error(sbl_solution_noisy, true_coef))
        results[-1]['ridge_mae_errors'].append(mean_absolute_error(ridge.coef_, true_coef))
        results[-1]['sbl_mae_errors'].append(mean_absolute_error(sbl_solution_noisy, true_coef))
        results[-1]['ridge_r2_scores'].append(r2_score(y_noisy, X @ ridge.coef_))
        results[-1]['sbl_r2_scores'].append(r2_score(y_noisy, X @ sbl_solution_noisy))
        results[-1]['ridge_exp_var_scores'].append(explained_variance_score(y_noisy, X @ ridge.coef_))
        results[-1]['sbl_exp_var_scores'].append(explained_variance_score(y_noisy, X @ sbl_solution_noisy))
        results[-1]['ridge_rmse_errors'].append(np.sqrt(mean_squared_error(ridge.coef_, true_coef)))
        results[-1]['sbl_rmse_errors'].append(np.sqrt(mean_squared_error(sbl_solution_noisy, true_coef)))
        results[-1]['ridge_max_errors'].append(max_error(ridge.coef_, true_coef))
        results[-1]['sbl_max_errors'].append(max_error(sbl_solution_noisy, true_coef))

fig, axes = plt.subplots(3, 5, figsize=(18, 12), constrained_layout = True)
for i, res in enumerate(results):
    axes[i, 0].bar(['Ridge', 'SBL'], [res['Ridge_sparsity'], res['SBL_sparsity']], color=['#ff7f0e', '#9467bd'])
    axes[i, 0].set_title(f'稀疏度比较 (密度={res["density"]})')
    axes[i, 0].set_ylabel('稀疏度')
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)

    axes[i, 1].plot(noise_levels, res['ridge_mse_errors'], label='Ridge MSE', marker='x', color='#ff7f0e')
    axes[i, 1].plot(noise_levels, res['sbl_mse_errors'], label='SBL MSE', marker='s', color='#9467bd')
    axes[i, 1].set_title('均方误差 (MSE)')
    axes[i, 1].set_xlabel('噪声水平')
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)

    axes[i, 2].plot(noise_levels, res['ridge_rmse_errors'], label='Ridge RMSE', marker='x', color='#ff7f0e')
    axes[i, 2].plot(noise_levels, res['sbl_rmse_errors'], label='SBL RMSE', marker='s', color='#9467bd')
    axes[i, 2].set_title('均方根误差 (RMSE)')
    axes[i, 2].set_xlabel('噪声水平')
    axes[i, 2].grid(True, linestyle='--', alpha=0.6)

    axes[i, 3].plot(noise_levels, res['ridge_max_errors'], label='Ridge Max Error', marker='x', color='#ff7f0e')
    axes[i, 3].plot(noise_levels, res['sbl_max_errors'], label='SBL Max Error', marker='s', color='#9467bd')
    axes[i, 3].set_title('最大误差')
    axes[i, 3].set_xlabel('噪声水平')
    axes[i, 3].grid(True, linestyle='--', alpha=0.6)

    axes[i, 4].plot(noise_levels, res['ridge_exp_var_scores'], label='Ridge 解释方差', marker='x', color='#ff7f0e')
    axes[i, 4].plot(noise_levels, res['sbl_exp_var_scores'], label='SBL 解释方差', marker='s', color='#9467bd')
    axes[i, 4].set_title('解释方差')
    axes[i, 4].set_xlabel('噪声水平')
    axes[i, 4].grid(True, linestyle='--', alpha=0.6)

plt.show()













