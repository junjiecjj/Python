#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:20:18 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_data = 500
# 生成1D数组
data = np.random.normal(loc=-2, scale=3, size=num_data)  # 注意这里去掉了size中的逗号
loc_array = np.arange(num_data)  # 更简单的索引生成方式

mu_data = data.mean()
sigma_data = data.std()

for sigma_band_factor in [1, 2]:
    inside_data = data.copy()
    outside_data = data.copy()

    plus_sigma = mu_data + sigma_band_factor * sigma_data
    minus_sigma = mu_data - sigma_band_factor * sigma_data

    outside_data[(outside_data >= minus_sigma) & (outside_data <= plus_sigma)] = np.nan
    inside_data[(inside_data >= plus_sigma) | (inside_data <= minus_sigma)] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})

    ax1.plot(loc_array, inside_data, marker='.', color='b', linestyle='None')
    ax1.plot(loc_array, outside_data, marker='x', color='r', linestyle='None')

    ax1.fill_between(loc_array, plus_sigma, minus_sigma, color='#DBEEF4', alpha=0.3)

    ax1.axhline(y=mu_data, color='r', linestyle='--')
    ax1.axhline(y=plus_sigma, color='r', linestyle='--')
    ax1.axhline(y=minus_sigma, color='r', linestyle='--')
    ax1.set_ylim([np.floor(data.min()) - 1, np.ceil(data.max()) + 1])
    ax1.set_xlim([0, num_data])

    # 使用kdeplot+rugplot组合替代distplot
    sns.kdeplot(y=data, ax=ax2, color='blue', fill=True)
    sns.rugplot(y=data, ax=ax2, color='k', height=0.06, lw=0.5, alpha=0.5)

    # 添加直方图
    ax2_hist = ax2.twinx()
    sns.histplot(y=data, ax=ax2_hist, bins=15, alpha=0.3, color='green', stat='density')
    ax2_hist.set_ylim(ax2.get_ylim())
    ax2_hist.set_yticks([])

    ax2.set_ylim([np.floor(data.min()) - 1, np.ceil(data.max()) + 1])
    ax2.axhline(y=mu_data, color='r', linestyle='--')
    ax2.axhline(y=plus_sigma, color='r', linestyle='--')
    ax2.axhline(y=minus_sigma, color='r', linestyle='--')

    print(f"Outside {sigma_band_factor} sigma: {np.count_nonzero(~np.isnan(outside_data))} points")
    plt.tight_layout()
    plt.show()

plt.close('all')
