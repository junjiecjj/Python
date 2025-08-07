#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 22:37:37 2025

@author: jack
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymc3 as pm

np.random.seed(42)

# 参数设置：药物A和药物B的效果评分均值和标准差
n_samples = 100# 每组样本数
mean_a, mean_b = 70, 65# 药物A和药物B的均值
std_a, std_b = 10, 12# 药物A和药物B的标准差

# 生成模拟数据
data_a = np.random.normal(mean_a, std_a, n_samples)  # 药物A的效果评分
data_b = np.random.normal(mean_b, std_b, n_samples)  # 药物B的效果评分

# 创建DataFrame
df = pd.DataFrame({
    'DrugA': data_a,
    'DrugB': data_b
})

# 查看前几行数据
df.head()

#%% 数据可视化：绘制药物A和药物B的效果评分分布
plt.figure(figsize=(10, 6))

# 绘制药物A的核密度估计图
sns.kdeplot(data_a, label='Drug A', color='blue', shade=True)

# 绘制药物B的核密度估计图
sns.kdeplot(data_b, label='Drug B', color='red', shade=True)

# 图形美化
plt.title("Distribution of Drug A and Drug B Effectiveness", fontsize=16)
plt.xlabel("Effectiveness Score", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()

plt.show()

# 定义贝叶斯模型
with pm.Model() as model:
    # 药物A的先验分布：均值为70，标准差为10
    mu_a = pm.Normal('mu_a', mu=70, sigma=10)  # 药物A的均值
    sigma_a = pm.HalfNormal('sigma_a', sigma=10)  # 药物A的标准差

    # 药物B的先验分布：均值为65，标准差为12
    mu_b = pm.Normal('mu_b', mu=65, sigma=10)  # 药物B的均值
    sigma_b = pm.HalfNormal('sigma_b', sigma=12)  # 药物B的标准差

    # 观测数据：药物A和药物B的效果评分分别服从正态分布
    obs_a = pm.Normal('obs_a', mu=mu_a, sigma=sigma_a, observed=data_a)
    obs_b = pm.Normal('obs_b', mu=mu_b, sigma=sigma_b, observed=data_b)

    # 使用MCMC方法进行采样，获取后验分布
    trace = pm.sample(2000, chains=1, cores=1, return_inferencedata=False)

#%% 使用traceplot显示后验分布的采样结果
pm.traceplot(trace)
plt.show()

#%% 可视化药物A和药物B均值的后验分布
plt.figure(figsize=(12, 6))

# 药物A的均值后验分布
sns.histplot(trace['mu_a'], kde=True, color='blue', label='Drug A Mean', stat="density")

# 药物B的均值后验分布
sns.histplot(trace['mu_b'], kde=True, color='red', label='Drug B Mean', stat="density")

# 添加图表标题和标签
plt.title("Posterior Distributions of Drug A and Drug B Means", fontsize=16)
plt.xlabel("Mean Effectiveness", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()

plt.show()

#%% 计算贝叶斯因子：比较药物A和药物B的效果评分
bf_10 = np.mean(trace['mu_a'] > trace['mu_b'])
print(f"Bayes Factor (BF10): {bf_10:.3f}")


#%%


























