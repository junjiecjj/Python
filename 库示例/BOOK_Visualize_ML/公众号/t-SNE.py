#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:17:27 2024

@author: jack
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 数据预处理
print("Loading data...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

# 标准化数据
print("Standardizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 openTSNE 进行 t-SNE 降维
# 设定 t-SNE 参数
tsne_params = {
    "n_components": 2,
    "perplexity": 30,
    "early_exaggeration": 12.0,
    "learning_rate": 200,
    "n_iter": 1000,
    "n_jobs": -1,
    "random_state": 42
}

print("Running t-SNE...")
start_time = time.time()
tsne = TSNE(**tsne_params)
X_tsne = tsne.fit(X_scaled)
end_time = time.time()
print(f"t-SNE done! Time elapsed: {end_time - start_time} seconds")

# 可视化
print("Creating visualization...")
plt.figure(figsize=(16, 10))
palette = sns.color_palette("bright", 10)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y.astype(int), palette=palette, legend="full", alpha=0.6)

plt.title("t-SNE visualization of MNIST dataset")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.legend(title="Digits", loc="best", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()


















