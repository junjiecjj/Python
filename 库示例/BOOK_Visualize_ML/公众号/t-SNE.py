#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:17:27 2024

https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247489459&idx=1&sn=411d0622eef2515a374ffae3f4b74929&chksm=9b14685aac63e14c0d9c1e2cf22b749d14e47067995271dfbd2d031bb4219f084cce3ef64120&mpshare=1&scene=1&srcid=07312qyq4bW16tG5nGvCdptd&sharer_shareinfo=0dd83cdadf6ce6fbda9b3b84d49ad039&sharer_shareinfo_first=0dd83cdadf6ce6fbda9b3b84d49ad039&exportkey=n_ChQIAhIQQedOmnY8zqbq2FeaZ7Vf8hKfAgIE97dBBAEAAAAAAOTJJUqB1tEAAAAOpnltbLcz9gKNyK89dVj0vksd5gNGg8S5qikZzC1z8VodCZMNXgq0qcoiFCuWa7e4ycUJzFN%2BW2FaLRM%2Fx07VDzfuaKBIF%2FwfDoWO9sBRlH6o3R%2FQdHPc1Gt5Nu1GAjmdA3baH1gMoFpwhoE8QL836HlyeT3K7jdyFv5Q9WDzk%2F6yBHyk5fQ0WFVjAZ54kRfQ3sx7zJwQdynfdeDFhAxuKh%2FSQrBJR7HjBPYu1AdfeCTfWwy1t1cb1OIXef2ucBE9OUx3uPw4S85yujsAGDLa3yrEhUrKwJwvHwawoED0oKehX7QB%2Facd48mDhhjMIC2Fwx9lvBrcTThyQWNf4r8GQzXN0pQskWHH&acctmode=0&pass_ticket=WFl%2BKlXhQItTEvpRu5jx%2BBRlDT%2BL2a3fJfh0VBO0eHP4cADRPu1QMB8nJX2mv%2FsH&wx_header=0#rd

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
# (70000, 784)
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
# (70000, 2)
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


















