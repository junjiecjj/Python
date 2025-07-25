#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:39:41 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247493204&idx=1&sn=37b000a7f8c68c1aec1fa2a3b8e6ce75&chksm=c37e83cf08cd129b8b674f01c41431da93cd2a4dfb44ed1b1ffafb225b34fcd6fa435411807f&mpshare=1&scene=1&srcid=05106BHMf6Jhvz3QvicspgKl&sharer_shareinfo=d822fd0503317c25221bcf9b21ec6047&sharer_shareinfo_first=8f008a34713fa9db007da1d8046dc6bf&exportkey=n_ChQIAhIQblVV69anWwlcgLJ80M9UahKfAgIE97dBBAEAAAAAAEC%2BGmaThOcAAAAOpnltbLcz9gKNyK89dVj0gCogfTG1ikpljA1iqElz9mQb5nFBIP34%2BXW%2BhK0FZm0Bnkt%2FiGby5OF51HaF8hF7VHVp3Fg0JHKS904bBbSc9AeB1jYt0AL3CWxOGW7F9I7WT5OmtQnOX8WzElR2HlB7JI4wz7xZEAbnqicr6YfM1HwRm8QOvHw1h9eR5qVl2%2BcXgHyWIYLYUVUY%2Flng36f30DD5w2te6nXiqnTi1JZsJzU5BWR%2FFBr26UYMpU9EIyPDe65dXzk6lS%2Fiq385nP%2FrtGmT5fLtq7dOMrjpPCmgtZBpPLw32Vvjh2BrZXOjATi3j2yVti3zWHJL0P%2B%2FZep9ihyAuOAwTwwR&acctmode=0&pass_ticket=YyulR25cvACZaaXWRRmONTeEFuC1JLD6qB2xKRV1qGs6KV5eFIIo0gYm62eX54RA&wx_header=0#rd


"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def entropy(y):
    """计算标签序列 y 的熵值"""
    counts = np.bincount(y)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def information_gain(X, y, feature_idx, threshold):
    """
    计算在 feature_idx 上以 threshold 划分的信息增益
    X: 特征矩阵，shape=(n_samples, n_features)
    y: 标签向量，shape=(n_samples,)
    """
    parent_entropy = entropy(y)
    # 划分索引
    left_idx = X[:, feature_idx] <= threshold
    right_idx = ~left_idx
    # 加权熵
    n, n_left, n_right = len(y), left_idx.sum(), right_idx.sum()
    if n_left == 0 or n_right == 0:
        return 0.0
    e_left = entropy(y[left_idx])
    e_right = entropy(y[right_idx])
    child_entropy = (n_left/n) * e_left + (n_right/n) * e_right
    return parent_entropy - child_entropy

# 载入鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 将 species 替换为中文
mapping = {'setosa':'山鸢尾', 'versicolor':'杂色鸢尾', 'virginica':'维吉尼亚鸢尾'}
df['species_cn'] = df['species'].map(mapping)
df.head()

import matplotlib.pyplot as plt

# 准备数据
X = df[iris.feature_names].values
y = iris.target
feat_idx = iris.feature_names.index('petal length (cm)')
values = np.unique(X[:, feat_idx])
thresholds = (values[:-1] + values[1:]) / 2# 中点作为候选阈值

# 计算信息增益
ig_list = [information_gain(X, y, feat_idx, t) for t in thresholds]

# 绘制
plt.figure(figsize=(10,6))
plt.plot(thresholds, ig_list, marker='o', linewidth=2)
plt.xlabel('Threshold (cm)')
plt.ylabel('Information Gain')
plt.title('Information Gain vs Threshold\non Petal Length')
plt.grid(True, linestyle='--', alpha=0.5)
plt.fill_between(thresholds, ig_list, color='orange', alpha=0.3)
plt.scatter(thresholds[np.argmax(ig_list)], max(ig_list), s=150, color='red', label='Best Split')
plt.legend()
plt.show()


def fast_best_split(X_col, y):
    """
    向量化求单特征 X_col 最优划分点及对应信息增益
    返回: (best_threshold, best_ig)
    """
    # 排序
    sorted_idx = np.argsort(X_col)
    Xs, Ys = X_col[sorted_idx], y[sorted_idx]
    # 全局熵
    H_parent = entropy(Ys)
    n = len(Ys)

    # 初始化左/右计数
    unique_classes = np.unique(Ys)
    left_count = np.zeros(unique_classes.max()+1, dtype=int)
    right_count = np.bincount(Ys, minlength=unique_classes.max()+1)
    best_ig, best_t = 0.0, None

    # 遍历可能分割点（跳过相同值）
    for i in range(1, n):
        c = Ys[i-1]
        left_count[c] += 1
        right_count[c] -= 1
        if Xs[i] == Xs[i-1]:
            continue
        # 当前阈值
        t = (Xs[i] + Xs[i-1]) / 2
        # 计算左右熵（增量式）
        H_left = -np.sum((left_count/ i) * np.log2(left_count/ i + 1e-9) * (left_count>0))
        H_right = -np.sum((right_count/ (n-i)) * np.log2(right_count/ (n-i) + 1e-9) * (right_count>0))
        ig = H_parent - (i/n)*H_left - ((n-i)/n)*H_right
        if ig > best_ig:
            best_ig, best_t = ig, t
    return best_t, best_ig

best_threshold, best_ig = fast_best_split(X[:, feat_idx], y)
print(f"Optimized Best Threshold: {best_threshold:.3f}, Info Gain: {best_ig:.4f}")





































