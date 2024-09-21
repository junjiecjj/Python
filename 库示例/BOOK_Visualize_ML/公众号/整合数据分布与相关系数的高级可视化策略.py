#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:06:39 2024

@author: jack
https://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247486563&idx=1&sn=363ff2c27a09b74b197fb65f7e39e1f9&chksm=c2f27b09359e63785e37e2589bc5c5ebd90839c5c5a5e76c31012d5e1c00e5dac463d102b75b&mpshare=1&scene=1&srcid=0921Oy9dHmCZhdVnl0IgmyZP&sharer_shareinfo=38ed8a1e33f88c1ac37f621fc1986996&sharer_shareinfo_first=38ed8a1e33f88c1ac37f621fc1986996&exportkey=n_ChQIAhIQzSS0Chw5jBFANEULLB8S7hKfAgIE97dBBAEAAAAAAGwGD5bDDJEAAAAOpnltbLcz9gKNyK89dVj0FDFlEW7sTEXNuW44fatbBbFdcRC0bRD4J%2BIszZqT1yCGajmzEOC2uHOSOW1vBwQm1fcdgu%2FZ4OeMhVN5qK1dxn%2B1gyx%2Fuq0%2BJvkrRe6aTEWfBSNxCiWYplZkxW%2FfQwPy7jo2%2FHwn%2FDFtxf3q51vL%2B%2Fmz%2FFtnO70Ym%2F1c6h%2BN6hn4mqdO0ZQimUmYY%2BONSrWKj8eZqdgDMUGDmospaMY4ib1oAnC7CweP3YForN5B7rokD%2Bb2HKHkXCVj5SGHO0tZ0fsOi4m7a0z%2BeoUCAmkFobLO%2FEhCieCcryYYgRJkaKIH55jveF1LemWsxMnvyjhTsBoHnvIqeMsi&acctmode=0&pass_ticket=xzXXzgXCdsfEZDKKEIMJOIAh%2F72O3CswtCGebLxcBb9H1BqszsECADbhMqdzx1Q1&wx_header=0#rd


"""

# 数据生成
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 示例数据集
df = pd.DataFrame(np.random.randn(100, 10), columns=["M", "V", "D", "t", "w", "n", "fy", "fc", "L", "d"])
df.head()



# 基础复现

# 计算皮尔逊相关系数矩阵
corr = df.corr()

# 创建 PairGrid
g = sns.PairGrid(df)

# 左下角绘制散点图
g.map_lower(sns.scatterplot)

# 对角线绘制直方图
g.map_diag(sns.histplot, kde=True)

# 右上角显示相关系数
for i, j in zip(*np.triu_indices_from(corr, 1)):
    g.axes[i, j].annotate(f'corr:{corr.iloc[i, j]:.2f}', (0.5, 0.5),
                          textcoords='axes fraction', ha='center', va='center', fontsize=20)

# plt.savefig("第一种.pdf", format='pdf', bbox_inches='tight')
plt.show()


# 改进——修改相关系数部分为热图

corr = df.corr()
g = sns.PairGrid(df)
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot, kde=True)
fig = g.fig

# 右上角替换为热力图（每个子图显示一个相关系数）
for i, j in zip(*np.triu_indices_from(corr, 1)):
    ax = g.axes[i, j]
    sns.heatmap(pd.DataFrame([[corr.iloc[i, j]]]), cmap=sns.diverging_palette(240, 10, as_cmap=True),
                cbar=False, annot=True, fmt=".2f", square=True, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 20})  # 设置相关系数数字字体大小为12

# 在图形旁边添加全局色条
fig.subplots_adjust(right=0.85)  # 调整图形右侧空间以显示色条
cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # 定义色条位置和大小
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=sns.diverging_palette(240, 10, as_cmap=True), norm=norm)
sm.set_array([])  # 为空数组设置色条
fig.colorbar(sm, cax=cbar_ax)  # 添加全局色条
# plt.savefig("第二种.pdf", format='pdf', bbox_inches='tight')
plt.show()




corr = df.corr()
n = len(df.columns)
fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

# 绘制每个位置的散点图和直方图
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            # 对角线：绘制直方图
            sns.histplot(df.iloc[:, i], kde=True, ax=ax)
        elif i > j:
            # 下三角：绘制散点图
            sns.scatterplot(x=df.iloc[:, j], y=df.iloc[:, i], ax=ax)
        else:
            # 上三角：绘制热图显示相关系数
            sns.heatmap(pd.DataFrame([[corr.iloc[i, j]]]), cmap=sns.diverging_palette(240, 10, as_cmap=True),
                        cbar=False, annot=True, fmt=".2f", square=True, ax=ax, vmin=-1, vmax=1,
                        annot_kws={"size": 20})  # 设置相关系数数字字体大小

        # 隐藏不需要的轴标签
        if i < n - 1:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 在图形旁边添加全局色条
fig.subplots_adjust(right=0.85)  # 调整图形右侧空间以显示色条
cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # 定义色条位置和大小
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=sns.diverging_palette(240, 10, as_cmap=True), norm=norm)
sm.set_array([])  # 为空数组设置色条
fig.colorbar(sm, cax=cbar_ax)  # 添加全局色条

# plt.savefig("改进后的可视化.pdf", format='pdf', bbox_inches='tight')
plt.show()







































