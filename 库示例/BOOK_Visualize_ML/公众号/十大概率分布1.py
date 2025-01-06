#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:06:23 2025

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247487882&idx=1&sn=d92e53a8bbc8cd84ee8637227aa45e6b&chksm=c152756eba600026e2e034905633dbf29335c0ee405786b17a7de2a2d2368b8a5490269591cd&mpshare=1&scene=1&srcid=01049l66UcWkE2LGIJ7JwVLK&sharer_shareinfo=5dd6e5dbe22bd526e732074e16d63f25&sharer_shareinfo_first=5dd6e5dbe22bd526e732074e16d63f25&exportkey=n_ChQIAhIQtj%2Fwmwr6%2B2leSpmAsjlCthKfAgIE97dBBAEAAAAAAHhLJzaF9IoAAAAOpnltbLcz9gKNyK89dVj0So3B%2FFSh8ukwqFbF4%2FtsuSQpTWcxWsnyO1Xne551jTxFNaAJKtX8JOW5hb0lQWG27jLM9si3rl%2Fmj61Zfo%2F%2FCUJdt2ey12GiaY72%2BuUS9pYg6Y4brb%2BLzpyeyZfrpAv4MxaeFHDAGGqI8ev7%2Fhb62ZizyFGGAYyQYORUouRkYYsGfDpRHIt6ZXA904C09eQ4XWg4qazr%2Bog4lFHKeg4sMzhK6no1YYfWIGv1ythPbmve1SnHz%2FAPoNAhVoISH%2BQ5CEylyshZkrqLiqX5oaTgIV0BNakva5l7N1VAcn8DLGx%2FJlDRqEmtRyA25GStmBjVFmD10xIr0rcw&acctmode=0&pass_ticket=DiAsGCDkN62bG1IRCChlh9gEV4TZnRFCQTwej4jeYHEdK%2FpHJ0dsZo8GupEt3FWg&wx_header=0#rd


@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>>>> 1. 正态分布（Normal Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成虚拟数据集
mu, sigma = 0, 1  # 平均值和标准差
data = np.random.normal(mu, sigma, 1000)

# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Multidimensional Analysis of Normal Distribution Data', fontsize=16)

# 概率密度函数
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
axs[0, 0].plot(x, pdf, color='blue', linewidth=2, label='PDF')
axs[0, 0].fill_between(x, pdf, color='cyan', alpha=0.3)
axs[0, 0].set_title('Probability Density Function (PDF)', fontsize=12)
axs[0, 0].legend()

# 直方图
sns.histplot(data, bins=30, kde=True, color='purple', ax=axs[0, 1])
axs[0, 1].set_title('Histogram with KDE', fontsize=12)
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Frequency')

# 累积分布函数 (CDF)
axs[0, 2].plot(x, stats.norm.cdf(x, mu, sigma), color='green', label='CDF', linewidth=2)
axs[0, 2].fill_between(x, stats.norm.cdf(x, mu, sigma), color='lime', alpha=0.3)
axs[0, 2].set_title('Cumulative Distribution Function (CDF)', fontsize=12)
axs[0, 2].legend()

# 箱线图 (Box plot)
sns.boxplot(data, color='orange', ax=axs[1, 0])
axs[1, 0].set_title('Box Plot', fontsize=12)
axs[1, 0].set_xlabel('Value')

# Q-Q图 (Quantile-Quantile plot)
stats.probplot(data, dist="norm", plot=axs[1, 1])
axs[1, 1].get_lines()[1].set_color('red')
axs[1, 1].get_lines()[1].set_linestyle('--')
axs[1, 1].set_title('Q-Q Plot', fontsize=12)

# 数据描述信息
axs[1, 2].axis('off')  # 关闭坐标轴
stats_text = f'''
Mean = {np.mean(data):.2f}
Standard Deviation = {np.std(data):.2f}
Min = {np.min(data):.2f}
Max = {np.max(data):.2f}
Median = {np.median(data):.2f}
'''
axs[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

# 调整子图布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>> 2. 伯努利分布（Bernoulli Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置种子和样本数
np.random.seed(42)
sample_size = 1000

# 生成伯努利分布数据（假设成功的概率为0.4）
p = 0.4
data = np.random.binomial(1, p, sample_size)

# 创建图形和子图
plt.figure(figsize=(12, 8))
plt.suptitle("Bernoulli Distribution and Data Analysis", fontsize=18)

# 1. 概率分布条形图
plt.subplot(2, 2, 1)
sns.countplot(data, palette="bright")
plt.title("Probability Distribution (Bar Chart)", fontsize=14)
plt.xlabel("Outcome (0 = Failure, 1 = Success)")
plt.ylabel("Frequency")

# 2. 累计分布函数（CDF）图
plt.subplot(2, 2, 2)
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.plot(sorted_data, cdf, color='darkorange', marker='o', linestyle='-', linewidth=2)
plt.title("Cumulative Distribution Function (CDF)", fontsize=14)
plt.xlabel("Outcome")
plt.ylabel("Cumulative Probability")

# 3. 箱线图
plt.subplot(2, 2, 3)
sns.boxplot(data, color="skyblue")
plt.title("Box Plot", fontsize=14)
plt.xlabel("Outcome")

# 4. 直方图
plt.subplot(2, 2, 4)
plt.hist(data, bins=2, color="limegreen", edgecolor="black")
plt.title("Histogram", fontsize=14)
plt.xlabel("Outcome")
plt.ylabel("Frequency")

# 调整布局和显示
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>> 3. 二项分布（Binomial Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

# 设置二项分布参数
n = 20  # 试验次数
p = 0.5  # 每次试验成功概率
size = 1000  # 模拟样本数

# 生成二项分布数据
data = binom.rvs(n=n, p=p, size=size)

# 创建画布
plt.figure(figsize=(16, 12))

# 1. 概率质量函数（PMF）图
plt.subplot(2, 2, 1)
x = np.arange(0, n + 1)
pmf_values = binom.pmf(x, n, p)
plt.stem(x, pmf_values, basefmt=" ", linefmt='C1-', markerfmt='C1o', label="PMF")  # Removed use_line_collection
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.title("Probability Mass Function (PMF) of Binomial Distribution")
plt.legend()

# 2. 累积分布函数（CDF）图
plt.subplot(2, 2, 2)
cdf_values = binom.cdf(x, n, p)
plt.plot(x, cdf_values, 'C2-', marker='o', label="CDF")
plt.xlabel("Number of Successes")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function (CDF) of Binomial Distribution")
plt.legend()

# 3. 数据直方图
plt.subplot(2, 2, 3)
sns.histplot(data, bins=n + 1, kde=False, color="C0", edgecolor="black", stat="probability")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Simulated Data (Histogram)")

# 4. 箱线图
plt.subplot(2, 2, 4)
sns.boxplot(data=data, color="C3")
plt.xlabel("Sample Data")
plt.title("Box Plot of Simulated Data")

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>> 4. 多项分布（Multinomial Distribution）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子，确保可重复性
np.random.seed(42)

# 定义商品类别数量
categories = ['A', 'B', 'C']

# 定义多项分布的参数（每种商品的平均销售概率）
probs = [0.5, 0.3, 0.2]  # 各个类别的销售概率

# 模拟销售数据 - 假设每天观察100次销售
n_trials = 100
sales_data = np.random.multinomial(n_trials, probs, size=1000)

# 计算每类商品的平均销售数量
mean_sales = sales_data.mean(axis=0)

# 图形设置
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
fig.suptitle('Product Sales Probability Distribution and Data Analysis', fontsize=16)

# 1. Bar Chart
axes[0].bar(categories, mean_sales, color=['red', 'green', 'blue'], alpha=0.7)
axes[0].set_title('Average Sales Quantity Bar Chart')
axes[0].set_xlabel('Product Category')
axes[0].set_ylabel('Average Sales Quantity')
for i, val in enumerate(mean_sales):
    axes[0].text(i, val + 2, f'{val:.1f}', ha='center', color='black')

# 2. CDF
cumulative_sales = np.cumsum(mean_sales / np.sum(mean_sales))
axes[1].plot(categories, cumulative_sales, marker='o', linestyle='-', color='purple')
axes[1].fill_between(categories, cumulative_sales, color='purple', alpha=0.2)
axes[1].set_title('Cumulative Distribution Function (CDF)')
axes[1].set_xlabel('Product Category')
axes[1].set_ylabel('Cumulative Probability')

# 3. PDF
for idx, cat in enumerate(categories):
    sns.kdeplot(sales_data[:, idx], label=cat, ax=axes[2], linewidth=2)
axes[2].set_title('Probability Density Function (PDF) of Sales Data')
axes[2].set_xlabel('Sales Quantity')
axes[2].set_ylabel('Density')
axes[2].legend(title="Product Category")


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>> 5. 泊松分布（Poisson Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# 设置随机种子，确保结果可重复
np.random.seed(42)

# 生成虚拟数据：泊松分布的随机数据，假设平均进店次数为4
lambda_val = 4
data = poisson.rvs(mu=lambda_val, size=1000)

# 设置图形风格和颜色
sns.set(style="whitegrid")
plt.figure(figsize=(14, 10))

# 绘制直方图
plt.subplot(2, 2, 1)
sns.histplot(data, bins=15, kde=False, color="skyblue", edgecolor="black")
plt.title("Histogram of Poisson Distribution (λ=4)", fontsize=14)
plt.xlabel("Number of Customers per Hour")
plt.ylabel("Frequency")

# 绘制核密度估计图 (KDE)
plt.subplot(2, 2, 2)
sns.kdeplot(data, color="red", fill=True, alpha=0.6, linewidth=2)
plt.title("KDE of Poisson Distribution", fontsize=14)
plt.xlabel("Number of Customers per Hour")
plt.ylabel("Density")

# 绘制累计分布函数 (CDF)
plt.subplot(2, 2, 3)
x = np.sort(data)
y = np.arange(1, len(x) + 1) / len(x)
plt.step(x, y, color="purple", where="post", linewidth=2)
plt.title("CDF of Poisson Distribution", fontsize=14)
plt.xlabel("Number of Customers per Hour")
plt.ylabel("Cumulative Probability")

# 绘制箱线图 (Box Plot)
plt.subplot(2, 2, 4)
sns.boxplot(data=data, color="lime")
plt.title("Box Plot of Poisson Data", fontsize=14)
plt.xlabel("Distribution")

# 显示所有图
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>> 6. 指数分布（Exponential Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机种子，确保结果可重复
np.random.seed(42)

# 生成虚拟数据集：指数分布数据
lambda_param = 1.5  # 指数分布的参数
data = np.random.exponential(1 / lambda_param, 1000)  # 生成1000个样本数据

# 创建子图画布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Exponential Distribution Data Analysis', fontsize=16, fontweight='bold')

# 绘制概率密度函数 (PDF)
sns.histplot(data, bins=30, kde=True, color="dodgerblue", ax=axes[0, 0])
axes[0, 0].set_title('Probability Density Function (PDF)', fontsize=14)
axes[0, 0].set_xlabel('Data Value')
axes[0, 0].set_ylabel('Probability Density')

# 绘制累积分布函数 (CDF)
sorted_data = np.sort(data)
cdf = np.arange(1, len(data) + 1) / len(data)
axes[0, 1].plot(sorted_data, cdf, color="darkorange", lw=2)
axes[0, 1].set_title('Cumulative Distribution Function (CDF)', fontsize=14)
axes[0, 1].set_xlabel('Data Value')
axes[0, 1].set_ylabel('Cumulative Probability')

# 绘制箱线图 (Boxplot)
sns.boxplot(data=data, color="lightcoral", ax=axes[1, 0])
axes[1, 0].set_title('Boxplot', fontsize=14)
axes[1, 0].set_xlabel('Exponential Distribution Data')

# 绘制QQ图 (QQ plot)
stats.probplot(data, dist="expon", plot=axes[1, 1])
axes[1, 1].get_lines()[1].set_color("green")  # 参考线颜色
axes[1, 1].set_title('QQ Plot', fontsize=14)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>7. 卡方分布（Chi-Square Distribution）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机数种子以保证结果可重复
np.random.seed(42)

# 定义卡方分布的自由度
df = 5

# 生成卡方分布的虚拟数据
chi_data = np.random.chisquare(df, 1000)

# 创建一个多图表画布
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Chi-Square Distribution Analysis", fontsize=16)

# 图1：概率密度函数曲线图（PDF）曲线
x = np.linspace(0, 20, 1000)
pdf_y = stats.chi2.pdf(x, df)
axs[0, 0].plot(x, pdf_y, color='magenta', lw=2, label=f'Chi-Square PDF (df={df})')
axs[0, 0].fill_between(x, pdf_y, color='magenta', alpha=0.3)
axs[0, 0].set_title("Probability Density Function (PDF)")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("Density")
axs[0, 0].legend()

# 图2：直方图（Histogram）
sns.histplot(chi_data, bins=30, color='orange', kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Histogram with KDE")
axs[0, 1].set_xlabel("Chi-Square Values")
axs[0, 1].set_ylabel("Frequency")

# 图3：箱线图（Box Plot）
sns.boxplot(chi_data, color='cyan', ax=axs[1, 0], orient='h')
axs[1, 0].set_title("Box Plot of Chi-Square Distribution")
axs[1, 0].set_xlabel("Chi-Square Values")

# 图4：QQ图（Q-Q Plot）
stats.probplot(chi_data, dist="chi2", sparams=(df,), plot=axs[1, 1])
axs[1, 1].get_lines()[1].set_color('blue')  # 设置QQ图的趋势线颜色
axs[1, 1].set_title("Q-Q Plot for Chi-Square Distribution")

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>> 8. t 分布（t-Distribution）
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置虚拟数据集和 t 分布参数
np.random.seed(42)
df = 5  # 自由度
sample_size = 1000
data = np.random.standard_t(df, sample_size)

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("T-Distribution Analysis with Degree of Freedom (df) = 5", fontsize=16)

# 1. 概率密度函数 (PDF)
x = np.linspace(-5, 5, 100)
pdf = stats.t.pdf(x, df)
axs[0, 0].plot(x, pdf, color="blue", lw=2, label="t-PDF (df=5)")
axs[0, 0].fill_between(x, pdf, color="skyblue", alpha=0.5)
axs[0, 0].set_title("Probability Density Function (PDF)", fontsize=12)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("Density")
axs[0, 0].legend()

# 2. 累积分布函数 (CDF)
cdf = stats.t.cdf(x, df)
axs[0, 1].plot(x, cdf, color="green", lw=2, label="t-CDF (df=5)")
axs[0, 1].fill_between(x, cdf, color="lightgreen", alpha=0.5)
axs[0, 1].set_title("Cumulative Distribution Function (CDF)", fontsize=12)
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("Cumulative Probability")
axs[0, 1].legend()

# 3. 直方图和 PDF 曲线
axs[1, 0].hist(data, bins=30, density=True, color="orange", alpha=0.6, label="Sample Data Histogram")
axs[1, 0].plot(x, pdf, color="blue", lw=2, linestyle="--", label="t-PDF (df=5)")
axs[1, 0].set_title("Histogram of Sample Data with t-PDF", fontsize=12)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("Density")
axs[1, 0].legend()

# 4. QQ图
stats.probplot(data, dist="t", sparams=(df,), plot=axs[1, 1])
axs[1, 1].get_lines()[1].set_color("purple")
axs[1, 1].set_title("Q-Q Plot for t-Distribution", fontsize=12)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>> 9. Gamma 分布（Gamma Distribution）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机种子
np.random.seed(42)

# 定义Gamma分布的参数
shape = 2.0  # 形状参数
scale = 2.0  # 尺度参数

# 生成随机数据
data = np.random.gamma(shape, scale, 1000)

# 创建一个图形和多个子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Gamma Distribution Analysis', fontsize=16)

# 1. Gamma分布的概率密度函数 (PDF)
x = np.linspace(0, 20, 1000)
pdf = stats.gamma.pdf(x, a=shape, scale=scale)
axs[0, 0].plot(x, pdf, 'r-', lw=2, label='Gamma PDF')
axs[0, 0].set_title('Gamma Distribution PDF')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Probability Density')
axs[0, 0].legend()

# 2. 直方图
sns.histplot(data, bins=30, kde=True, color='blue', ax=axs[0, 1])
axs[0, 1].set_title('Histogram of Random Samples')
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Frequency')

# 3. 箱形图
sns.boxplot(x=data, ax=axs[1, 0], color='green')
axs[1, 0].set_title('Box Plot of Random Samples')
axs[1, 0].set_xlabel('Value')

# 4. Q-Q图
stats.probplot(data, dist="gamma", sparams=(shape, 0, scale), plot=axs[1, 1])
axs[1, 1].set_title('Q-Q Plot against Gamma Distribution')

# 调整子图布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>> 10. Beta 分布（Beta Distribution）

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 设置Beta分布的参数
alpha, beta_params = 2, 5  # 通过调整alpha和beta参数可以控制分布的形状

# 生成Beta分布的数据
x = np.linspace(0, 1, 1000)
y = beta.pdf(x, alpha, beta_params)

# 样本数据
np.random.seed(42)
sample_data = np.random.beta(alpha, beta_params, 1000)

# 创建画布
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# 子图1：Beta分布的PDF曲线
axs[0, 0].plot(x, y, color='orange', lw=2, label=f'Beta PDF (α={alpha}, β={beta_params})')
axs[0, 0].fill_between(x, y, color='orange', alpha=0.2)
axs[0, 0].set_title("Beta Distribution PDF", fontsize=14)
axs[0, 0].set_xlabel("x", fontsize=12)
axs[0, 0].set_ylabel("Density", fontsize=12)
axs[0, 0].legend()

# 子图2：直方图和核密度估计 (KDE) 图
axs[0, 1].hist(sample_data, bins=30, density=True, alpha=0.6, color='dodgerblue', edgecolor='black', label="Sample Histogram")
axs[0, 1].plot(x, y, color='red', lw=2, label="Beta PDF")
axs[0, 1].set_title("Histogram and KDE", fontsize=14)
axs[0, 1].set_xlabel("x", fontsize=12)
axs[0, 1].set_ylabel("Density", fontsize=12)
axs[0, 1].legend()

# 子图3：累积分布函数 (CDF)
y_cdf = beta.cdf(x, alpha, beta_params)
axs[1, 0].plot(x, y_cdf, color='purple', lw=2, label=f'Beta CDF (α={alpha}, β={beta_params})')
axs[1, 0].fill_between(x, y_cdf, color='purple', alpha=0.2)
axs[1, 0].set_title("Cumulative Distribution Function (CDF)", fontsize=14)
axs[1, 0].set_xlabel("x", fontsize=12)
axs[1, 0].set_ylabel("Cumulative Probability", fontsize=12)
axs[1, 0].legend()

# 子图4：箱线图和小提琴图
axs[1, 1].boxplot(sample_data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen', color='green'),
                  medianprops=dict(color='green'), whiskerprops=dict(color='green'))
axs[1, 1].violinplot(sample_data, vert=False, showmedians=True, showmeans=True)
axs[1, 1].set_title("Box Plot and Violin Plot", fontsize=14)
axs[1, 1].set_xlabel("Sample Data", fontsize=12)

# 显示图形
plt.suptitle("Complex Analysis of Beta Distribution", fontsize=16, fontweight='bold')
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>





#%%>>>>>>>>>>>>>>>>>>>>>>>>




