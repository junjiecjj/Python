#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:45:42 2024

@author: jack


https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486820&idx=1&sn=2fb611e6bd9d2ab1b8f371c3656da52e&chksm=c1b2f4f990be3ac527a276e1eae3da2ed22e7a5e00ecdeb6771f3ed855c6acbe6f35e3de9dba&mpshare=1&scene=1&srcid=0918BX8AGEJos89z831V0VL2&sharer_shareinfo=3471011affd27e6aff0a4b8b0ca58c30&sharer_shareinfo_first=3471011affd27e6aff0a4b8b0ca58c30&exportkey=n_ChQIAhIQriXhpdUeTeU0%2F0rlxnsKshKfAgIE97dBBAEAAAAAAOh2Jb464esAAAAOpnltbLcz9gKNyK89dVj0ymj8rFLJ8rx00y1GEML5QWD2fnz%2BhqVeJOSH2EjmjOFsr%2BuD5PeXt3jboAtSLOmRmL67UNdnn40rB3FfszcNv8aXoBDvbP33RCzHviR6etpMlHsTNaf%2F9i%2FMvuqbiQy3RXE3Cp6T82abHKCEgBWrlCohKxhPDYFQEoKeAZIrwcWVwq%2BkweaA6KZNoqzuqDASa5qYQQIMsg93EREeFwn83%2Bt80OV6dMkKXT1qoicZu%2FFLvIky5VsSJHJu0XbLPOrvr56xxy2VMKx96QEK%2Fp4Aht6Wbzv5Qh1Gzo56CzV6t4BZ33lnlvr9Ua5%2F4bxHZg4vJBLCY98%2FsItv&acctmode=0&pass_ticket=bE%2BSO3jBmFOiTPckODRuW6f5VD2LlVdqhHL7fxclykHaHFSImDzcDwNd5nsynKKo&wx_header=0#rd
"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. Shapiro-Wilk 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置随机种子，便于结果复现
np.random.seed(42)

# 生成虚拟数据集：一个正态分布和一个非正态分布数据
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布
data_skewed = np.random.exponential(scale=2, size=1000)  # 非正态分布

# Shapiro-Wilk检验
stat_normal, p_normal = stats.shapiro(data_normal)
stat_skewed, p_skewed = stats.shapiro(data_skewed)

# 创建图形
fig, axs = plt.subplots(2, 3, figsize=(16, 10))

# 设置颜色
colors = ['#FF6F61', '#6B5B95', '#88B04B']

# 直方图 - 正态分布
sns.histplot(data_normal, kde=True, color=colors[0], ax=axs[0, 0])
axs[0, 0].set_title(f'Normal Distribution\nShapiro-Wilk p-value: {p_normal:.3f}', fontsize=12)
axs[0, 0].set_xlabel('Value')
axs[0, 0].set_ylabel('Frequency')

# 直方图 - 非正态分布
sns.histplot(data_skewed, kde=True, color=colors[1], ax=axs[1, 0])
axs[1, 0].set_title(f'Skewed Distribution\nShapiro-Wilk p-value: {p_skewed:.3f}', fontsize=12)
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')

# Q-Q图 - 正态分布
stats.probplot(data_normal, dist="norm", plot=axs[0, 1])
axs[0, 1].get_lines()[1].set_color(colors[0])  # 线颜色
axs[0, 1].get_lines()[0].set_markerfacecolor(colors[2])  # 点颜色
axs[0, 1].set_title('Q-Q Plot (Normal)')

# Q-Q图 - 非正态分布
stats.probplot(data_skewed, dist="norm", plot=axs[1, 1])
axs[1, 1].get_lines()[1].set_color(colors[1])  # 线颜色
axs[1, 1].get_lines()[0].set_markerfacecolor(colors[2])  # 点颜色
axs[1, 1].set_title('Q-Q Plot (Skewed)')

# 箱线图 - 正态分布
sns.boxplot(data=data_normal, color=colors[0], ax=axs[0, 2])
axs[0, 2].set_title('Boxplot (Normal)')
axs[0, 2].set_xlabel('Data')

# 箱线图 - 非正态分布
sns.boxplot(data=data_skewed, color=colors[1], ax=axs[1, 2])
axs[1, 2].set_title('Boxplot (Skewed)')
axs[1, 2].set_xlabel('Data')

# 调整布局
plt.tight_layout()
plt.show()







#%%>>>>>>>>>>>>>>>>>>>>>>> 2. Kolmogorov-Smirnov 检验 (K-S 检验)


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成虚拟数据集
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_non_normal = np.random.exponential(scale=1, size=1000)  # 非正态分布数据

# Kolmogorov-Smirnov 检验
ks_stat_normal, p_value_normal = stats.kstest(data_normal, 'norm')
ks_stat_non_normal, p_value_non_normal = stats.kstest(data_non_normal, 'norm')

# 绘制直方图和累积分布函数（CDF）
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle('Kolmogorov-Smirnov Test for Normality', fontsize=16)

# 正态分布数据的直方图
axes[0, 0].hist(data_normal, bins=30, density=True, alpha=0.6, color='c', label='Histogram (Normal)')
xmin, xmax = axes[0, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, data_normal.mean(), data_normal.std())
axes[0, 0].plot(x, p, 'k', linewidth=2, label='PDF (Normal)')
axes[0, 0].set_title(f'Normal Data\nK-S Statistic: {ks_stat_normal:.4f}, P-value: {p_value_normal:.4f}')
axes[0, 0].legend()

# 非正态分布数据的直方图
axes[0, 1].hist(data_non_normal, bins=30, density=True, alpha=0.6, color='m', label='Histogram (Non-Normal)')
xmin, xmax = axes[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, data_non_normal.mean(), data_non_normal.std())
axes[0, 1].plot(x, p, 'k', linewidth=2, label='PDF (Non-Normal)')
axes[0, 1].set_title(f'Non-Normal Data\nK-S Statistic: {ks_stat_non_normal:.4f}, P-value: {p_value_non_normal:.4f}')
axes[0, 1].legend()

# 正态分布数据的累积分布函数（CDF）
x_cdf = np.sort(data_normal)
cdf_normal = np.arange(1, len(data_normal)+1) / len(data_normal)
axes[1, 0].plot(x_cdf, cdf_normal, marker='.', linestyle='none', color='c', label='Empirical CDF (Normal)')
axes[1, 0].plot(x, stats.norm.cdf(x, data_normal.mean(), data_normal.std()), 'k-', lw=2, label='Theoretical CDF (Normal)')
axes[1, 0].set_title('CDF (Normal Data)')
axes[1, 0].legend()

# 非正态分布数据的累积分布函数（CDF）
x_cdf_non_normal = np.sort(data_non_normal)
cdf_non_normal = np.arange(1, len(data_non_normal)+1) / len(data_non_normal)
axes[1, 1].plot(x_cdf_non_normal, cdf_non_normal, marker='.', linestyle='none', color='m', label='Empirical CDF (Non-Normal)')
axes[1, 1].plot(x, stats.norm.cdf(x, data_non_normal.mean(), data_non_normal.std()), 'k-', lw=2, label='Theoretical CDF (Normal)')
axes[1, 1].set_title('CDF (Non-Normal Data)')
axes[1, 1].legend()

plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 3. Anderson-Darling 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import anderson

# 生成虚拟数据集 (一个偏态数据和一个正态数据)
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000)
data_skewed = np.random.exponential(scale=2, size=1000)

# Anderson-Darling 检验函数
def anderson_darling_test(data, title):
    result = anderson(data)
    print(f"Anderson-Darling Test for {title}:")
    print(f"Statistic: {result.statistic}")
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        print(f"At {sl}% significance level, critical value is {cv}")
        if result.statistic > cv:
            print(f"Reject null hypothesis (not normal) at {sl}% level")
        else:
            print(f"Fail to reject null hypothesis (normal) at {sl}% level")
    print("\n")

# 可视化分析函数
def visualize_data(data, title, color):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左侧绘制直方图 + KDE
    sns.histplot(data, bins=30, kde=True, color=color, ax=axes[0])
    axes[0].set_title(f"Histogram and KDE - {title}", fontsize=14)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # 右侧绘制 Q-Q 图
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(color)  # 设置Q-Q图中数据点的颜色
    axes[1].get_lines()[1].set_color('red')  # 设置Q-Q图中拟合线的颜色
    axes[1].set_title(f"Q-Q Plot - {title}", fontsize=14)

    plt.tight_layout()
    plt.show()

# 对两组数据进行Anderson-Darling正态性检验
anderson_darling_test(data_normal, "Normal Data")
anderson_darling_test(data_skewed, "Skewed Data")

# 对两组数据进行可视化分析
visualize_data(data_normal, "Normal Data", "blue")
visualize_data(data_skewed, "Skewed Data", "green")




#%%>>>>>>>>>>>>>>>>>>>>>>> 4. Lilliefors 检验

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors
import scipy.stats as stats

# 生成虚拟数据集
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_non_normal = np.random.exponential(scale=2, size=1000)  # 指数分布数据

# Lilliefors 检验
stat_normal, p_normal = lilliefors(data_normal)
stat_non_normal, p_non_normal = lilliefors(data_non_normal)

# 创建一个图形
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 正态数据的直方图和 KDE
sns.histplot(data_normal, kde=True, color='dodgerblue', ax=axs[0, 0])
axs[0, 0].set_title(f'Normal Data Histogram\nLilliefors p={p_normal:.3f}')

# 正态数据的 QQ 图
stats.probplot(data_normal, dist="norm", plot=axs[0, 1])
axs[0, 1].get_lines()[1].set_color('red')
axs[0, 1].set_title('Normal Data QQ Plot')

# 正态数据的核密度估计 (KDE)
sns.kdeplot(data_normal, color='green', shade=True, ax=axs[0, 2])
axs[0, 2].set_title('Normal Data KDE')

# 非正态数据的直方图和 KDE
sns.histplot(data_non_normal, kde=True, color='orangered', ax=axs[1, 0])
axs[1, 0].set_title(f'Non-Normal Data Histogram\nLilliefors p={p_non_normal:.3f}')

# 非正态数据的 QQ 图
stats.probplot(data_non_normal, dist="norm", plot=axs[1, 1])
axs[1, 1].get_lines()[1].set_color('purple')
axs[1, 1].set_title('Non-Normal Data QQ Plot')

# 非正态数据的核密度估计 (KDE)
sns.kdeplot(data_non_normal, color='orange', shade=True, ax=axs[1, 2])
axs[1, 2].set_title('Non-Normal Data KDE')

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 5. Jarque-Bera 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, norm, probplot

# 生成一个随机的虚拟数据集
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_skewed = np.random.exponential(scale=1, size=1000) - 1  # 偏态数据

# 进行 Jarque-Bera 检验
jb_stat_normal, p_normal = jarque_bera(data_normal)
jb_stat_skewed, p_skewed = jarque_bera(data_skewed)

# 打印 Jarque-Bera 检验结果
print(f"Normal Data JB Stat: {jb_stat_normal}, p-value: {p_normal}")
print(f"Skewed Data JB Stat: {jb_stat_skewed}, p-value: {p_skewed}")

# 设置图形颜色
color1 = '#ff6f61'  # 鲜艳的红色
color2 = '#6fa3ef'  # 鲜艳的蓝色

# 创建绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. 绘制正态数据的直方图和密度图
sns.histplot(data_normal, kde=True, color=color1, ax=axs[0, 0])
axs[0, 0].set_title('Normal Data Histogram & Density')

# 2. 绘制偏态数据的直方图和密度图
sns.histplot(data_skewed, kde=True, color=color2, ax=axs[0, 1])
axs[0, 1].set_title('Skewed Data Histogram & Density')

# 3. 绘制正态数据的 Q-Q 图
probplot(data_normal, dist="norm", plot=axs[1, 0])
axs[1, 0].get_lines()[0].set_color(color1)
axs[1, 0].get_lines()[1].set_color('black')
axs[1, 0].set_title('Normal Data Q-Q Plot')

# 4. 绘制偏态数据的 Q-Q 图
probplot(data_skewed, dist="norm", plot=axs[1, 1])
axs[1, 1].get_lines()[0].set_color(color2)
axs[1, 1].get_lines()[1].set_color('black')
axs[1, 1].set_title('Skewed Data Q-Q Plot')

# 调整布局，展示图形
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 6. D'Agostino's K-squared 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot, normaltest

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 生成虚拟数据集
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_non_normal = np.random.chisquare(df=2, size=1000)  # 非正态分布数据

# D'Agostino's K-squared 检验
stat_normal, p_normal = normaltest(data_normal)
stat_non_normal, p_non_normal = normaltest(data_non_normal)

# 输出检验结果
print(f"正态分布数据集的检验统计量: {stat_normal}, p值: {p_normal}")
print(f"非正态分布数据集的检验统计量: {stat_non_normal}, p值: {p_non_normal}")

# 创建图形
plt.figure(figsize=(12, 10))

# 1. 正态数据集的直方图
plt.subplot(2, 2, 1)
sns.histplot(data_normal, kde=True, color='blue', bins=30, label='Normal Data')
plt.title('Histogram - Normal Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 2. 非正态数据集的直方图
plt.subplot(2, 2, 2)
sns.histplot(data_non_normal, kde=True, color='red', bins=30, label='Non-normal Data')
plt.title('Histogram - Non-normal Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 3. 正态数据集的Q-Q图
plt.subplot(2, 2, 3)
probplot(data_normal, dist="norm", plot=plt)
plt.title('Q-Q Plot - Normal Data')

# 4. 非正态数据集的Q-Q图
plt.subplot(2, 2, 4)
probplot(data_non_normal, dist="norm", plot=plt)
plt.title('Q-Q Plot - Non-normal Data')

# 调整图形布局并展示
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. Cramér-von Mises 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

# 设置随机种子，生成虚拟数据集
np.random.seed(42)
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_non_normal = np.random.exponential(scale=1, size=1000)  # 非正态分布数据

# 进行 Cramér-von Mises 检验
def cramervonmises_test(data):
    result = stats.cramervonmises(data, 'norm')
    return result

# 进行检验
result_normal = cramervonmises_test(data_normal)
result_non_normal = cramervonmises_test(data_non_normal)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 设置 Seaborn 样式
sns.set(style="whitegrid")

# 绘制第一个数据集的直方图和 Q-Q 图（正态数据）
sns.histplot(data_normal, kde=True, color='magenta', ax=axes[0, 0])
axes[0, 0].set_title('Histogram of Normal Data', fontsize=14)
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

stats.probplot(data_normal, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[1].set_color('red')  # 设置拟合线的颜色
axes[0, 1].get_lines()[0].set_markerfacecolor('blue')  # 设置数据点颜色
axes[0, 1].set_title('Q-Q Plot of Normal Data', fontsize=14)

# 绘制第二个数据集的直方图和 Q-Q 图（非正态数据）
sns.histplot(data_non_normal, kde=True, color='cyan', ax=axes[1, 0])
axes[1, 0].set_title('Histogram of Non-Normal Data', fontsize=14)
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

stats.probplot(data_non_normal, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[1].set_color('orange')  # 设置拟合线的颜色
axes[1, 1].get_lines()[0].set_markerfacecolor('green')  # 设置数据点颜色
axes[1, 1].set_title('Q-Q Plot of Non-Normal Data', fontsize=14)

# 增加整体标题
fig.suptitle('Cramér-von Mises Normality Test with Data Visualization', fontsize=16)

# 输出 Cramér-von Mises 检验结果
print(f"Normal Data Cramér-von Mises Test Result: {result_normal}")
print(f"Non-Normal Data Cramér-von Mises Test Result: {result_non_normal}")

# 调整布局并显示图形
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 8. Shapiro-Francia 检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad

# 生成虚拟数据集
np.random.seed(0)
data_normal = np.random.normal(loc=50, scale=10, size=500)  # 正态分布数据
data_skewed = np.random.exponential(scale=2, size=500) + 30  # 偏态数据

# 创建DataFrame
df = pd.DataFrame({
    'Normal': data_normal,
    'Skewed': data_skewed
})

# Shapiro-Francia 检验函数（这里用 Anderson-Darling 检验作为替代，因为 Shapiro-Francia 实现较少）
def shapiro_francia_test(data):
    stat, p_value = normal_ad(data)
    return stat, p_value

# 正态性检验
normal_stat, normal_p = shapiro_francia_test(df['Normal'])
skewed_stat, skewed_p = shapiro_francia_test(df['Skewed'])

# 图形绘制
plt.figure(figsize=(14, 10), dpi=100)

# 图1：直方图
plt.subplot(2, 2, 1)
sns.histplot(df['Normal'], kde=True, color='cyan', bins=30, label=f'Normal (p={normal_p:.4f})')
plt.title('Histogram of Normal Data', fontsize=14)
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(df['Skewed'], kde=True, color='magenta', bins=30, label=f'Skewed (p={skewed_p:.4f})')
plt.title('Histogram of Skewed Data', fontsize=14)
plt.legend()

# 图2：QQ图
plt.subplot(2, 2, 3)
stats.probplot(df['Normal'], dist="norm", plot=plt)
plt.title('QQ Plot of Normal Data', fontsize=14)

plt.subplot(2, 2, 4)
stats.probplot(df['Skewed'], dist="norm", plot=plt)
plt.title('QQ Plot of Skewed Data', fontsize=14)

# 总体图形布局调整
plt.suptitle('Shapiro-Francia Test and Data Distribution Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 展示图形
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 9. Pearson 卡方检验
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布数据
data_non_normal = np.random.exponential(scale=2, size=1000)  # 非正态分布数据

# 进行卡方检验
def pearson_chi_square_test(data, bins=10):
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    expected_freq = len(data) / bins  # 假设期望频率均等
    chi2_stat, p_val = stats.chisquare(observed_freq, f_exp=[expected_freq]*bins)
    return chi2_stat, p_val

# 卡方检验结果
chi2_normal, p_normal = pearson_chi_square_test(data_normal)
chi2_non_normal, p_non_normal = pearson_chi_square_test(data_non_normal)

# 打印结果
print(f"正态分布数据: Chi2值 = {chi2_normal:.2f}, p值 = {p_normal:.4f}")
print(f"非正态分布数据: Chi2值 = {chi2_non_normal:.2f}, p值 = {p_non_normal:.4f}")

# 绘制图形
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

# 直方图和QQ图
colors = ['#FF6F61', '#6B5B95']  # 鲜艳的颜色

# Normal Distribution Histogram
sns.histplot(data_normal, bins=30, kde=True, color=colors[0], ax=axes[0, 0])
axes[0, 0].set_title("Histogram of Normal Distribution", fontsize=14)

# Non-normal Distribution Histogram
sns.histplot(data_non_normal, bins=30, kde=True, color=colors[1], ax=axes[0, 1])
axes[0, 1].set_title("Histogram of Non-normal Distribution", fontsize=14)

# Normal Distribution QQ Plot
stats.probplot(data_normal, dist="norm", plot=axes[1, 0])
axes[1, 0].get_lines()[1].set_color(colors[0])  # QQ plot line color
axes[1, 0].set_title("QQ Plot of Normal Distribution", fontsize=14)

# Non-normal Distribution QQ Plot
stats.probplot(data_non_normal, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[1].set_color(colors[1])  # QQ plot line color
axes[1, 1].set_title("QQ Plot of Non-normal Distribution", fontsize=14)

# Show plot
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. Z 检验


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 生成虚拟数据集
np.random.seed(42)
n = 1000
# 生成均值为0，标准差为1的正态分布数据
data = np.random.normal(0, 1, n)

# Z检验
z_statistic, p_value = stats.normaltest(data)

# 打印检验结果
print(f"Z检验统计量: {z_statistic}")
print(f"p值: {p_value}")

# 设置画布大小
plt.figure(figsize=(12, 8))

# 绘制直方图和正态分布曲线
plt.subplot(2, 2, 1)
sns.histplot(data, bins=30, kde=True, color="purple", edgecolor="black")
plt.title("Histogram with KDE", fontsize=15)
plt.axvline(np.mean(data), color='red', linestyle='dashed', linewidth=2)
plt.axvline(np.median(data), color='blue', linestyle='dashed', linewidth=2)
plt.legend({'Mean':np.mean(data), 'Median':np.median(data)})

# 绘制QQ图
plt.subplot(2, 2, 2)
stats.probplot(data, dist="norm", plot=plt)
plt.title("QQ Plot", fontsize=15)

# 绘制盒图（Boxplot）
plt.subplot(2, 2, 3)
sns.boxplot(data, color="limegreen")
plt.title("Box Plot", fontsize=15)

# 绘制小提琴图（Violin Plot）
plt.subplot(2, 2, 4)
sns.violinplot(data, color="orange")
plt.title("Violin Plot", fontsize=15)

# 调整图形布局
plt.tight_layout()
plt.show()










