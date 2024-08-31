#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:43:19 2024

@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. t检验（t-test）
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 生成样本数据
np.random.seed(42)
group1 = np.random.normal(5, 1.5, 30)
group2 = np.random.normal(6, 1.8, 30)

# t检验
t_stat, p_value = stats.ttest_ind(group1, group2)

# 数据分析图
plt.figure(figsize=(10, 6))
sns.histplot(group1, color="skyblue", kde=True, label="Group 1", bins=10)
sns.histplot(group2, color="salmon", kde=True, label="Group 2", bins=10)
plt.axvline(np.mean(group1), color="blue", linestyle="--", label="Mean Group 1")
plt.axvline(np.mean(group2), color="red", linestyle="--", label="Mean Group 2")
plt.title(f't-test: t-statistic={t_stat:.3f}, p-value={p_value:.3f}')
plt.legend()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 卡方检验（Chi-square Test）

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 生成分类数据
data = {'Gender': ['Male', 'Female'] * 50, 'Purchase': ['Yes', 'No'] * 50}
df = pd.DataFrame(data)

# 创建交叉表
contingency_table = pd.crosstab(df['Gender'], df['Purchase'])

# 卡方检验
chi2, p, dof, ex = chi2_contingency(contingency_table)

# 数据分析图
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title(f'Chi-square Test: chi2={chi2:.3f}, p-value={p:.3f}')
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 3. ANOVA（方差分析）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# 生成多组数据
np.random.seed(0)
group1 = np.random.normal(5, 1, 30)
group2 = np.random.normal(6, 1, 30)
group3 = np.random.normal(7, 1, 30)

# 方差分析
f_stat, p_value = f_oneway(group1, group2, group3)

# 数据分析图
data = pd.DataFrame({'Value': np.concatenate([group1, group2, group3]),
                     'Group': ['Group1'] * 30 + ['Group2'] * 30 + ['Group3'] * 30})

plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Value', data=data, palette="Set2")
plt.title(f'ANOVA: F-statistic={f_stat:.3f}, p-value={p_value:.3f}')
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 4. F检验（F-test）
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成两组数据
np.random.seed(42)
group1 = np.random.normal(5, 2, 50)
group2 = np.random.normal(5, 1.5, 50)

# F检验
f_stat, p_value = stats.f_oneway(group1, group2)

# 数据分析图
plt.figure(figsize=(10, 6))
plt.hist(group1, bins=15, alpha=0.7, label='Group 1', color='orange')
plt.hist(group2, bins=15, alpha=0.7, label='Group 2', color='purple')
plt.axvline(np.var(group1), color='orange', linestyle='--', label='Variance Group 1')
plt.axvline(np.var(group2), color='purple', linestyle='--', label='Variance Group 2')
plt.title(f'F-test: F-statistic={f_stat:.3f}, p-value={p_value:.3f}')
plt.legend()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 5. Z检验（Z-test）
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成大样本数据
np.random.seed(24)
group1 = np.random.normal(5.5, 0.8, 100)
group2 = np.random.normal(5.2, 0.9, 100)

# Z检验
z_stat, p_value = stats.ttest_ind(group1, group2)

# 数据分析图
plt.figure(figsize=(10, 6))
plt.hist(group1, bins=20, alpha=0.7, label='Group 1', color='green')
plt.hist(group2, bins=20, alpha=0.7, label='Group 2', color='red')
plt.axvline(np.mean(group1), color='green', linestyle='--', label='Mean Group 1')
plt.axvline(np.mean(group2), color='red', linestyle='--', label='Mean Group 2')
plt.title(f'Z-test: Z-statistic={z_stat:.3f}, p-value={p_value:.3f}')
plt.legend()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 非参数检验

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 生成非正态分布数据
np.random.seed(0)
group1 = np.random.exponential(scale=1, size=30)
group2 = np.random.exponential(scale=1.5, size=30)

# Mann-Whitney U检验
u_stat, p_value = stats.mannwhitneyu(group1, group2)

# 数据分析图
plt.figure(figsize=(10, 6))
sns.kdeplot(group1, shade=True, label="Group 1", color="darkorange")
sns.kdeplot(group2, shade=True, label="Group 2", color="teal")
plt.title(f'Mann-Whitney U Test: U-statistic={u_stat:.3f}, p-value={p_value:.3f}')
plt.legend()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 假设检验的p值分析

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
data = np.random.normal(0, 1, 100)

# 单样本t检验
t_stat, p_value = stats.ttest_1samp(data, popmean=0)

# 数据分析图
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True, color="royalblue", bins=15)
plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
plt.title(f't-test: t-statistic={t_stat:.3f}, p-value={p_value:.3f}')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 双样本比例检验
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 定义两个样本的成功数和样本数
success_a = 30
n_a = 100
success_b = 45
n_b = 100

# 计算样本比例
p1 = success_a / n_a
p2 = success_b / n_b

# 合并比例
p = (success_a + success_b) / (n_a + n_b)

# Z检验
z_stat = (p1 - p2) / np.sqrt(p * (1 - p) * (1/n_a + 1/n_b))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# 数据分析图
labels = ['Sample A', 'Sample B']
proportions = [p1, p2]

plt.figure(figsize=(8, 6))
plt.bar(labels, proportions, color=['lightcoral', 'lightblue'])
plt.title(f'Proportion Comparison: Z-statistic={z_stat:.3f}, p-value={p_value:.3f}')
plt.ylim(0, 1)
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 回归分析中的假设检验

import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# 生成回归数据
np.random.seed(10)
X = np.random.normal(0, 1, 100)
y = 2 * X + np.random.normal(0, 0.5, 100)

# 回归分析
X = sm.add_constant(X)  # 添加常数项（截距项）
model = sm.OLS(y, X).fit()

# 获取t值和p值
t_values = model.tvalues
p_values = model.pvalues

# 数据分析图
plt.figure(figsize=(8, 6))
sns.regplot(x=X[:, 1], y=y, color="magenta")
plt.title(f'Regression Analysis: t-value={t_values[1]:.3f}, p-value={p_values[1]:.3f}')
plt.xlabel('X')
plt.ylabel('y')
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 模型诊断中的假设检验



import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
import seaborn as sns
import matplotlib.pyplot as plt

# 生成回归数据
np.random.seed(1)
X = np.random.normal(0, 1, 100)
y = 3 * X + np.random.normal(0, 2, 100)

# 回归分析
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Breusch-Pagan检验
bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
bp_stat = bp_test[0]
bp_pvalue = bp_test[1]

# 数据分析图
plt.figure(figsize=(8, 6))
sns.residplot(x=X[:, 1], y=model.resid, lowess=True, color="darkgreen")
plt.title(f'Breusch-Pagan Test: BP-statistic={bp_stat:.3f}, p-value={bp_pvalue:.3f}')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()




















