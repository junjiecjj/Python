#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:42:56 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485888&idx=1&sn=22ceb1574856dd7189ee10858ec4cf0b&chksm=c0e5d306f7925a1020310564f8bd8caf5866b43ab094b35dcfc1d00f42f41c675998f5f5ea3a&cur_album_id=3445855686331105280&scene=190#rd


"""


#%%>>>>>>>>>>>>>>>>>>>>>>> 1. T检验 (T-Test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 设置随机种子以便重现
np.random.seed(42)

# 生成虚拟数据集
group_A_scores = np.random.normal(loc=75, scale=10, size=100)  # 方法A的成绩，均值75，标准差10
group_B_scores = np.random.normal(loc=70, scale=12, size=100)  # 方法B的成绩，均值70，标准差12

# 进行独立样本T检验
t_stat, p_value = ttest_ind(group_A_scores, group_B_scores)

# 打印T检验结果
print(f"T检验统计量: {t_stat:.2f}")
print(f"P值: {p_value:.4f}")

# 创建一个DataFrame来存储数据
df = pd.DataFrame({
    'Scores': np.concatenate([group_A_scores, group_B_scores]),
    'Group': ['A'] * len(group_A_scores) + ['B'] * len(group_B_scores)
})

# 设置图形大小
plt.figure(figsize=(14, 7))

# 子图1：直方图和密度图
plt.subplot(1, 2, 1)
sns.histplot(df, x="Scores", hue="Group", kde=True, stat="density", common_norm=False, palette="dark", alpha=0.6)
plt.title('Score Distribution (Histogram & Density)')

# 子图2：箱线图和小提琴图
plt.subplot(1, 2, 2)
sns.boxplot(data=df,  x='Group', y='Scores', hue = 'Group',legend=False, palette="Set2", showmeans=True)
sns.violinplot(data=df, x='Group', y='Scores', hue = 'Group',legend=False, palette="Set2", inner=None, alpha=0.3)
plt.title('Boxplot & Violin Plot of Scores')

# 显示图形
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 卡方检验 (Chi-Square Test)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 创建虚拟数据集
np.random.seed(42)
data = {
    'Age Group': np.random.choice(['18-25', '26-35', '36-45'], size=300, p=[0.3, 0.4, 0.3]),
    'Product Type': np.random.choice(['Product A', 'Product B', 'Product C'], size=300, p=[0.4, 0.3, 0.3])
}

df = pd.DataFrame(data)

# 创建交叉表（列联表）
contingency_table = pd.crosstab(df['Age Group'], df['Product Type'])

# 执行卡方检验
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 输出卡方检验结果
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# 绘制数据分析的图形
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# 图1：热力图（显示实际的交叉表数据）
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax[0, 0])
ax[0, 0].set_title('Observed Frequencies (Heatmap)')
ax[0, 0].set_xlabel('Product Type')
ax[0, 0].set_ylabel('Age Group')

# 图2：热力图（显示期望的频数数据）
sns.heatmap(expected, annot=True, fmt='.2f', cmap='Oranges', ax=ax[0, 1])
ax[0, 1].set_title('Expected Frequencies (Heatmap)')
ax[0, 1].set_xlabel('Product Type')
ax[0, 1].set_ylabel('Age Group')

# 图3：堆叠条形图（展示不同年龄组的产品购买情况）
contingency_table.plot(kind='bar', stacked=True, ax=ax[1, 0], colormap='viridis')
ax[1, 0].set_title('Product Type by Age Group (Stacked Bar)')
ax[1, 0].set_xlabel('Age Group')
ax[1, 0].set_ylabel('Frequency')

# 图4：群组条形图（展示不同产品类型的年龄组分布）
contingency_table.T.plot(kind='bar', ax=ax[1, 1], colormap='Set2')
ax[1, 1].set_title('Age Group by Product Type (Grouped Bar)')
ax[1, 1].set_xlabel('Product Type')
ax[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 方差分析 (ANOVA)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 生成虚拟数据集
n = 300  # 每组样本数量
group_A = np.random.normal(loc=120, scale=5, size=n)  # 药物A的血压变化
group_B = np.random.normal(loc=115, scale=7, size=n)  # 药物B的血压变化
group_C = np.random.normal(loc=130, scale=6, size=n)  # 药物C的血压变化

# 创建数据框
df = pd.DataFrame({
    'Blood_Pressure_Change': np.concatenate([group_A, group_B, group_C]),
    'Drug': ['A']*n + ['B']*n + ['C']*n
})

# 方差分析 (ANOVA)
model = ols('Blood_Pressure_Change ~ C(Drug)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 画图
plt.figure(figsize=(14, 7))

# 子图1：箱线图
plt.subplot(1, 2, 1)
sns.boxplot(x='Drug', y='Blood_Pressure_Change', data=df, hue = 'Drug', legend = False,)
plt.title('Blood Pressure Change by Drug (Boxplot)')
plt.xlabel('Drug')
plt.ylabel('Blood Pressure Change')

# 子图2：均值误差条形图
plt.subplot(1, 2, 2)
sns.pointplot(x='Drug', y='Blood_Pressure_Change', data=df, capsize=.2)
plt.title('Mean Blood Pressure Change by Drug with Error Bars')
plt.xlabel('Drug')
plt.ylabel('Mean Blood Pressure Change')

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 皮尔逊相关系数检验 (Pearson Correlation Test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# 生成虚拟数据
np.random.seed(42)
study_hours = np.random.normal(5, 2, 100)  # 学习时间（小时）
exam_scores = 50 + 10 * study_hours + np.random.normal(0, 5, 100)  # 考试成绩

# 创建 DataFrame
data = pd.DataFrame({
    'Study Hours': study_hours,
    'Exam Scores': exam_scores
})

# 计算皮尔逊相关系数
corr, p_value = pearsonr(data['Study Hours'], data['Exam Scores'])

# 线性回归
X = data['Study Hours'].values.reshape(-1, 1)
y = data['Exam Scores'].values.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
data['Predicted Scores'] = reg.predict(X)
data['Residuals'] = y - np.array(data['Predicted Scores']).reshape(-1, 1,)

# 绘图
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# 散点图和回归线
sns.scatterplot(x='Study Hours', y='Exam Scores', data=data, ax=axs[0, 0])
sns.lineplot(x='Study Hours', y='Predicted Scores', data=data, color='red', ax=axs[0, 0])
axs[0, 0].set_title('Scatter Plot with Regression Line')
axs[0, 0].text(0.05, 0.95, f'Pearson r = {corr:.2f}\nP-value = {p_value:.2e}',
               transform=axs[0, 0].transAxes, fontsize=12, verticalalignment='top')

# 残差图
sns.residplot(x='Study Hours', y='Exam Scores', data=data, ax=axs[0, 1])
axs[0, 1].set_title('Residual Plot')

# 学习时间和成绩的分布图
sns.histplot(data['Study Hours'], kde=True, color='blue', ax=axs[1, 0], bins=15)
sns.histplot(data['Exam Scores'], kde=True, color='green', ax=axs[1, 0], bins=15)
axs[1, 0].set_title('Distribution of Study Hours and Exam Scores')
axs[1, 0].legend(['Study Hours', 'Exam Scores'])

# 相关系数热力图
corr_matrix = data[['Study Hours', 'Exam Scores']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axs[1, 1])
axs[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 正态性检验 (Normality Test, 如Shapiro-Wilk Test)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot

# 生成虚拟数据集
np.random.seed(0)
# 正态分布数据
normal_data = np.random.normal(loc=0, scale=1, size=1000)
# 非正态分布数据（对数正态分布）
non_normal_data = np.random.lognormal(mean=0, sigma=1, size=1000)

# Shapiro-Wilk 检验
shapiro_normal = shapiro(normal_data)
shapiro_non_normal = shapiro(non_normal_data)

# 创建图形
plt.figure(figsize=(14, 10))

# 1. 正态分布数据的直方图和核密度图
plt.subplot(2, 2, 1)
sns.histplot(normal_data, kde=True, color='skyblue')
plt.title(f'Normal Data Histogram & KDE\nShapiro-Wilk p-value: {shapiro_normal.pvalue:.4f}')

# 2. 非正态分布数据的直方图和核密度图
plt.subplot(2, 2, 2)
sns.histplot(non_normal_data, kde=True, color='salmon')
plt.title(f'Non-Normal Data Histogram & KDE\nShapiro-Wilk p-value: {shapiro_non_normal.pvalue:.4f}')

# 3. 正态分布数据的Q-Q图
plt.subplot(2, 2, 3)
probplot(normal_data, dist="norm", plot=plt)
plt.title('Normal Data Q-Q Plot')

# 4. 非正态分布数据的Q-Q图
plt.subplot(2, 2, 4)
probplot(non_normal_data, dist="norm", plot=plt)
plt.title('Non-Normal Data Q-Q Plot')

# 调整布局
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 非参数检验 (Non-parametric Tests, 如Mann-Whitney U检验)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
group_A = np.random.normal(loc=55, scale=10, size=100)
group_B = np.random.normal(loc=60, scale=15, size=100)

# 创建DataFrame
df = pd.DataFrame({
    'Treatment': ['A']*100 + ['B']*100,
    'Effectiveness': np.concatenate([group_A, group_B])
})

# Mann-Whitney U检验
stat, p = mannwhitneyu(group_A, group_B)
print(f'Mann-Whitney U 检验结果: 统计量={stat}, p值={p}')

# 设置绘图样式
sns.set(style="whitegrid")

# 创建图形对象
plt.figure(figsize=(14, 7))

# 子图1: 箱线图
plt.subplot(1, 2, 1)
sns.boxplot(x='Treatment', y='Effectiveness', data=df, hue = 'Treatment', legend = False, palette="Set3")
plt.title('Boxplot of Treatment Effectiveness')

# 子图2: 小提琴图
plt.subplot(1, 2, 2)
sns.violinplot(x='Treatment', y='Effectiveness', data=df, hue = 'Treatment', legend = False, palette="Set3")
plt.title('Violin Plot of Treatment Effectiveness')

# 显示图形
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 方差齐性检验 (Levene's Test)

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 生成虚拟数据集
np.random.seed(42)

# 组1：正态分布，较小方差
group1 = np.random.normal(loc=50, scale=5, size=100)

# 组2：正态分布，较大方差
group2 = np.random.normal(loc=50, scale=15, size=100)

# 组3：偏态分布，较大方差
group3 = np.random.gamma(shape=2, scale=10, size=100)

# 整合数据
data = pd.DataFrame({
    'Value': np.concatenate([group1, group2, group3]),
    'Group': ['Group 1']*100 + ['Group 2']*100 + ['Group 3']*100
})

# 进行Levene's Test
levene_stat, levene_p = stats.levene(group1, group2, group3)

print(f"Levene's Test Statistic: {levene_stat:.3f}")
print(f"P-value: {levene_p:.3f}")

# 图形可视化
plt.figure(figsize=(14, 8))

# 子图1: 箱线图
plt.subplot(2, 2, 1)
sns.boxplot(data=data, x='Group', y='Value', hue = 'Group', legend = False, palette='Set2')
plt.title("Boxplot of Values by Group")
plt.xlabel("Group")
plt.ylabel("Value")

# 子图2: 散点图（带抖动）
plt.subplot(2, 2, 2)
sns.stripplot(data=data, x='Group', y='Value', hue = 'Group', legend = False, jitter=True, palette='Set1')
plt.title("Stripplot of Values by Group")
plt.xlabel("Group")
plt.ylabel("Value")

# 子图3: 密度图
plt.subplot(2, 2, 3)
sns.kdeplot(group1, fill=True, label='Group 1', color='blue')
sns.kdeplot(group2, fill=True, label='Group 2', color='orange')
sns.kdeplot(group3, fill=True, label='Group 3', color='green')
plt.title("Density Plot of Values by Group")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()

# 子图4: Q-Q图
plt.subplot(2, 2, 4)
stats.probplot(group2, dist="norm", plot=plt)
plt.title("Q-Q Plot of Group 2")

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 科尔莫哥洛夫-斯米尔诺夫检验 (Kolmogorov-Smirnov Test, KS检验)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成两个虚拟数据集
np.random.seed(42)
data1 = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布, 均值为0, 标准差为1
data2 = np.random.normal(loc=0.5, scale=1.2, size=1000)  # 正态分布, 均值为0.5, 标准差为1.2

# 进行KS检验
ks_statistic, p_value = stats.ks_2samp(data1, data2)

# 打印检验结果
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# 创建图形
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 数据分布图（直方图）
ax[0].hist(data1, bins=30, alpha=0.5, label='Data1: N(0, 1)', color='blue')
ax[0].hist(data2, bins=30, alpha=0.5, label='Data2: N(0.5, 1.2)', color='red')
ax[0].legend(loc='upper right')
ax[0].set_title('Histogram of Data1 and Data2')
ax[0].set_xlabel('Value')
ax[0].set_ylabel('Frequency')

# 图2: 累积分布函数（CDF）图
# 计算CDF
x = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)
cdf1 = np.array([np.sum(data1 <= xi) / len(data1) for xi in x])
cdf2 = np.array([np.sum(data2 <= xi) / len(data2) for xi in x])

# 绘制CDF
ax[1].plot(x, cdf1, label='Data1 CDF', color='blue')
ax[1].plot(x, cdf2, label='Data2 CDF', color='red')
ax[1].legend(loc='lower right')
ax[1].set_title('CDF of Data1 and Data2')
ax[1].set_xlabel('Value')
ax[1].set_ylabel('Cumulative Probability')

# 显示图形
plt.suptitle(f'Kolmogorov-Smirnov Test: KS Statistic={ks_statistic:.3f}, P-value={p_value:.3f}', fontsize=16)
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 逻辑回归检验 (Logistic Regression Test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                           n_classes=2, random_state=42)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立逻辑回归模型
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 绘制混淆矩阵
cmd.plot(ax=ax[0])
ax[0].set_title('Confusion Matrix')

# 绘制ROC曲线
ax[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='gray', linestyle = '--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic')
ax[1].legend(loc='lower right')

# 显示图形
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 线性回归检验 (Linear Regression Test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, Y)
Y_pred = model.predict(X)

# 计算残差
residuals = Y - Y_pred

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：散点图和拟合线
axes[0, 0].scatter(X, Y, color='blue', label='Data')
axes[0, 0].plot(X, Y_pred, color='red', linewidth=2, label='Fitted line')
axes[0, 0].set_title('Scatter Plot with Linear Regression Line')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].legend()

# 图2：残差的分布直方图
axes[0, 1].hist(residuals, bins=20, color='green', edgecolor='black')
axes[0, 1].set_title('Distribution of Residuals')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')

# 图3：残差图
axes[1, 0].scatter(Y_pred, residuals, color='purple')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Residual Plot')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')

# 图4：残差与自变量X的关系
axes[1, 1].scatter(X, residuals, color='orange')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('Residuals vs. X')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Residuals')

# 调整布局和显示图形
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 1. t检验（T-test）


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 数据
group1 = np.array([85, 90, 78, 92, 88])
group2 = np.array([95, 85, 89, 91, 93])

# 计算均值和标准差
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

# 独立样本 t 检验
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# 可视化
plt.figure(figsize=(10, 6))

# 绘制两组数据的箱线图
plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
plt.title('Boxplot of Two Groups')
plt.ylabel('Scores')

# 显示均值和标准差
plt.text(1, mean1, f'Mean: {mean1:.2f}\nSD: {std1:.2f}', horizontalalignment='center', verticalalignment='bottom')
plt.text(2, mean2, f'Mean: {mean2:.2f}\nSD: {std2:.2f}', horizontalalignment='center', verticalalignment='bottom')

plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 卡方检验（Chi-Square Test）
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 观察到的频数表
observed = np.array([[60, 40],
                     [30, 70]])

# 执行卡方检验
chi2, p, dof, expected = stats.chi2_contingency(observed)

# 打印结果
print("Chi-squared:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)

# 可视化观察到的和期望的频数
categories = ['拥有汽车', '没有汽车']
groups = ['男性', '女性']

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 观察到的频数
im1 = ax[0].imshow(observed, cmap='Blues', aspect='auto')
ax[0].set_title('观察到的频数')
ax[0].set_xticks(np.arange(len(categories)))
ax[0].set_yticks(np.arange(len(groups)))
ax[0].set_xticklabels(categories)
ax[0].set_yticklabels(groups)

# 期望的频数
im2 = ax[1].imshow(expected, cmap='Reds', aspect='auto')
ax[1].set_title('期望的频数')
ax[1].set_xticks(np.arange(len(categories)))
ax[1].set_yticks(np.arange(len(groups)))
ax[1].set_xticklabels(categories)
ax[1].set_yticklabels(groups)

# 添加数值标签
for i in range(len(groups)):
    for j in range(len(categories)):
        text = ax[0].text(j, i, int(observed[i, j]),
                          ha="center", va="center", color="black")
        text = ax[1].text(j, i, int(expected[i, j]),
                          ha="center", va="center", color="black")

# 调整布局
fig.tight_layout()
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])

# 显示图形
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 方差分析（ANOVA）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 设置随机种子
np.random.seed(42)

# 生成数据
n = 10  # 每组样本数
mu_A, mu_B, mu_C = 20, 22, 19  # 每组均值
sigma = 2  # 标准差

A = np.random.normal(mu_A, sigma, n)
B = np.random.normal(mu_B, sigma, n)
C = np.random.normal(mu_C, sigma, n)

# 创建DataFrame
df = pd.DataFrame({
    '肥料': ['A']*n + ['B']*n + ['C']*n,
    '生长高度': np.concatenate([A, B, C])
})

# 打印前5行数据
print(df.head())

# 使用Seaborn绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='肥料', y='生长高度', data=df)
plt.title('不同肥料类型对植物生长高度的影响')
plt.xlabel('肥料类型')
plt.ylabel('生长高度（厘米）')
plt.show()

# 使用statsmodels进行单因素方差分析
model = ols('生长高度 ~ C(肥料)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)



#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 线性回归（Linear Regression）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成模拟数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 广告费用
y = 4 + 3 * X + np.random.randn(100, 1)  # 销售额 (带有噪声)

# 绘制原始数据
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='原始数据')
plt.xlabel('广告费用')
plt.ylabel('销售额')
plt.title('广告费用与销售额的关系')
plt.legend()
plt.show()

# 创建线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 预测值
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)

# 绘制回归线
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X_new, y_predict, color='red', linewidth=2, label='回归线')
plt.xlabel('广告费用')
plt.ylabel('销售额')
plt.title('广告费用与销售额的线性回归')
plt.legend()
plt.show()

# 输出回归系数
print(f'回归截距 (intercept): {lin_reg.intercept_[0]}')
print(f'回归系数 (slope): {lin_reg.coef_[0][0]}')




#%%>>>>>>>>>>>>>>>>>>>>>>> 5. Logistic回归（Logistic Regression）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 生成二分类数据集
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建Logistic回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 绘制原理图
def plot_decision_boundary(model, X, y):
    # 设置图像范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# 绘制训练集的决策边界
plot_decision_boundary(model, X_train, y_train)




#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 皮尔逊相关系数（Pearson Correlation Coefficient）

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 数据
study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90])

# 计算皮尔逊相关系数
corr_coef, _ = pearsonr(study_hours, exam_scores)
print(f'皮尔逊相关系数: {corr_coef}')

# 绘制散点图和回归线
plt.scatter(study_hours, exam_scores, color='blue', label='数据点')
plt.xlabel('学习时间（小时）')
plt.ylabel('考试成绩（分数）')
plt.title('学习时间与考试成绩的关系')

# 计算回归线
m, b = np.polyfit(study_hours, exam_scores, 1)
plt.plot(study_hours, m * study_hours + b, color='red', label=f'回归线 (y={m:.2f}x + {b:.2f})')

plt.legend()
plt.grid(True)
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 斯皮尔曼秩相关系数（Spearman's Rank Correlation Coefficient）

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# 数据
math_scores = [85, 80, 78, 90, 70]
physics_scores = [88, 82, 78, 92, 72]

# 创建 DataFrame
data = pd.DataFrame({
    'Math': math_scores,
    'Physics': physics_scores
})

# 计算斯皮尔曼秩相关系数
corr, p_value = spearmanr(data['Math'], data['Physics'])

print(f"Spearman's rank correlation coefficient: {corr:.2f}")
print(f"P-value: {p_value:.4f}")

# 绘制散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Math', y='Physics', data=data)
plt.title(f"Scatter plot of Math and Physics scores\nSpearman's rank correlation: {corr:.2f}, P-value: {p_value:.4f}")
plt.xlabel('Math Scores')
plt.ylabel('Physics Scores')
plt.grid(True)
plt.show()

# 绘制秩图
data['Math_rank'] = data['Math'].rank()
data['Physics_rank'] = data['Physics'].rank()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Math_rank', y='Physics_rank', data=data)
plt.title('Rank Scatter Plot of Math and Physics scores')
plt.xlabel('Math Rank')
plt.ylabel('Physics Rank')
plt.grid(True)
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 非参数检验（Non-parametric Tests）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# 样本数据
group_a = [85, 78, 92, 88, 76]
group_b = [80, 70, 90, 82, 74]

# Mann-Whitney U 检验
u_statistic, p_value = mannwhitneyu(group_a, group_b)

print(f"U 统计量: {u_statistic}")
print(f"P 值: {p_value}")

# 绘制分布图
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 绘制小提琴图
sns.violinplot(data=[group_a, group_b])
plt.xticks([0, 1], ['组A', '组B'])
plt.title('组A与组B的成绩分布')

# 显示图像
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 9. F检验（F-test）
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成样本数据
np.random.seed(0)  # 设置随机种子以保证结果可重复
sample_A = np.random.normal(loc=20, scale=5, size=30)
sample_B = np.random.normal(loc=22, scale=10, size=30)

# 计算样本组的方差
var_A = np.var(sample_A, ddof=1)
var_B = np.var(sample_B, ddof=1)

# 计算F统计量
F = var_A / var_B

# 进行F检验
dof_A = len(sample_A) - 1
dof_B = len(sample_B) - 1
p_value = 1 - stats.f.cdf(F, dof_A, dof_B)

# 打印结果
print(f"样本组A的方差: {var_A:.2f}")
print(f"样本组B的方差: {var_B:.2f}")
print(f"F统计量: {F:.2f}")
print(f"p值: {p_value:.4f}")

# 绘制F分布图
x = np.linspace(0, 5, 500)
y = stats.f.pdf(x, dof_A, dof_B)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label=f'F-distribution (dof={dof_A},{dof_B})')
plt.axvline(F, color='r', linestyle='--', label=f'F-statistic = {F:.2f}')
plt.fill_between(x, 0, y, where=(x >= F), color='r', alpha=0.5)
plt.title('F-distribution with F-statistic')
plt.xlabel('F value')
plt.ylabel('Density')
plt.legend()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 贝叶斯检验（Bayesian Test）

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 先验概率
prior_fair = 0.5
prior_unfair = 0.5

# 数据
observed_heads = 6
observed_tails = 4

# 似然函数
def likelihood_fair(heads, tails):
    return 0.5**(heads + tails)

def likelihood_unfair(heads, tails, p):
    return (p**heads) * ((1-p)**tails)

# 贝叶斯更新
posterior_fair = likelihood_fair(observed_heads, observed_tails) * prior_fair

# 计算后验概率分布
p_values = np.linspace(0, 1, 100)
posterior_unfair = [likelihood_unfair(observed_heads, observed_tails, p) for p in p_values]
posterior_unfair = np.array(posterior_unfair) * prior_unfair

# 正则化
posterior_unfair /= np.sum(posterior_unfair)
posterior_unfair /= np.max(posterior_unfair)  # 归一化到最大值为1

# 画图
plt.figure(figsize=(10, 6))

# 画出先验分布
plt.subplot(2, 1, 1)
plt.title('Prior Distribution')
plt.bar(['Fair', 'Unfair'], [prior_fair, prior_unfair], color=['blue', 'orange'])

# 画出后验分布
plt.subplot(2, 1, 2)
plt.title('Posterior Distribution')
plt.plot(p_values, posterior_unfair, label='Unfair')
plt.axhline(posterior_fair, color='blue', linestyle='--', label='Fair')
plt.legend()

plt.tight_layout()
plt.show()







