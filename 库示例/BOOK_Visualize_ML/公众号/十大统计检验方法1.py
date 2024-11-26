#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:20:50 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488338&idx=1&sn=bf611c5e512c6b5da46c8d3ace571ba0&chksm=c13852d6e06b08e8298f13d7695043a8c2c2386c6d4c6c12e708cc356f2de2acc7f83367fceb&mpshare=1&scene=1&srcid=1124umcasJZuqs0crfNVMvFB&sharer_shareinfo=ab3166cdb09abd1cbc4aff5e50dd4b73&sharer_shareinfo_first=ab3166cdb09abd1cbc4aff5e50dd4b73&exportkey=n_ChQIAhIQh4uccbCnx8gzr%2Fj79gt3yhKfAgIE97dBBAEAAAAAAGrbB72LCkgAAAAOpnltbLcz9gKNyK89dVj0RwJc0ceBZEUc9oGdZ1q%2F0grevPZX5osLfEYf%2FleOh3fxFBxyLFd%2B2qjqdqvtd7UTWQwqtN5Rec6hdybRohbPjqkwsgxlvg23y7emzBzcOgJcIvkITQL5T5%2FBkBM%2FZ5oCHUKTllbd%2FjKbV93WtLgwM9oUNIYhIgvoRyFSgXdNkRrjhu0IjBA2m1KwzTsetNJpwi05Qjwpi3tKU9RgGDrlZgOuJKJnRcjW7uP43SAsLWsfMCShUdmEYw97vbXGtcUPEHfd1zqhLiIzji1QPDhUDcC3lDKKVpHLfHPUG7T4kYGZnohNY8of4RE1j5nlHU21O%2F0dBEc3mmAW&acctmode=0&pass_ticket=6MSi1WiWPjeObBO2cW0Q27woEtN1RrzqquFLfPOJvsyWW%2FfaK93mmrdlCMrCpNHC&wx_header=0#rd

"""


#%%>>>>>>>>>>>>>>>>>>>>> 1. T检验 (T-Test)
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
sns.boxplot(x='Group', y='Scores', data=df, palette="Set2", showmeans=True)
sns.violinplot(x='Group', y='Scores', data=df, palette="Set2", inner=None, alpha=0.3)
plt.title('Boxplot & Violin Plot of Scores')

# 显示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>> 2. 卡方检验 (Chi-Square Test)

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

#%%>>>>>>>>>>>>>>>>>>>>> 3. 方差分析 (ANOVA)
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
sns.boxplot(x='Drug', y='Blood_Pressure_Change', data=df)
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



#%%>>>>>>>>>>>>>>>>>>>>> 4. 皮尔逊相关系数检验 (Pearson Correlation Test)
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
data['Residuals'] = y - data['Predicted Scores']

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



#%%>>>>>>>>>>>>>>>>>>>>> 5. 正态性检验 (Normality Test, 如Shapiro-Wilk Test)

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



#%%>>>>>>>>>>>>>>>>>>>>> 6. 非参数检验 (Non-parametric Tests, 如Mann-Whitney U检验)

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
sns.boxplot(x='Treatment', y='Effectiveness', data=df, palette="Set3")
plt.title('Boxplot of Treatment Effectiveness')

# 子图2: 小提琴图
plt.subplot(1, 2, 2)
sns.violinplot(x='Treatment', y='Effectiveness', data=df, palette="Set3")
plt.title('Violin Plot of Treatment Effectiveness')

# 显示图形
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>> 7. 方差齐性检验 (Levene's Test)

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
sns.boxplot(x='Group', y='Value', data=data, palette='Set2')
plt.title("Boxplot of Values by Group")
plt.xlabel("Group")
plt.ylabel("Value")

# 子图2: 散点图（带抖动）
plt.subplot(2, 2, 2)
sns.stripplot(x='Group', y='Value', data=data, jitter=True, palette='Set1')
plt.title("Stripplot of Values by Group")
plt.xlabel("Group")
plt.ylabel("Value")

# 子图3: 密度图
plt.subplot(2, 2, 3)
sns.kdeplot(group1, shade=True, label='Group 1', color='blue')
sns.kdeplot(group2, shade=True, label='Group 2', color='orange')
sns.kdeplot(group3, shade=True, label='Group 3', color='green')
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


#%%>>>>>>>>>>>>>>>>>>>>> 8. 科尔莫哥洛夫-斯米尔诺夫检验 (Kolmogorov-Smirnov Test, KS检验)
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

#%%>>>>>>>>>>>>>>>>>>>>> 9. 逻辑回归检验 (Logistic Regression Test)

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
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

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
ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic')
ax[1].legend(loc='lower right')

# 显示图形
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>> 10. 线性回归检验 (Linear Regression Test)

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






#%%>>>>>>>>>>>>>>>>>>>>>





















