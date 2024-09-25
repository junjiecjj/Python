#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:25:54 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486966&idx=1&sn=61bd362135916833bd84413b2eb000d2&chksm=c194954b69f1ef5c3c5af865ca05ca5483f62a4fb8c611b3db1c873e8c4a728738921359cffb&mpshare=1&scene=1&srcid=0925HpBhViv0niynC4xYDJdG&sharer_shareinfo=c39c428c651e1071ff734c4b30c61e76&sharer_shareinfo_first=c39c428c651e1071ff734c4b30c61e76&exportkey=n_ChQIAhIQraJAtFedygDEOR%2BmFgK8PxKfAgIE97dBBAEAAAAAAGymI%2Bio1g0AAAAOpnltbLcz9gKNyK89dVj0QQW4AzJ7cBVBIXm0Sll5HoP%2F8xhAmLlTcd41iLbIgiOUTxfjPdMwytbgNM92fld5xZgRSC2NL37D3J%2BqmRW3vjmwbXjLVa9VDFFlrrGgp19SDMZtmcMyJt99g6tpPOtTJWmYVZSxK2%2FsC%2F4o41gMu72JG%2FOvl3fFrUz22zKXj5WadmNGn4dGQZOP1VhOJIW7o9kNUPQMf30N%2FDzqZJh9cf7hJJ3E9htIL9nlvaransTXyltu6qwtUqqI4n6XKDz1yAgBa7i6Q5i1jDS6H8AcnR7gTLduIkaxPQ1uPbs6an22uygXdijne696WeqcuK5pCdyARjwkJHry&acctmode=0&pass_ticket=ac%2FzhBhOwgQQDF4Wg25KSn29znWe%2F%2ByMGNTxHEw2pLP8arJeWncBpBMfQR6wvH97&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>>
# 可视化T分布：
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成T分布和标准正态分布的概率密度函数
x = np.linspace(-5, 5, 100)
t_dist_df_2 = stats.t.pdf(x, df=2)  # 自由度为2
t_dist_df_10 = stats.t.pdf(x, df=10)  # 自由度为10
normal_dist = stats.norm.pdf(x)  # 标准正态分布

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, t_dist_df_2, label='T-distribution (df=2)', color='red')
plt.plot(x, t_dist_df_10, label='T-distribution (df=10)', color='blue')
plt.plot(x, normal_dist, label='Normal Distribution', color='green', linestyle='--')
plt.title("Comparison of T-distributions and Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()




# Python实现：T检验和可视化
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(0)

# 生成两个正态分布的样本数据
group1 = np.random.normal(60, 10, 30)  # 均值60，标准差10，样本量30
group2 = np.random.normal(65, 12, 30)  # 均值65，标准差12，样本量30

# 进行独立样本T检验
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"T统计量: {t_stat}")
print(f"p值: {p_value}")

# 可视化两组数据的分布
plt.figure(figsize=(8, 6))
sns.histplot(group1, color="blue", label="Group 1", kde=True, stat="density", bins=10)
sns.histplot(group2, color="red", label="Group 2", kde=True, stat="density", bins=10)
plt.title("Group 1 vs Group 2 - T-Test Example")
plt.legend()
plt.show()



# 机器学习中的T检验应用
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载Iris数据集
data = load_iris()
X = data.data
y = data.target

# 使用ANOVA的F值检验（类似于T检验）进行特征选择
F_values, p_values = f_classif(X, y)

print("各特征的F值:", F_values)
print("各特征的p值:", p_values)

# 根据p值选择显著特征
selected_features = X[:, p_values < 0.05]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=0)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并计算准确率
y_pred = model.predict(X_test)
print("模型准确率:", accuracy_score(y_test, y_pred))
































