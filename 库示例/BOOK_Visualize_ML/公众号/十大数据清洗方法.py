#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:17:02 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247490190&idx=1&sn=5c6902580f679588eb629b9bc815bde3&chksm=c3ade30f5d9bba9b675cb31bce55a9f13743314bb3a22f09f005a28d3f23030c830e0fddd819&mpshare=1&scene=1&srcid=0924hP5JDFiZTWtnsnDjRlwp&sharer_shareinfo=51fa4c80a5fca066893431e58f272d90&sharer_shareinfo_first=51fa4c80a5fca066893431e58f272d90&exportkey=n_ChQIAhIQGtK%2B%2B1zHOVSz1oS%2FhXiQJhKfAgIE97dBBAEAAAAAAHT4BUofoSQAAAAOpnltbLcz9gKNyK89dVj0F2Z0qwSGU%2Bu0ejLE8ho2DuIwx6DWr0AzraAEGejpeFngge5gyn9upkyolDrLqW8wIQFU4vM8IzNVqw9fwJ0alHcjAMUb8Muvqf3rekqY0LO97gTAszcgkRoue%2BVIVG%2Bm4cvj0R%2Bri%2Bm4ODxO6CV3WoS6XwDGhI%2FyBQhYC4DqEeJl1PP%2BAMTIQlyCjMi8AIOVEmh3jd%2By%2FYzVEPVFTph4LRd%2FlfxUq4IwKVf2g99JYlGMlRj03VnHHX1CTC%2Fnekpvmycq7wz7nov95PZf8aGUP%2F%2BS5h0qKHNVc8JIClKOyTZXpw5FTcn%2BJrQxWVYumUKttvL5AxZYhnws&acctmode=0&pass_ticket=%2FRTzC9bUSnDgkrKNN8%2FOBTLDSkBPHm5MJlqqfEGH8buVlMTvLeEE4%2FbmZoGUBAmp&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 缺失值处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个示例数据集
data = {
    'Age': [25, np.nan, 30, 22, np.nan, 35],
    'Salary': [50000, 60000, np.nan, 52000, 49000, np.nan]
}
df = pd.DataFrame(data)

# 缺失值处理：使用均值填补
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# 数据分析：绘制直方图和箱线图
plt.figure(figsize=(10, 6))

# 绘制直方图
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=5, color='blue', kde=True)
plt.title('Age Distribution After Imputation')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 绘制箱线图
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Salary'], color='orange')
plt.title('Salary Boxplot After Imputation')
plt.xlabel('Salary')

plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 异常值处理

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 生成一个包含异常值的示例数据集
data = {
    'Value': [10, 12, 12, 13, 12, 11, 100, 12, 11, 10, 12, 13, 11, 9, 300]
}
df = pd.DataFrame(data)

# 计算IQR
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1

# 检测异常值
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Outlier'] = ((df['Value'] < lower_bound) | (df['Value'] > upper_bound))

# 数据分析：绘制箱线图和散点图
plt.figure(figsize=(12, 6))

# 绘制箱线图
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Value'], color='lightgreen')
plt.title('Boxplot of Values')
plt.xlabel('Value')

# 绘制散点图
plt.subplot(1, 2, 2)
sns.scatterplot(x=df.index, y='Value', hue='Outlier', data=df, palette={True: 'red', False: 'blue'})
plt.title('Scatter Plot of Values with Outliers')
plt.xlabel('Index')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 重复数据移除

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个包含重复数据的示例数据集
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob'],
    'Age': [25, 30, 35, 25, 40, 30],
    'Salary': [50000, 60000, 70000, 50000, 80000, 60000]
}
df = pd.DataFrame(data)

# 移除重复数据
df_unique = df.drop_duplicates()

# 数据分析：绘制条形图和饼图
plt.figure(figsize=(12, 6))

# 绘制条形图
plt.subplot(1, 2, 1)
sns.countplot(x='Name', data=df, palette='viridis')
plt.title('Count of Names (Before Removing Duplicates)')
plt.xlabel('Name')
plt.ylabel('Count')

# 绘制饼图
plt.subplot(1, 2, 2)
df_unique['Name'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Proportion of Unique Names (After Removing Duplicates)')
plt.ylabel('')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 数据一致性处理

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个示例数据集
data = {
    'ID': [1, 2, 3, 4],
    'Date': ['2024-01-01', '01/02/2024', '2024-03-01', 'March 4, 2024'],
    'Salary': [50000, '60000$', '70000', 80000.0]
}
df = pd.DataFrame(data)

# 数据一致性处理：统一日期格式和薪资格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
df['Salary'] = df['Salary'].replace({'\$': '', ' ': ''}, regex=True).astype(float)

# 数据分析：绘制条形图和折线图
plt.figure(figsize=(12, 6))

# 绘制条形图
plt.subplot(1, 2, 1)
sns.barplot(x=df['ID'], y=df['Salary'], palette='rocket')
plt.title('Salaries by ID (After Consistency Check)')
plt.xlabel('ID')
plt.ylabel('Salary')

# 绘制折线图
plt.subplot(1, 2, 2)
plt.plot(df['ID'], df['Salary'], marker='o', linestyle='-', color='purple')
plt.title('Salary Trend by ID (After Consistency Check)')
plt.xlabel('ID')
plt.ylabel('Salary')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 数据归一化/标准化

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个示例数据集
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

# 数据归一化
df_normalized = (df - df.min()) / (df.max() - df.min())

# 数据分析：绘制散点图和热力图
plt.figure(figsize=(12, 6))

# 绘制散点图
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_normalized['Height'], y=df_normalized['Weight'], color='cyan', s=100)
plt.title('Normalized Height vs Weight')
plt.xlabel('Normalized Height')
plt.ylabel('Normalized Weight')

# 绘制热力图
plt.subplot(1, 2, 2)
sns.heatmap(df_normalized.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Normalized Data')

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 数据离散化
# 创建一个示例数据集
data = {
    'Score': [58, 73, 80, 90, 55, 88, 66, 74, 99, 61]
}
df = pd.DataFrame(data)

# 数据离散化：等宽分箱
bins = [0, 60, 70, 80, 90, 100]
labels = ['F', 'D', 'C', 'B', 'A']
df['Grade'] = pd.cut(df['Score'], bins=bins, labels=labels, right=False)

# 数据分析：绘制条形图和饼图
plt.figure(figsize=(12, 6))

# 绘制条形图
plt.subplot(1, 2, 1)
sns.countplot(x='Grade', data=df, palette='pastel')
plt.title('Grade Distribution After Discretization')
plt.xlabel('Grade')
plt.ylabel('Count')

# 绘制饼图
plt.subplot(1, 2, 2)
df['Grade'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Proportion of Grades After Discretization')
plt.ylabel('')

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 类别不平衡处理
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# 创建一个类别不平衡的数据集
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], n_informative=3,
                           n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=10)

# 处理类别不平衡：使用SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 数据分析：绘制条形图和散点图
plt.figure(figsize=(12, 6))

# 绘制条形图
plt.subplot(1, 2, 1)
sns.countplot(x=y, palette='pastel')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')

# 绘制散点图
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_res[:, 0], y=X_res[:, 1], hue=y_res, palette='deep')
plt.title('Scatter Plot After SMOTE')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 数据类型转换

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个示例数据集
data = {
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'Value': ['100', '200.5', '300']
}
df = pd.DataFrame(data)

# 数据类型转换
df['Date'] = pd.to_datetime(df['Date'])
df['Value'] = df['Value'].astype(float)

# 数据分析：绘制折线图和条形图
plt.figure(figsize=(12, 6))

# 绘制折线图
plt.subplot(1, 2, 1)
plt.plot(df['Date'], df['Value'], marker='o', color='skyblue')
plt.title('Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value')

# 绘制条形图
plt.subplot(1, 2, 2)
sns.barplot(x=df['Date'], y=df['Value'], palette='viridis')
plt.title('Bar Plot of Values by Date')
plt.xlabel('Date')
plt.ylabel('Value')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 特征工程

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 加载California Housing数据集
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 特征工程：构造多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 建立线性回归模型
model = LinearRegression()
model.fit(X_poly, y)

# 数据分析：绘制实际 vs 预测图和特征重要性图
plt.figure(figsize=(12, 6))

# 绘制实际 vs 预测图
plt.subplot(1, 2, 1)
plt.scatter(y, model.predict(X_poly), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# 绘制特征重要性图
importance = model.coef_
plt.subplot(1, 2, 2)
sns.barplot(x=importance, y=poly.get_feature_names_out(), palette='viridis')
plt.title('Feature Importance from Polynomial Regression')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()











