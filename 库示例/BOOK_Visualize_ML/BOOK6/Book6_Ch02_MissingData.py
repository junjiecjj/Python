

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html





from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 可视化鸢尾花数据
X, y = load_iris(as_frame=True, return_X_y=True)
X.head()

iris_df = X.copy()
iris_df['species'] = y
sns.pairplot(iris_df, hue='species', palette = "bright")


# 随机引入缺失值缺失值
X_NaN = X.copy()
mask = np.random.uniform(0,1,size = X_NaN.shape)
mask = (mask <= 0.4)
X_NaN[mask] = np.NaN
print(X_NaN.tail)

iris_df_NaN = X_NaN.copy()
iris_df_NaN['species'] = y
sns.pairplot(iris_df_NaN, hue='species', palette = "bright")

# 可视化缺失值
# 用isna()方法查找缺失值
is_NaN = iris_df_NaN.isna()
# print(is_NaN)

fig, ax = plt.subplots()
sns.heatmap(is_NaN, ax = ax, cmap='gray_r', cbar=False)

not_NaN = iris_df_NaN.notna()
# sum_rows = not_NaN.sum(axis=1)
# print(not_NaN)


fig, ax = plt.subplots()
ax = sns.heatmap(not_NaN, cmap='gray_r', cbar=False)

import missingno as msno
# missingno has to be installed first
# pip install missingno

msno.matrix(iris_df_NaN)

# 总结缺失值

print("\nCount total NaN at each column:\n", X_NaN.isnull().sum())

print("\nPercentage of NaN at each column:\n", X_NaN.isnull().sum()/len(X_NaN)*100)




# 删除含有缺失值的行
#%% drop missing value rows

X_NaN_drop = X_NaN.dropna(axis=0)

iris_df_NaN_drop = pd.DataFrame(X_NaN_drop, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_drop['species'] = y
sns.pairplot(iris_df_NaN_drop, hue='species', palette = "bright")

#%% imputing the data using median imputation



# 单变量插补
from sklearn.impute import SimpleImputer

# The imputation strategy:
# 'mean', replace missing values using the mean along each column
# 'median', replace missing values using the median along each column
# 'most_frequent', replace missing using the most frequent value along each column
# 'constant', replace missing values with fill_value

si = SimpleImputer(strategy='median')
# impute training data
X_NaN_median = si.fit_transform(X_NaN)

iris_df_NaN_median = pd.DataFrame(X_NaN_median, columns=X_NaN.columns, index=X_NaN.index)

iris_df_NaN_median['species'] = y
sns.pairplot(iris_df_NaN_median, hue='species', palette = "bright")





# kNN插补
from sklearn.impute import KNNImputer

knni = KNNImputer(n_neighbors=5)
X_NaN_kNN = knni.fit_transform(X_NaN)

iris_df_NaN_kNN = pd.DataFrame(X_NaN_kNN, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_kNN['species'] = y

sns.pairplot(iris_df_NaN_kNN, hue='species', palette = "bright")





# 多变量插补
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

rf_imp = IterativeImputer(estimator= RandomForestRegressor(random_state=0), max_iter=20)
X_NaN_RF = rf_imp.fit_transform(X_NaN)

iris_df_NaN_RF = pd.DataFrame(X_NaN_RF, columns=X_NaN.columns, index=X_NaN.index)
iris_df_NaN_RF['species'] = y
sns.pairplot(iris_df_NaN_RF, hue='species', palette = "bright")


































































































































































































































