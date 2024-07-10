









# 线性回归
# https://blog.csdn.net/qq_35226955/article/details/118578496

# 导入包
import random
import statistics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# 产生数据
num = 50
random.seed(0)
x_data = [random.uniform(0, 10) for _ in range(num)]
# 噪音
noise =  [random.gauss(0,1) for _ in range(num)]
y_data = [0.5 * x_data[idx] + 1 + noise[idx] for idx in range(num)]


# 绘制散点图
fig, ax = plt.subplots()
ax.scatter(x_data, y_data)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,10); ax.set_ylim(-2,8)
ax.grid()



# 一元线性回归

slope, intercept = statistics.linear_regression(x_data, y_data)
# 生成一个等差数列
start, end, step = 0, 10, 0.5
x_array = []
x_i = start

while x_i <= end:
    x_array.append(x_i)
    x_i += step


# 计算x_array预测值
y_array_predicted = [slope * x_i + intercept for x_i in x_array]


# 可视化一元线性回归直线
fig, ax = plt.subplots()
ax.scatter(x_data, y_data)
ax.plot(x_array, y_array_predicted, color = 'r')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,10); ax.set_ylim(-2,8)
ax.grid()












# import matplotlib.pyplot as plt
# import numpy as np

# from sklearn.metrics import r2_score




# np.random.seed(42)

# n_samples, n_features = 200, 50
# X = np.random.randn(n_samples, n_features)
# true_coef = 3 * np.random.randn(n_features)
# # Threshold coefficients to render them non-negative
# true_coef[true_coef < 0] = 0
# y = np.dot(X, true_coef)

# # Add some noise
# y += 5 * np.random.normal(size=(n_samples,))



# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# from sklearn.linear_model import LinearRegression

# reg_nnls = LinearRegression(positive=True)
# y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
# r2_score_nnls = r2_score(y_test, y_pred_nnls)
# print("NNLS R2 score", r2_score_nnls)


# reg_ols = LinearRegression()
# y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
# r2_score_ols = r2_score(y_test, y_pred_ols)
# print("OLS R2 score", r2_score_ols)


# fig, ax = plt.subplots()
# ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

# low_x, high_x = ax.get_xlim()
# low_y, high_y = ax.get_ylim()
# low = max(low_x, low_y)
# high = min(high_x, high_y)
# ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
# ax.set_xlabel("OLS regression coefficients", fontweight="bold")
# ax.set_ylabel("NNLS regression coefficients", fontweight="bold")




















































































































































































