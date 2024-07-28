








#%% 8. 高斯过程（Gaussian Processes）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 生成数据
np.random.seed(1)
X = np.linspace(0, 10, 100)
y = np.sin(X) + 0.5 * np.random.randn(100)

# 训练数据
X_train = X[::5][:, np.newaxis]
y_train = y[::5]

# 定义高斯过程模型
kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合模型
gp.fit(X_train, y_train)

# 预测
X_pred = np.linspace(0, 10, 1000)[:, np.newaxis] # (1000, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True) # (1000,), (1000,)

# 画图
plt.figure(figsize=(14, 6))

# 原始数据和训练数据
plt.subplot(1, 2, 1)
plt.scatter(X, y, c='k', label='Data')
plt.scatter(X_train, y_train, c='r', label='Train Data')
plt.title('Data and Train Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 高斯过程回归预测
plt.subplot(1, 2, 2)
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k', label='95% Confidence Interval')
plt.scatter(X_train, y_train, c='r', label='Train Data')
plt.title('Gaussian Process Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()

























































































































































































































































































