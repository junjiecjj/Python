#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:45:11 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486322&idx=1&sn=0c7a0e04f27d8c34b9fa28dcd1f1d5d4&chksm=c1284f8adafd826a04c570837cdc267565615d38a576f27862bddfed44b571ddcfbfee0447c0&mpshare=1&scene=1&srcid=0904JUaxxqY43w4AR2eyaioW&sharer_shareinfo=8335e098352e1123ce0c27f75d093ca1&sharer_shareinfo_first=8335e098352e1123ce0c27f75d093ca1&exportkey=n_ChQIAhIQmDqDDPxQJfBqoZLLbH36xxKfAgIE97dBBAEAAAAAAL0BGUsrpkUAAAAOpnltbLcz9gKNyK89dVj09j8%2BXfZf9GN9jCSRpSUuoFYrYvRpjGyC3qyIHP1lQp49CIpXWoVZx6zn0g%2FMYnFJVqvwmJNf7bYd%2BwYjUzJJglXsTAej9SM5jhX3s2nQbdy4JtE0DNkEq2quqehOW1Js78slptlJaaCA5x%2B8Ohb0eh8%2BMh3TQgpCkcrryquIA6X%2FXp9QF0%2FQEGFfmcDK3%2FkncsxFyZKMMXkBEKkJ6yRNhMTRI0PUlk%2BFJupWKcyVe0FD3jieIdbz%2BdbQquE3Xdax00bYNLYzGV34w6YIPx6pOu%2BNeF0Xk8H%2B7dzDElwmtAYDW1Ggc0xGCo1tUEP0N9GCkxyER0ERuLo3&acctmode=0&pass_ticket=muCJ6a%2BTupWa596KgFzgjwhTAdUZbIYww8Um7m7%2BroNxuLA8lgn6Rq%2FhEAfAc%2FlF&wx_header=0#rd
"""

#%%>>>>>>>>>>>>>> 1. 梯度下降法
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 初始化参数
theta = np.random.randn(2, 1)  # 两个参数：截距和斜率
learning_rate = 0.1
n_iterations = 100
m = len(X)

# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((m, 1)), X]

# 梯度下降优化
theta_path = []
losses = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    theta_path.append(theta)
    loss = np.mean((X_b.dot(theta) - y) ** 2)
    losses.append(loss)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b.dot(theta_path[i]), color='red', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='green', linewidth=2, label='Final Model')
axs[0].set_title('Linear Regression with Gradient Descent')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(n_iterations), losses, color='purple', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 2. 随机梯度下降
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)

# 增加多项式特征
X_poly = np.c_[np.ones((1000, 1)), X, X**2]

# 定义学习率、迭代次数和初始参数
learning_rate = 0.01
n_iterations = 1000
m = 100
theta = np.random.randn(3, 1)

# 存储每次迭代的损失
loss_history = []

# 随机梯度下降
for iteration in range(n_iterations):
    # 随机选择一个样本
    random_index = np.random.randint(m)
    xi = X_poly[random_index:random_index+1]
    yi = y[random_index:random_index+1]

    # 计算预测值
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)

    # 更新参数
    theta -= learning_rate * gradients

    # 计算所有数据点上的损失（均方误差）
    loss = np.mean((X_poly.dot(theta) - y) ** 2)
    loss_history.append(loss)

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly.dot(theta)

# 画图
plt.figure(figsize=(12, 8))

# 子图1：模型拟合结果
plt.subplot(2, 1, 1)
plt.plot(X, y, "b.", label="Training Data")
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Model Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression Model Fitting")
plt.legend(loc="upper left")

# 子图2：损失随迭代次数的变化
plt.subplot(2, 1, 2)
plt.plot(range(n_iterations), loss_history, "g-", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Loss Over Iterations")

# 显示图形
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 3. 动量法
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可重复
np.random.seed(42)

# 定义一个简单的二次函数：f(x) = x^2 + y^2
def loss_function(x, y):
    return x**2 + y**2

# 定义函数的梯度
def gradient(x, y):
    return np.array([2*x, 2*y])

# 动量法优化
def momentum_optimizer(learning_rate, momentum, num_iterations):
    # 初始化参数x, y，初始速度为0
    x, y = np.random.randn(2)
    v_x, v_y = 0, 0

    # 用于存储历史数据
    loss_history = []
    trajectory = []

    for i in range(num_iterations):
        # 计算梯度
        grad_x, grad_y = gradient(x, y)

        # 更新速度
        v_x = momentum * v_x - learning_rate * grad_x
        v_y = momentum * v_y - learning_rate * grad_y

        # 更新参数
        x += v_x
        y += v_y

        # 记录损失和轨迹
        loss_history.append(loss_function(x, y))
        trajectory.append((x, y))

    return np.array(loss_history), np.array(trajectory)

# 参数设置
learning_rate = 0.1
momentum = 0.9
num_iterations = 100

# 运行动量优化器
loss_history, trajectory = momentum_optimizer(learning_rate, momentum, num_iterations)

# 创建子图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 图1：损失函数随迭代次数的变化
ax1.plot(range(num_iterations), loss_history, 'r-', linewidth=2, label='Loss')
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# 添加第二个y轴共享x轴
ax2 = ax1.twinx()

# 图2：参数更新轨迹
ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', marker='o', markersize=5, label='Trajectory')
ax2.set_ylabel('Parameter Space', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# 设置标题和图例
fig.suptitle('Momentum Optimization: Loss and Parameter Trajectory', fontsize=16)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>> 4. Adagrad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# 生成一个虚拟的二元分类数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化模型参数
np.random.seed(42)
W = np.random.randn(2)
b = 0.0

# 定义Adagrad参数
learning_rate = 0.1
epsilon = 1e-8
W_cache = np.zeros_like(W)
b_cache = 0.0

# 记录数据
losses = []
gradients_W = []
gradients_b = []

# 模型预测
def predict(X, W, b):
    return 1 / (1 + np.exp(-(X.dot(W) + b)))

# 训练模型
for epoch in range(100):
    # 前向传播
    y_pred = predict(X_train, W, b)

    # 计算损失
    loss = log_loss(y_train, y_pred)
    losses.append(loss)

    # 计算梯度
    error = y_pred - y_train
    grad_W = X_train.T.dot(error) / len(X_train)
    grad_b = np.mean(error)

    gradients_W.append(np.linalg.norm(grad_W))
    gradients_b.append(np.abs(grad_b))

    # Adagrad 更新参数
    W_cache += grad_W**2
    b_cache += grad_b**2

    W -= learning_rate * grad_W / (np.sqrt(W_cache) + epsilon)
    b -= learning_rate * grad_b / (np.sqrt(b_cache) + epsilon)

    # 输出训练过程信息
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 绘制图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制损失函数变化
ax1.plot(losses, 'o-', color='red', label='Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', color='red', fontsize=14)
ax1.tick_params(axis='y', labelcolor='red')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制梯度变化
ax2.plot(gradients_W, 's-', color='blue', label='Gradient Norm (W)', linewidth=2)
ax2.plot(gradients_b, 'd-', color='green', label='Gradient (b)', linewidth=2)
ax2.set_ylabel('Gradient', color='blue', fontsize=14)
ax2.tick_params(axis='y', labelcolor='blue')

# 添加图例
fig.legend(loc='upper right', bbox_to_anchor=(1, 0.85), bbox_transform=ax1.transAxes)

# 设置标题
plt.title('Training with Adagrad: Loss and Gradient Norms', fontsize=16)
plt.show()


#%%>>>>>>>>>>>>>> 5. RMSprop
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
np.random.seed(42)

# 创建一个简单的损失函数，假设它有两个参数 w1 和 w2
def loss_function(w):
    return (w[0] - 3)**2 + (w[1] + 4)**2 + np.sin(w[0]*5) + np.cos(w[1]*5)

# 计算损失函数的梯度
def grad_loss_function(w):
    dw1 = 2 * (w[0] - 3) + 5 * np.cos(w[0]*5)
    dw2 = 2 * (w[1] + 4) - 5 * np.sin(w[1]*5)
    return np.array([dw1, dw2])

# RMSprop优化算法的实现
def rmsprop_optimizer(lr, beta, epsilon, num_iterations):
    w = np.random.randn(2)  # 随机初始化参数 w1 和 w2
    s = np.zeros_like(w)    # 初始化s为零向量
    loss_history = [loss_function(w)]  # 记录初始损失
    w_history = [w.copy()]

    for i in range(num_iterations):
        grad = grad_loss_function(w)
        s = beta * s + (1 - beta) * grad**2
        w -= lr * grad / (np.sqrt(s) + epsilon)

        # 记录损失值和参数更新轨迹
        loss = loss_function(w)
        loss_history.append(loss)
        w_history.append(w.copy())

    return np.array(w_history), loss_history

# 参数设置
learning_rate = 0.01
beta = 0.9
epsilon = 1e-8
num_iterations = 200

# 执行优化算法
w_history, loss_history = rmsprop_optimizer(learning_rate, beta, epsilon, num_iterations)

# 创建画布和子图
fig = plt.figure(figsize=(12, 6))

# 子图1：损失函数的收敛过程
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(loss_history, color='red', linewidth=2, label='Loss')
ax1.set_title('Loss Function Convergence', fontsize=14)
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True)

# 子图2：参数更新轨迹 (w1, w2)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
w1_vals = np.linspace(-4, 4, 100)
w2_vals = np.linspace(-8, 8, 100)
w1_grid, w2_grid = np.meshgrid(w1_vals, w2_vals)
loss_grid = loss_function([w1_grid, w2_grid])

ax2.plot_surface(w1_grid, w2_grid, loss_grid, cmap='viridis', alpha=0.6)
ax2.plot(w_history[:, 0], w_history[:, 1], loss_history, color='orange', marker='o', label='Optimization Path')
ax2.set_title('Parameter Update Path', fontsize=14)
ax2.set_xlabel('w1', fontsize=12)
ax2.set_ylabel('w2', fontsize=12)
ax2.set_zlabel('Loss', fontsize=12)
ax2.view_init(elev=30, azim=120)
ax2.legend()

# 显示图像
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 6. Adadelta
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 添加偏置项
X_b = np.c_[np.ones((1000, 1)), X]

# 初始化参数
theta = np.random.randn(2, 1)

# Adadelta 超参数
rho = 0.95
epsilon = 1e-6

# 初始化累积梯度平方和累积参数更新平方
Eg = np.zeros((2, 1))
Edelta = np.zeros((2, 1))

# 学习率和迭代次数
n_iterations = 50

# 存储损失和参数值
loss_history = []
theta_history = []

# 损失函数（均方误差）
def compute_loss(X_b, y, theta):
    m = len(y)
    y_pred = X_b.dot(theta)
    loss = (1/2*m) * np.sum(np.square(y_pred - y))
    return loss

# 训练模型
for iteration in range(n_iterations):
    gradients = 2/len(X_b) * X_b.T.dot(X_b.dot(theta) - y)

    # 累积梯度平方和
    Eg = rho * Eg + (1 - rho) * gradients**2

    # 计算更新步长
    delta_theta = -np.sqrt(Edelta + epsilon) / np.sqrt(Eg + epsilon) * gradients

    # 更新参数
    theta += delta_theta

    # 累积参数更新平方
    Edelta = rho * Edelta + (1 - rho) * delta_theta**2

    # 记录损失和参数值
    loss_history.append(compute_loss(X_b, y, theta))
    theta_history.append(theta.copy())

# 转换为numpy数组方便绘图
theta_history = np.array(theta_history).squeeze()

# 绘制损失和权重更新图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制损失随迭代次数的变化
ax1.plot(loss_history, 'r-', linewidth=2, label="Loss")
ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.tick_params(axis='y', labelcolor='r')

# 创建第二个坐标轴用于绘制权重
ax2 = ax1.twinx()
ax2.plot(theta_history[:, 0], 'b--', linewidth=2, label="Bias (theta_0)")
ax2.plot(theta_history[:, 1], 'g--', linewidth=2, label="Weight (theta_1)")
ax2.set_ylabel('Theta Values', fontsize=14)
ax2.tick_params(axis='y', labelcolor='b')

# 设置图例
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

plt.title("Adadelta Optimization: Loss and Theta Updates", fontsize=16)
plt.show()




#%%>>>>>>>>>>>>>> 7. Adam
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = np.linspace(-10, 10, 200)
Y = X**2 + 10*np.sin(X) + np.random.normal(0, 10, X.shape)

# 定义目标函数及其梯度
def objective_function(w):
    return w[0]**2 + w[1]**2 + 10*np.sin(w[0]) + 10*np.sin(w[1])

def gradient(w):
    return np.array([2*w[0] + 10*np.cos(w[0]), 2*w[1] + 10*np.cos(w[1])])

# Adam优化算法参数
alpha = 0.1  # 学习率
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
max_iter = 100

# 初始化
w = np.array([8.0, 8.0])  # 初始点
m = np.zeros(2)
v = np.zeros(2)
losses = []
positions = [w.copy()]

# Adam优化
for t in range(1, max_iter + 1):
    g = gradient(w)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    positions.append(w.copy())
    losses.append(objective_function(w))

# 结果可视化
positions = np.array(positions)
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X1, Y1 = np.meshgrid(x_range, y_range)
Z = objective_function([X1, Y1])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：损失函数随迭代次数的变化
ax[0].plot(losses, color='red', linewidth=2)
ax[0].set_title('Loss vs Iterations', fontsize=14)
ax[0].set_xlabel('Iterations', fontsize=12)
ax[0].set_ylabel('Loss', fontsize=12)
ax[0].grid(True)

# 子图2：优化路径在二维空间中的轨迹
cp = ax[1].contourf(X1, Y1, Z, levels=50, cmap='viridis')
ax[1].plot(positions[:, 0], positions[:, 1], marker='o', color='yellow', markerfacecolor='red', linewidth=2, markersize=5)
ax[1].set_title('Optimization Path in 2D Space', fontsize=14)
ax[1].set_xlabel('w1', fontsize=12)
ax[1].set_ylabel('w2', fontsize=12)
ax[1].grid(True)
fig.colorbar(cp, ax=ax[1])

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 8. Nesterov加速梯度

import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def f(x, y):
    return 0.5 * (x ** 2 + y ** 2)

# 目标函数的梯度
def grad_f(x, y):
    return np.array([x, y])

# Nesterov加速梯度算法
def nesterov_accelerated_gradient(x_init, y_init, learning_rate, gamma, iterations):
    x, y = x_init, y_init
    v_x, v_y = 0, 0  # 初始化速度为0

    path = [(x, y)]
    gradients = []
    learning_rates = []

    for i in range(iterations):
        x_nesterov = x - gamma * v_x
        y_nesterov = y - gamma * v_y

        grad_x, grad_y = grad_f(x_nesterov, y_nesterov)
        gradients.append((grad_x, grad_y))

        v_x = gamma * v_x + learning_rate * grad_x
        v_y = gamma * v_y + learning_rate * grad_y

        x = x - v_x
        y = y - v_y

        path.append((x, y))
        learning_rates.append(learning_rate)

    return path, gradients, learning_rates

# 参数设置
x_init, y_init = 5, 5  # 初始点
learning_rate = 0.1  # 学习率
gamma = 0.9  # 动量项
iterations = 50  # 迭代次数

# 运行NAG算法
path, gradients, learning_rates = nesterov_accelerated_gradient(x_init, y_init, learning_rate, gamma, iterations)

# 提取路径点和梯度
path_x, path_y = zip(*path)
grad_x, grad_y = zip(*gradients)

# 创建画布
fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# 绘制目标函数等高线图及优化路径
X, Y = np.meshgrid(np.linspace(-6, 6, 400), np.linspace(-6, 6, 400))
Z = f(X, Y)

ax[0, 0].contour(X, Y, Z, levels=50, cmap='coolwarm')
ax[0, 0].plot(path_x, path_y, 'o-', color='lime', label='Optimization Path')
ax[0, 0].set_title('Optimization Path on Contour Plot')
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y')
ax[0, 0].legend()

# 绘制梯度变化图
ax[0, 1].quiver(path_x[:-1], path_y[:-1], grad_x, grad_y, scale=25, color='red')
ax[0, 1].plot(path_x, path_y, 'o-', color='blue', label='Path')
ax[0, 1].set_title('Gradient Changes')
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y')
ax[0, 1].legend()

# 绘制路径中的x和y随迭代次数的变化图
ax[1, 0].plot(range(iterations + 1), path_x, 'o-', color='blue', label='x')
ax[1, 0].plot(range(iterations + 1), path_y, 'o-', color='orange', label='y')
ax[1, 0].set_title('x and y Values over Iterations')
ax[1, 0].set_xlabel('Iteration')
ax[1, 0].set_ylabel('Value')
ax[1, 0].legend()

# 绘制学习率随迭代次数的变化图
ax[1, 1].plot(range(iterations), learning_rates, 'o-', color='purple', label='Learning Rate')
ax[1, 1].set_title('Learning Rate over Iterations')
ax[1, 1].set_xlabel('Iteration')
ax[1, 1].set_ylabel('Learning Rate')
ax[1, 1].legend()

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>> 9. L-BFGS
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 定义目标函数（一个二次函数）
def objective_function(x):
    return 0.5 * (x[0]**2 + 3 * x[1]**2)

# 定义目标函数的梯度
def objective_gradient(x):
    return np.array([x[0], 3 * x[1]])

# 生成目标函数的等高线图数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + 3 * Y**2)

# 初始点
initial_point = np.array([2.0, 2.0])

# 使用L-BFGS算法进行优化
result = minimize(objective_function, initial_point, method='L-BFGS-B', jac=objective_gradient, options={'disp': True})

# 获取优化过程中记录的点
trajectory = result.x_iters if hasattr(result, 'x_iters') else [initial_point, result.x]

# 绘制图形
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# 1. 损失函数值的变化图
losses = [objective_function(p) for p in trajectory]
ax[0].plot(range(len(losses)), losses, marker='o', color='blue', label='Loss')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss vs. Iteration')
ax[0].grid(True)
ax[0].legend()

# 2. 参数值的变化图
trajectory = np.array(trajectory)
ax[1].plot(trajectory[:, 0], marker='o', color='red', label='x1 (param 1)')
ax[1].plot(trajectory[:, 1], marker='o', color='green', label='x2 (param 2)')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Parameter Values vs. Iteration')
ax[1].grid(True)
ax[1].legend()

# 3. 目标函数的等高线图和优化轨迹
ax[2] = plt.axes([0.7, 0.1, 0.25, 0.8])  # 调整axes位置，以便在右侧绘制
ax[2].contour(X, Y, Z, levels=30, cmap='coolwarm')
ax[2].plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='black', label='Optimization Path')
ax[2].plot(initial_point[0], initial_point[1], 'ro', label='Start')
ax[2].plot(result.x[0], result.x[1], 'go', label='End')
ax[2].set_xlabel('x1')
ax[2].set_ylabel('x2')
ax[2].set_title('Optimization Path on Contour Plot')
ax[2].legend()

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 10. AMSGrad

import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 线性回归模型
def predict(X, theta):
    return X.dot(theta)

# AMSGrad优化算法
def amsgrad(X, y, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, n_iterations=1000):
    m, n = X.shape
    theta = np.random.randn(n, 1)
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    v_hat = np.zeros_like(theta)
    losses = []
    thetas = []

    for iteration in range(n_iterations):
        gradients = -2/m * X.T.dot(y - predict(X, theta))
        m_t = beta1 * m_t + (1 - beta1) * gradients
        v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)
        v_hat = np.maximum(v_hat, v_t)
        theta -= learning_rate * m_t / (np.sqrt(v_hat) + epsilon)

        # Compute and store the loss
        loss = np.mean((y - predict(X, theta)) ** 2)
        losses.append(loss)
        thetas.append(theta.flatten())

    return theta, losses, np.array(thetas)

# 添加偏置项
X_b = np.c_[np.ones((100, 1)), X]
theta_best, losses, thetas = amsgrad(X_b, y, learning_rate=0.01, n_iterations=200)

# 绘图
plt.figure(figsize=(12, 6))

# 绘制损失函数的变化情况
plt.subplot(1, 2, 1)
plt.plot(losses, color='red')
plt.title('Loss Function')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)

# 绘制参数的变化情况
plt.subplot(1, 2, 2)
plt.plot(thetas[:, 0], label='Intercept', color='blue')
plt.plot(thetas[:, 1], label='Slope', color='green')
plt.title('Parameter Changes')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()













