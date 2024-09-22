#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:56:56 2024

@author: jack
"""


# #%%>>>>>>>>>>>>>> 4. Adagrad
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import log_loss

# # 生成一个虚拟的二元分类数据集
# X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 标准化特征
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 初始化模型参数
# np.random.seed(42)
# W = np.random.randn(2)
# b = 0.0

# # 定义Adagrad参数
# learning_rate = 0.1
# epsilon = 1e-8
# W_cache = np.zeros_like(W)
# b_cache = 0.0

# # 记录数据
# losses = []
# gradients_W = []
# gradients_b = []

# # 模型预测
# def predict(X, W, b):
#     return 1 / (1 + np.exp(-(X.dot(W) + b)))

# # 训练模型
# for epoch in range(100):
#     # 前向传播
#     y_pred = predict(X_train, W, b)

#     # 计算损失
#     loss = log_loss(y_train, y_pred)
#     losses.append(loss)

#     # 计算梯度
#     error = y_pred - y_train
#     grad_W = X_train.T.dot(error) / len(X_train)
#     grad_b = np.mean(error)

#     gradients_W.append(np.linalg.norm(grad_W))
#     gradients_b.append(np.abs(grad_b))

#     # Adagrad 更新参数
#     W_cache += grad_W**2
#     b_cache += grad_b**2

#     W -= learning_rate * grad_W / (np.sqrt(W_cache) + epsilon)
#     b -= learning_rate * grad_b / (np.sqrt(b_cache) + epsilon)

#     # 输出训练过程信息
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss:.4f}')

# # 绘制图形
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # 绘制损失函数变化
# ax1.plot(losses, 'o-', color='red', label='Loss', linewidth=2)
# ax1.set_xlabel('Epoch', fontsize=14)
# ax1.set_ylabel('Loss', color='red', fontsize=14)
# ax1.tick_params(axis='y', labelcolor='red')

# # 创建第二个y轴
# ax2 = ax1.twinx()

# # 绘制梯度变化
# ax2.plot(gradients_W, 's-', color='blue', label='Gradient Norm (W)', linewidth=2)
# ax2.plot(gradients_b, 'd-', color='green', label='Gradient (b)', linewidth=2)
# ax2.set_ylabel('Gradient', color='blue', fontsize=14)
# ax2.tick_params(axis='y', labelcolor='blue')

# # 添加图例
# fig.legend(loc='upper right', bbox_to_anchor=(1, 0.85), bbox_transform=ax1.transAxes)

# # 设置标题
# plt.title('Training with Adagrad: Loss and Gradient Norms', fontsize=16)
# plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 4. AdaGrad (Adaptive Gradient Algorithm), 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

def Adagrad(X, y, theta, lr = 0.01, num_iters = 100, epsilon=1e-8):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}
    G = np.zeros_like(theta)

    for i in range(num_iters):
        gradient = 2/m * X.T @ (X@theta  - y)
        G += gradient ** 2
        theta = theta - lr / (np.sqrt(G + epsilon)) * gradient
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 1000
epsilon = 1e-8

# 运行梯度下降算法
theta_path, history = Adagrad(X_b, y, theta_initial, lr = lr, num_iters = n_iterations, epsilon = epsilon)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 50):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b@theta_path[i], color='red', alpha=0.5)
axs[0].plot(X, X_b.dot(theta_path[-1]), color='green', linewidth=4, label='Final Model')
axs[0].set_title('Linear Regression with Gradient Descent')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(n_iterations), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>> 4. AdaGrad (Adaptive Gradient Algorithm), 机器学习例子: 多项式拟合

import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

def Adagrad(X, y, theta, lr = 0.01, num_iters = 100, epsilon=1e-8):
    m = len(y)
    # 梯度下降优化
    theta_path = []
    history = {'cost': []}
    G = np.zeros_like(theta)

    for i in range(num_iters):
        gradient = 2/m * X.T @ (X@theta  - y)
        G += gradient ** 2
        theta = theta - lr / (np.sqrt(G + epsilon)) * gradient
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history
# 初始化参数 theta
theta_initial = np.random.randn(X_b.shape[1], 1)
lr = 0.1
n_iterations = 1000
epsilon = 1e-8

# 运行梯度下降算法
theta_path, history = Adagrad(X_b, y, theta_initial, lr = lr, num_iters = n_iterations, epsilon = epsilon)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly@theta_path[-1]

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 50):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='g', linewidth=2,)
axs[0].plot(X_new, y_predict, color='r', linewidth=4, label='Final Model')
axs[0].set_title('Polyomial Regression with Gradient Descent')
axs[0].set_xlabel('X')
axs[0].set_ylabel('y')
axs[0].legend()

# 绘制损失函数的变化
axs[1].plot(range(n_iterations), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Mean Squared Error')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 4. AdaGrad (Adaptive Gradient Algorithm): 解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 定义要最小化的函数（简单的二次函数）
def f(x, y):
    return x** 2 + y** 2

# 定义函数关于 x 和 y 的偏导数
def df_dx(x, y):
    return  2 * x

def df_dy(x, y):
    return  2 * y

# 定义梯度下降算法
def Adagrad_Optimizer(start_x, start_y, lr, num_iterations, epsilon=1e-8, perturbation = 0.1):
    # 初始化参数
    x = start_x
    y = start_y
    history = []
    Gx = 0
    Gy = 0
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad_x = df_dx(x, y) + perturbation * np.random.randn(1)[0]
        grad_y = df_dy(x, y) + perturbation * np.random.randn(1)[0]

        Gx += grad_x**2
        Gy += grad_y**2

        # 更新参数
        x = x - lr / (np.sqrt(Gx + epsilon)) * grad_x
        y = y - lr / (np.sqrt(Gy + epsilon)) * grad_y
        # 保存参数的历史记录
        history.append([x, y, f(x, y)])

    return x, y, f(x, y), np.array(history)

# 定义用于绘制函数的网格
x_range = np.arange(- 10 , 10 , 0.1 )
y_range = np.arange(- 10 , 10 , 0.1 )
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 执行梯度下降并绘制结果
start_x, start_y = 8 , 8
learning_rate = 1
num_iterations = 1000
epsilon=1e-8
pb = 1

x_opt, y_opt, f_opt, history = Adagrad_Optimizer(start_x, start_y, learning_rate, num_iterations, epsilon=1e-8, perturbation = pb)

## 3D
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
ax.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
ax.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font,  )

ax.tick_params(axis='both', direction='in', width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()

## 2D courter
fig, ax = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)
CS = ax.contour(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())

ax.grid(False)
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()






































