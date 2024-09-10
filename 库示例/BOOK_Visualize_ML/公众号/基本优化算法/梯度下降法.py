#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:43:44 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486322&idx=1&sn=0c7a0e04f27d8c34b9fa28dcd1f1d5d4&chksm=c1284f8adafd826a04c570837cdc267565615d38a576f27862bddfed44b571ddcfbfee0447c0&mpshare=1&scene=1&srcid=0904JUaxxqY43w4AR2eyaioW&sharer_shareinfo=8335e098352e1123ce0c27f75d093ca1&sharer_shareinfo_first=8335e098352e1123ce0c27f75d093ca1&exportkey=n_ChQIAhIQ05N8QvneS%2FPtiPNNRI2LBRKfAgIE97dBBAEAAAAAAGvvFtuH1YwAAAAOpnltbLcz9gKNyK89dVj01HntTJPdSNSf4yO8kEvdCckdRqvgNJT2sQkSdFND9G8peW2wVvqhMbWOkvIFNCyPRjeKRVyP0Eqj8jNoBhJVONWEYfz6HEnvf3%2Bep96WC5ymYiRDDDmDpNMWMLsZNTSgKzjq7Epl5Hj0i0HZcIkkf5CTycLhCJjLkYG%2FQw7Q52vnTQv71aO8P%2F3u8Dn9m%2FuZ0v7TS%2FfCeFlzi9z7q8%2FhjvegFE7JLT%2FO7Zz%2BadhKbTW0YlHGTAJWjTRXJnnEw%2FLLRTi7kp0Q%2F%2BUxoHDIzh6KEeivSogxQwLf%2FrQ9wviCxWJ3aUSeLCWKhLfiAZWCzC6ckuRrJOJoAW8E&acctmode=0&pass_ticket=v1hukaDWnis4NcTZf75tLOZDI1xdg%2BuUtdCQhMzx5yel%2B%2B%2FVMkpmnMZjCZa9ZKdz&wx_header=0#rd

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484520&idx=1&sn=c2c25b442734b44ddfa412a1129f8dca&chksm=c1fa6683ca294107ef4816e6add6a6422cc0bbb593e0c64b5c64adecafd09abe82e05939450a&mpshare=1&scene=1&srcid=0812uv9NtQrWpJQ6prUy2C6J&sharer_shareinfo=39a4c9769f55194efb3a702d8cb1281d&sharer_shareinfo_first=6bc86de1b2f94b09b94707b820ffe899&exportkey=n_ChQIAhIQKrh0y7hnY1su%2FG0qKUy7YRKfAgIE97dBBAEAAAAAAJlBAfWZUzkAAAAOpnltbLcz9gKNyK89dVj0ARjnCEJReDN%2BekOxv9jldjPmruLo8eMl4ISbYnjrBD54DjpPj6pPM3eufDEONXWO7dWBQlqEgZgul99Yh7vsovX3APoqj1gQLYjVSO0%2FXHbvbA71b96Igkv87KoHI%2BtG3oEvjoM2NH0wpSTHBHmpJsjin%2FJp1vP0DfGhip%2FV52b%2BkLxyiHQdQiQTflKAWV%2F9%2FL3WeOm8etKlUzFwOdZZ2Q51LKjUfBFxPPI86cZwDrqK1OLwr6lpfg3JV5mhOQA8%2BxtTTm8TU79AcnoivDARJwvs1DflypQd04%2BsFaVrgVmTDEJndKDXwLf3sbPgebp1sOnNYlON8Pk5&acctmode=0&pass_ticket=3qZUWMNvP47Z0UY45NrWgr2ouMyo0YE60mjzpevzs4uA7ojRXiuA04r%2FlCWjD%2B%2Fh&wx_header=0#rd

https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247489994&idx=1&sn=547a867910dc6c716d6e1aa8a1b23fce&chksm=9a988d08c1e7c8acd535440b3178ff281a0e223336bbc1926f1cd08f537b5d0562c77b78907a&mpshare=1&scene=1&srcid=0826nROhswSKldKC5YfgrOHj&sharer_shareinfo=6e00c2c84c476594917af616de276677&sharer_shareinfo_first=fde095ab92bc0b5b8dfd10cad35ae1e0&exportkey=n_ChQIAhIQhD1MbJr4TUVl1iutH7BWBhKfAgIE97dBBAEAAAAAAH1wMF8tsMwAAAAOpnltbLcz9gKNyK89dVj04%2Bt6arXRlSMq3WCk4%2BGktvOewv7bVvGgpfKK%2B%2BPlgSzymm7LOhz%2BTqAY%2Bawun9euEwjnfNTS7tc5HyTxeIJpV0b%2F5bk5RB7xasNJGcczu%2F3E3qoYmCBHQ5uq6wQwU0Yz42cNbRw9siua6pqSZvy12LgqnPqOwJuGkPlhK8nHHIIyMFN7f0XGx8ww6I0OhuQkN3H8zynSy3gHA02hhJHxXDnvI6VveDAze4BWMdmofzxVPvfTgAJcVZGNrNWum%2FmpZBtodxy7AWr9Ak91bJIYYrmDRSZ%2F0ntDrxx9zDY5vezjeeUfExw0bvQUpihq1Br2A7Nt5AxA2nJg&acctmode=0&pass_ticket=c9NbbXXfvZRQG300JwNSe9uwByZMrHribm56HhM9obSIJ4U3Jlu5BGp36iqCWGpX&wx_header=0#rd

https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247488869&idx=1&sn=da676c439e96fb942ae75ecdc82411b4&chksm=9af2fbc8649d130fa4466688cab5f9259fc952e96aa4d097abbf2c7b68887b58499f2a5ed947&mpshare=1&scene=1&srcid=0812Tr5fId9u8wgQbbX3SiXK&sharer_shareinfo=5fc9ebd18e24a29a580b58a88a243e53&sharer_shareinfo_first=798d0355173f43ae25290b16a6097c5a&exportkey=n_ChQIAhIQh%2BNQamhBm8zWMomf98Av6xKfAgIE97dBBAEAAAAAAGd3GeCDP6wAAAAOpnltbLcz9gKNyK89dVj0e0OAo3uYgeT27MWdNp8alBChm%2BOkGgnMSUs07O%2BuwJsv%2B2rX0eOSIzUtq3p05Fsjg%2FmKBSBWlA5wufVPbd6DERc%2F8gmLKUg3l8RjajWPZrBsyA11j8URNQDamW650fEtgFoDEYnwgxxL9OLKG%2Be9%2BDYPgkM03lE5MjcFIpu5yGiYBsir8lbFVapNus7vdWPWdtZTo3bLs9shRfHHWPU%2FBesZcpLTlSLHBQhmGFFsvAkfS43hJTqLVPSnVEDmu0DrzCm2v9VSowgMYbfN6wKTcZplQzpuPtrJ0C9iyEXmhI36GJTD3ZHONfYdMLAG1%2B3ezEIS%2FnL7ZuAb&acctmode=0&pass_ticket=y59RsitGoUwbjjdbrjX0CQusdphpY6lI5ABKS8gij4bCBfKUbQntXGkyPdvLcMNs&wx_header=0#rd

"""


#%%>>>>>>>>>>>>>> 1. 梯度下降法: 机器学习例子: 线性回归
import numpy as np
import matplotlib.pyplot as plt
import os

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

# 初始化参数
theta = np.random.randn(X_b.shape[1], 1)  # 两个参数：截距和斜率
lr = 0.1
n_iterations = 100

# 定义梯度下降函数
def gradient_descent(X, y, theta_init, learning_rate=0.01, iterations=100):
    m = len(y)
    theta = theta_init

    theta_path = []; theta_path.append(theta)
    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    for iteration in range(iterations):
        gradients = 2/m * X.T @ (X@theta  - y)
        theta = theta - learning_rate * gradients
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_path, history = gradient_descent(X_b, y, theta_initial, learning_rate = lr, iterations = n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X, X_b@theta_path[i], color='green', linewidth=2,)
axs[0].plot(X, X_b@theta_path[-1], color='r', linewidth=4, label='Final Model')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs[0].set_title('Linear Regression with Gradient Descent', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)


# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_LineRe.eps', bbox_inches='tight', pad_inches=0,)
plt.show()

#%%>>>>>>>>>>>>>> 1. 梯度下降法: 机器学习例子: 多项式回归
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(1000, 1)
# 增加多项式特征
X_b = np.c_[np.ones((1000, 1)), X, X**2]

# 初始化参数
theta = np.random.randn(3, 1)  # 两个参数：截距和斜率
lr = 0.1
n_iterations = 100

# 定义梯度下降函数
def gradient_descent(X, y, theta_init, learning_rate=0.01, iterations=100):
    m = len(y)
    theta = theta_init
    theta_path = []; theta_path.append(theta)

    history = {'cost': []}
    cost = np.mean((X @ theta  - y) ** 2)
    history['cost'].append(cost)

    for iteration in range(iterations):
        gradients = 2/m * X.T @ (X@theta  - y)
        theta = theta - learning_rate * gradients
        theta_path.append(theta)
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta_path, history

# 运行梯度下降算法
theta_path, history = gradient_descent(X_b, y, theta, learning_rate = lr, iterations = n_iterations)

# 将路径转换为数组
theta_path = np.array(theta_path)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 生成用于画图的曲线
X_new = np.linspace(0, 2, 100).reshape(100, 1)
X_new_poly = np.c_[np.ones((100, 1)), X_new, X_new**2]
y_predict = X_new_poly@theta_path[-1]

# 绘制数据集及拟合的直线
axs[0].scatter(X, y, color='blue', label='Data')
for i in range(0, n_iterations, 10):  # 每10次迭代绘制一次直线
    axs[0].plot(X_new, X_new_poly@theta_path[i], color='g', linewidth=2,)
axs[0].plot(X_new, y_predict, color='r', linewidth=4, label='Final Model')
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs[0].set_title('Polyomial Regression with Gradient Descent', fontdict = font )
axs[0].set_xlabel('x', fontdict = font, labelpad = 2)
axs[0].set_ylabel('y', fontdict = font, labelpad = 2)

font = {'weight' : 'normal', 'size': 20,}
axs[0].legend(loc='best', prop =  font )

axs[0].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)

# 绘制损失函数的变化
axs[1].plot(range(len(history['cost'])), history['cost'], color='red', linewidth=2, label='Loss Function')
axs[1].set_title('Loss Function Over Iterations', fontdict = font )
axs[1].set_xlabel('Iteration', fontdict = font, labelpad = 2)
axs[1].set_ylabel('Mean Squared Error', fontdict = font, labelpad = 2)
font = {'weight' : 'normal', 'size': 20,}
axs[1].legend(loc='best', prop =  font )
axs[1].tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)


# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_Polyomial.eps', bbox_inches='tight', pad_inches=0,)
plt.show()


# #%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent): 解析
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# ## 定义要最小化的函数（简单的二次函数）
# def f(x, y):
#     return x**2 / 4 + y**2
# ## 定义函数关于 x 和 y 的偏导数
# def df_dx(x, y):
#     return  x/2
# def df_dy(x, y):
#     return  2*y

# # 定义梯度下降算法
# def Gradient_descent(start_x, start_y, lr, num_iterations, perturbation = 0.1):
#     # 初始化参数
#     x = start_x
#     y = start_y
#     history = []
#     history.append([x, y, f(x, y)])
#     # 执行梯度下降迭代
#     for i in  range (num_iterations):
#         # 计算梯度
#         grad_x = df_dx(x, y) + perturbation * np.random.randn(1)[0]
#         grad_y = df_dy(x, y) + perturbation * np.random.randn(1)[0]
#         # 更新参数
#         x = x - lr * grad_x
#         y = y - lr * grad_y
#         # 保存参数的历史记录
#         history.append([x, y, f(x, y)])
#     return x, y, f(x, y), np.array(history)

# pb = 0
# # 定义用于绘制函数的网格
# x_range = np.arange(- 10 , 10 , 0.1 )
# y_range = np.arange(- 10 , 10 , 0.1 )
# X, Y = np.meshgrid(x_range, y_range)
# Z = f(X, Y)

# # 执行梯度下降并绘制结果
# start_x, start_y = 8 , 8
# learning_rate = 0.1
# num_iterations = 100
# x_opt, y_opt, f_opt, history = Gradient_descent(start_x, start_y, learning_rate, num_iterations, perturbation = pb)

# ## 3D
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
# ax.set_proj_type('ortho')

# norm_plt = plt.Normalize(Z.min(), Z.max())
# colors = cm.RdYlBu_r(norm_plt(Z))
# ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

# ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
# ax.set_proj_type('ortho')

# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
# ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font,  )

# ax.tick_params(axis='both', direction='in', width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# # ax.view_init(azim=-120, elev=30)
# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()

# ## 2D courter
# fig, ax = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# # CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# # fig.colorbar(CS)
# CS = ax.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

# ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

# ax.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=15, labelfontfamily = 'Times New Roman', pad = 1)

# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())

# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()


# #%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent): 解析
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# ## 定义要最小化的函数（简单的二次函数）
# def f(w):
#     return w[0] ** 2/4 + w[1] ** 2
# ## 定义函数关于 x 和 y 的偏导数
# def grad_f(w):
#     return np.array([w[0]/2.0, 2*w[1]])

# ## 定义梯度下降算法
# def Gradient_descent(theta_init, lr, num_iterations, perturbation = 0.1):
#     # 初始化参数
#     theta = theta_init
#     history = []
#     history.append([theta[0], theta[1], f(theta)])
#     # 执行梯度下降迭代
#     for i in  range (num_iterations):
#         # 计算梯度
#         grad = grad_f(theta) + perturbation * np.random.randn(*theta.shape)
#         # 更新参数
#         theta = theta - lr * grad
#         # 保存参数的历史记录
#         history.append([theta[0], theta[1], f(theta)])
#     return theta[0], theta[1], f(theta), np.array(history)

# # 定义用于绘制函数的网格
# x_range = np.arange(-10 ,10 , 0.1 )
# y_range = np.arange(-10 ,10 , 0.1 )
# X, Y = np.meshgrid(x_range, y_range)
# W_array = np.vstack([X.ravel(), Y.ravel()])
# Z = f(W_array).reshape(X.shape)

# # 执行梯度下降并绘制结果
# theta_init = np.array([8, 8])
# learning_rate = 0.1
# num_iterations = 100
# pb = 2

# x_opt, y_opt, f_opt, history = Gradient_descent(theta_init, learning_rate, num_iterations, perturbation = pb)

# ## 3D
# fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
# axs.set_proj_type('ortho')

# norm_plt = plt.Normalize(Z.min(), Z.max())
# colors = cm.RdYlBu_r(norm_plt(Z))
# axs.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

# axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
# axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
# axs.set_proj_type('ortho')

# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
# axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
# axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
# axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
# axs.set_title(f"Gradient descent, pb = {pb}", fontdict = font, )

# axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
# axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# # ax1.view_init(azim=-100, elev=20)

# # 设置坐标轴线宽
# axs.xaxis.line.set_lw(1)
# axs.yaxis.line.set_lw(1)
# axs.zaxis.line.set_lw(1)
# axs.grid(False)

# # 显示图形
# out_fig = plt.gcf()
# optimizer = "GD"
# savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
# os.makedirs(savedir, exist_ok = True)
# # out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_easyfun.eps', )
# plt.show()


# ## 2D courter
# fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# # CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# # fig.colorbar(CS)
# CS = axs.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

# axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
# axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
# axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
# axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
# axs.set_title(f"Gradient descent, pb = {pb}", fontdict = font,  )

# axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

# axs.set_xlim(X.min(), X.max())
# axs.set_ylim(Y.min(), Y.max())
# axs.grid(False)

# # 显示图形
# out_fig = plt.gcf()
# optimizer = "GD"
# savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
# os.makedirs(savedir, exist_ok = True)
# # out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_easyfun.eps', )
# plt.show()


# #%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent): 解析
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# def f(x, y):
#     return (1-y**5+x**5)*np.exp(-x**2-y**2)

# def df_dx(x, y):
#     return (5 * x**4 * np.exp(-x**2-y**2) - 2*x * (1 - y**5 + x**5) * np.exp(-x**2-y**2))

# def df_dy(x, y):
#     return (-5 * y**4 * np.exp(-x**2-y**2) - 2*y * (1-y**5 + x**5) * np.exp(-x**2-y**2))

# # 定义梯度下降算法
# def Gradient_descent(start_x, start_y, lr, num_iterations, perturbation = 0.1):
#     # 初始化参数
#     x = start_x
#     y = start_y
#     history = []
#     history.append([x, y, f(x, y)])
#     # 执行梯度下降迭代
#     for i in  range (num_iterations):
#         # 计算梯度
#         grad_x = df_dx(x, y) + perturbation * np.random.randn(1)[0]
#         grad_y = df_dy(x, y) + perturbation * np.random.randn(1)[0]
#         # 更新参数
#         x = x - lr * grad_x
#         y = y - lr * grad_y
#         # 保存参数的历史记录
#         history.append([x, y, f(x, y)])
#     return x, y, f(x, y), np.array(history)

# pb = 0.1
# # 定义用于绘制函数的网格
# x_range = np.arange(-2 , 2 , 0.1 )
# y_range = np.arange(-2 , 2 , 0.1 )
# X, Y = np.meshgrid(x_range, y_range)
# Z = f(X, Y)

# # 执行梯度下降并绘制结果
# start_x, start_y = 0 , 0
# learning_rate = 0.1
# num_iterations = 100
# x_opt, y_opt, f_opt, history = Gradient_descent(start_x, start_y, learning_rate, num_iterations, perturbation = pb)

# ## 3D
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
# ax.set_proj_type('ortho')

# norm_plt = plt.Normalize(Z.min(), Z.max())
# colors = cm.RdYlBu_r(norm_plt(Z))
# ax.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

# ax.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
# ax.set_proj_type('ortho')

# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)
# ax.set_zlabel(r'$\it{f}(\it{x_1},\it{x_2}$)', fontdict = font, )

# ax.tick_params(axis = 'both', direction = 'in', width=3, length = 5, labelsize = 15, labelfontfamily = 'Times New Roman', pad = 1)

# # ax.view_init(azim=-120, elev=30)
# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()

# ## 2D courter
# fig, ax = plt.subplots(figsize = (7.5, 6), constrained_layout = True)
# # CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# # fig.colorbar(CS)
# CS = ax.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)

# ax.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
# ax.scatter(history[-1,0], history[-1,1],  c = 'r' , marker = '*', s = 200, zorder = 10)
# font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,} # 'family':'Times New Roman',
# ax.set_xlabel(r'$\it{x_1}$', fontdict = font, labelpad = 2)
# ax.set_ylabel(r'$\it{x_2}$', fontdict = font, labelpad = 2)

# ax.tick_params(axis = 'both', direction='in', left = True, right = True, top = True, bottom = True, width = 3, length = 5,  labelsize = 15, labelfontfamily = 'Times New Roman', pad = 1)

# # ax.set_xlim(X.min(), X.max())
# # ax.set_ylim(Y.min(), Y.max())

# ax.grid(False)
# # fig.savefig('Figures/只保留网格线.svg', format='svg')
# plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 梯度下降法 (Gradient Descent): 解析
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mod = "hard"
if mod == 'easy':
    funkind = 'convex fun'
    ## 定义要最小化的函数（简单的二次函数）
    def f(w):
        return w[0] ** 2/4 + w[1] ** 2
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        return np.array([w[0]/2.0, 2*w[1]])
elif mod == 'hard':
    funkind = 'non-convex fun'
    ## 定义要最小化的函数（简单的二次函数）
    def f(w):
        return (1-w[1]**5 + w[0]**5)*np.exp(-w[0]**2 - w[1]**2)
    ## 定义函数关于 x 和 y 的偏导数
    def grad_f(w):
        gx = (5 * w[0]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[0] * (1 - w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        gy = (-5 * w[1]**4 * np.exp(-w[0]**2-w[1]**2) - 2*w[1] * (1-w[1]**5 + w[0]**5) * np.exp(-w[0]**2-w[1]**2))
        return np.array([gx, gy])

# 定义梯度下降算法
def Gradient_descent(theta_init, lr, num_iterations, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0], theta[1], f(theta)])
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad = grad_f(theta) + perturbation * np.random.randn(*theta.shape)
        # 更新参数
        theta = theta - lr * grad
        # 保存参数的历史记录
        history.append([theta[0], theta[1], f(theta)])

    return theta[0], theta[1], f(theta), np.array(history)

if mod == 'easy':
    # 定义用于绘制函数的网格
    x_range = np.arange(-10 ,10 , 0.1 )
    y_range = np.arange(-10 ,10 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)

    # 执行梯度下降并绘制结果
    theta_init = np.array([8, 8])
elif mod == 'hard':
    # 定义用于绘制函数的网格
    x_range = np.arange(-6 , 6 , 0.1 )
    y_range = np.arange(-6 , 6 , 0.1 )
    X, Y = np.meshgrid(x_range, y_range)
    W_array = np.vstack([X.ravel(), Y.ravel()])
    Z = f(W_array).reshape(X.shape)
    # 执行梯度下降并绘制结果
    theta_init = np.array([0.01, 0.01])

learning_rate = 0.1
num_iterations = 100
pb = 0

x_opt, y_opt, f_opt, history = Gradient_descent(theta_init, learning_rate, num_iterations, perturbation = pb)

## 3D
fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8), constrained_layout=True)
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(X, Y, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
axs.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"GD on {funkind}, pb = {pb}", fontdict = font, )

axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# ax1.view_init(azim=-100, elev=20)

# 设置坐标轴线宽
axs.xaxis.line.set_lw(1)
axs.yaxis.line.set_lw(1)
axs.zaxis.line.set_lw(1)
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_{mod}fun.eps', bbox_inches='tight', pad_inches=0, )
plt.show()


## 2D courter
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
# CS = ax.contourf(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 1)
# fig.colorbar(CS)
CS = axs.contour(X, Y, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"GD on {funkind}, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(X.min(), X.max())
axs.set_ylim(Y.min(), Y.max())
axs.grid(False)

# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_{mod}fun.eps', bbox_inches='tight', pad_inches=0,)
plt.show()


#%%>>>>>>>>>>>>>> 1. 梯度下降法: 线性回归, 解析

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
# matplotlib.use('Agg')

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 (X0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

## 定义要最小化的函数（简单的二次函数）
def f(theta, X_b, y ):
    return np.mean((X_b @ theta  - y) ** 2, axis = 0)
## 定义函数关于 x 和 y 的偏导数
def grad_f(theta, X_b, y ):
    m = len(y)
    return 2/m * X_b.T @ (X_b @ theta  - y)

## 定义梯度下降算法
def Gradient_descent(theta_init, lr, num_iterations, perturbation = 0.1):
    # 初始化参数
    theta = theta_init
    history = []
    history.append([theta[0][0], theta[1][0], f(theta, X_b, y )[0]])
    # 执行梯度下降迭代
    for i in  range (num_iterations):
        # 计算梯度
        grad = grad_f(theta, X_b, y) + perturbation * np.random.randn(*theta.shape)
        # 更新参数
        theta = theta - lr * grad
        # print(f"{i}:{theta}")
        # 保存参数的历史记录
        history.append([theta[0][0], theta[1][0], f(theta, X_b, y)[0]])

    return theta[0][0], theta[1][0], f(theta, X_b, y)[0], np.array(history)

# 定义用于绘制函数的网格
theta1_range = np.arange(-40, 60 , 0.2)
theta2_range = np.arange(-40, 60 , 0.2)
Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)
Theta_array = np.vstack([Theta1.ravel(), Theta2.ravel()])
Z = f(Theta_array, X_b, y).reshape(Theta1.shape)

# 执行梯度下降并绘制结果
theta_init = np.array([39, 59]).reshape(-1,1)
learning_rate = 0.05
num_iterations = 100
pb = 0

x_opt, y_opt, f_opt, history = Gradient_descent(theta_init, learning_rate, num_iterations, perturbation = pb)

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

fig, axs = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8, 8), ) # constrained_layout=True
###### 3D wireframe
# ax1 = fig.add_subplot(111, projection = '3d')
axs.set_proj_type('ortho')

norm_plt = plt.Normalize(Z.min(), Z.max())
colors = cm.RdYlBu_r(norm_plt(Z))
axs.plot_wireframe(Theta1, Theta2, Z, color = [0.6, 0.6, 0.6], linewidth = 0.5) # color = '#0070C0',

axs.plot(history[:,0], history[:,1], history[:,2], c = 'b' , marker= 'o', ms = 5, zorder = 2)
axs.scatter(history[-1,0], history[-1,1], history[-1,2], c = 'r' , marker= '*', s = 40, zorder = 10 )
axs.set_proj_type('ortho')

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 2)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 2)
axs.set_zlabel(r'$\mathrm{f}(\mathrm{\theta_1},\mathrm{\theta_2}$)', fontdict = font, labelpad = 6 )
axs.set_title(f"Gradient descent on data, pb = {pb}", fontdict = font, )

axs.tick_params(axis='both', direction='in', width = 3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.ticklabel_format(style='sci', scilimits=(-1,2), axis='z')
# ax1.view_init(azim=-100, elev=20)

# 设置坐标轴线宽
axs.xaxis.line.set_lw(1)
axs.yaxis.line.set_lw(1)
axs.zaxis.line.set_lw(1)
axs.grid(False)


# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
out_fig.savefig(savedir + f'{optimizer}_pb{pb}_3D_Xy.eps',bbox_inches='tight', pad_inches=0, )
plt.show()

###### 2D contour
fig, axs = plt.subplots(figsize=(7, 6), constrained_layout=True)
CS = axs.contour(Theta1, Theta2, Z, levels = 30, cmap = 'RdYlBu_r', linewidths = 1)
fig.colorbar(CS)

axs.plot(history[:,0], history[:,1],  c = 'b' , marker= 'o', ms = 5 )
axs.scatter(history[-1,0], history[-1,1],  c = 'r' , marker= '*', s = 200, zorder = 10)
font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel(r'$\it{\theta_1}$', fontdict = font, labelpad = 0)
axs.set_ylabel(r'$\it{\theta_2}$', fontdict = font, labelpad = 0)
axs.set_title(f"Gradient descent on data, pb = {pb}", fontdict = font,  )

axs.tick_params(axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)

axs.set_xlim(Theta1.min(), Theta1.max())
axs.set_ylim(Theta2.min(), Theta2.max())
axs.grid(False)


# 显示图形
out_fig = plt.gcf()
optimizer = "GD"
savedir = f'/home/jack/公共的/Figure/optimfigs/{optimizer}/'
os.makedirs(savedir, exist_ok = True)
# out_fig.savefig(savedir + f'{optimizer}_pb{pb}_2D_Xy.eps',bbox_inches='tight', pad_inches=0, )
plt.show()
















































































