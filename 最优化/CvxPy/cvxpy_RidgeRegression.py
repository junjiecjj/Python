#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:16:47 2024

@author: jack
"""


#=================================================================
#                   机器学习：岭回归
# https://www.wuzao.com/document/cvxpy/examples/machine_learning/ridge_regression.html
#=================================================================


import numpy as np
import cvxpy as cp
import scipy as scipy
# import cvxopt as cvxopt
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

fontpath = "/usr/share/fonts/truetype/windows/"
plotconfig = {
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
    "font.size": 18,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(plotconfig)


# 修复随机数生成器以便重复实验。
np.random.seed(0)

#%% 编写目标函数
def loss_fn(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


#%% 生成数据
def generate_data(m=100, n=20, sigma=5):
    "生成数据矩阵 X 和观测值 Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    # 生成一个病态数据矩阵
    X = np.random.randn(m, n)
    # 用加性高斯噪声破坏观测值
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y

m = 100
n = 20
sigma = 5

X, Y = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]


#%% 拟合模型
beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)


#%% 评估模型
def plot_train_test_errors(train_errors, test_errors, lambd_values):
    fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)
    axs.plot(lambd_values, train_errors, label= "Train error")
    axs.plot(lambd_values, test_errors, label="Test error")
    axs.set_xscale("log")
    # font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    axs.legend(loc="upper left", )
    axs.set_xlabel(r"$\lambda$", fontsize=16)
    font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    axs.set_title(" (MSE)", fontproperties=font3 )
    filepath2 = '/home/jack/snap/'
    out_fig = plt.gcf()
    out_fig.savefig(filepath2+'ridge_regression.eps', format='eps',  )
    #out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
    plt.show()
    return

plot_train_test_errors(train_errors, test_errors, lambd_values)

#%% 正则化路径
def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])

    # font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    # axs.legend(loc="upper left", )
    axs.set_xscale("log")
    axs.set_xlabel(r"$\lambda$", fontsize=16)
    # font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    axs.set_title("Regularization Path",  )
    filepath2 = '/home/jack/snap/'
    out_fig = plt.gcf()
    out_fig.savefig(filepath2+'ridge_regressionRegularization.eps', format='eps',  )
    #out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
    plt.show()

plot_regularization_path(lambd_values, beta_values)

















































































































































































































































































