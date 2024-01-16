#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:03:05 2024

@author: jack
"""
#=================================================================
#        稀疏协方差估计的高斯变量
# https://www.wuzao.com/document/cvxpy/examples/applications/sparse_covariance_est.html
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
    "font.size": 28,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(plotconfig)


# 修复随机数生成器以便重复实验。
np.random.seed(0)

# 矩阵的维度。
n = 10

# 样本数，y_i
N = 1000

# 创建稀疏、对称PSD矩阵S
A = np.random.randn(n, n)  # 单位正态分布。
A[scipy.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # 稀疏化矩阵。
Strue = A@A.T + 0.05 * np.eye(n)  # 强制严格正定。

# 创建与S相关的协方差矩阵R。
R = np.linalg.inv(Strue)

# 从具有协方差R的分布中创建样本y_i。 sqrtm(A) = A^0.5
y_sample = scipy.linalg.sqrtm(R).dot(np.random.randn(n, N))

# 计算样本的协方差矩阵。
Y = np.cov(y_sample)

# 每个尝试生成稀疏逆协方差矩阵的alpha值。
alphas = [10, 2, 1]

# 空的结果矩阵S列表
Ss = []

# 对每个alpha值解决优化问题。
for alpha in alphas:
    # 创建一个被限制在正半定锥中的变量。
    S = cp.Variable(shape=(n,n), PSD=True)

    # 形成logdet(S) - tr(SY)目标函数。注意使用集合推导来形成S*Y的对角线元素的集合，
    # 并且使用原生的sum函数来计算迹。TODO：如果cvxpy中提供了迹运算符，请使用它！
    obj = cp.Maximize(cp.log_det(S) - sum([(S@Y)[i, i] for i in range(n)]))

    # 设置约束条件。
    constraints = [cp.sum(cp.abs(S)) <= alpha]

    # 形成和解决优化问题
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise Exception('CVXPY错误')

    # 如果需要协方差矩阵R，可以这样创建。
    R_hat = np.linalg.inv(S.value)

    # 将S元素阈值化以确保精确零值：
    S = S.value
    S[abs(S) <= 1e-4] = 0

    # 将此S存储在结果列表中以备后续绘图使用。
    Ss += [S]

    print('以alpha = {} 为参数的优化完成，目标值为 {}'.format(alpha, obj.value))



# 创建图形。
plt.figure()
plt.figure(figsize=(12, 12), constrained_layout=True)

# 绘制真实协方差矩阵的稀疏模式。
plt.subplot(2, 2, 1)
plt.spy(Strue)
plt.title('Inverse of true covariance matrix', fontsize = 16)

# 对于每个特定的 alpha 值，绘制相应的稀疏模式。
for i in range(len(alphas)):
    plt.subplot(2, 2, 2+i)
    plt.spy(Ss[i])
    plt.title("Estimated inv. cov matrix" + r"$\alpha={}$".format(alphas[i]), fontsize = 16)
# r"$\mathrm{{label}}:{} \rightarrow {}$".format(real_lab, real_lab),
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'sparse_covariance_est.eps', format='eps',  )
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()























