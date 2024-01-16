#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:04:16 2024

@author: jack
"""




import numpy as np
import cvxpy as cp
import scipy as scipy
import cvxopt as cvxopt
import matplotlib
# matplotlib.get_backend()
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


import cvxpy as cp
import numpy as np


#%% 生成问题数据
# 确保随机问题数据是可重现的
np.random.seed(0)

# 生成随机数据矩阵 A
m = 10
n = 10
k = 5
A = np.random.rand(m, k).dot(np.random.rand(k, n))

# 随机初始化 Y
Y_init = np.random.rand(m, k)



#%%% 执行交替最小化
# 确保初始的随机 Y 相同，而不是在每次执行这个单元格时生成新的随机 Y
Y = Y_init

# 执行交替最小化
MAX_ITERS = 30
residual = np.zeros(MAX_ITERS)
for iter_num in range(1, 1+MAX_ITERS):
    # 在迭代开始时，X 和 Y 是 NumPy 数组类型，而不是 CVXPY 变量

    # 对于奇数迭代，保持 Y 不变，优化 X
    if iter_num % 2 == 1:
        X = cp.Variable(shape=(k, n))
        constraint = [X >= 0]
    # 对于偶数迭代，保持 X 不变，优化 Y
    else:
        Y = cp.Variable(shape=(m, k))
        constraint = [Y >= 0]

    # 解决问题
    # 增加最大迭代次数，否则有少数迭代是 "OPTIMAL_INACCURATE"
    # （例如，X 或 Y 中的少数条目在标准公差之外为负数）
    obj = cp.Minimize(cp.norm(A - Y@X, 'fro'))
    prob = cp.Problem(obj, constraint)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise Exception("求解器未收敛！")

    print('迭代 {}，残差范数 {}'.format(iter_num, prob.value))
    residual[iter_num-1] = prob.value

    # 将变量转为 NumPy 数组常量供下一次迭代使用
    if iter_num % 2 == 1:
        X = X.value
    else:
        Y = Y.value


#%% 绘制残差图。

fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)
# 设置绘图属性。


# 创建绘图。
axs.plot(residual)
axs.set_xlabel('迭代次数')
axs.set_ylabel('残差范数')
axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,   )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'NonnegativeMatrixFactorization.eps', format='eps',  )
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

#
# 打印结果。
#
print('原始矩阵：')
print(A)
print('左因子 Y：')
print(Y)
print('右因子 X：')
print(X)
print('残差 A - Y * X：')
print(A - Y.dot(X))
print('经过 {} 次迭代的残差：{}'.format(iter_num, prob.value))

















































































































































































































































































































































































































































































