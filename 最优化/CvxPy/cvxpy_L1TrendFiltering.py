#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:54:21 2024

@author: jack
"""


#=================================================================
#        L1 趋势滤波
# https://www.wuzao.com/document/cvxpy/examples/applications/l1_trend_filter.html
#=================================================================


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

# 加载时间序列数据：S&P 500 价格的对数。
y = np.loadtxt(open('data/snp500.txt', 'rb'), delimiter=",", skiprows=1)
n = y.size

# 形成二阶差分矩阵。
e = np.ones((1, n))
D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)

# 设置正则化参数。
vlambda = 50

# 解决 l1 趋势过滤问题。
x = cp.Variable(shape=n)
obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                  + vlambda * cp.norm(D*x, 1) )
prob = cp.Problem(obj)

# ECOS 和 SCS 求解器在迭代次数限制之前无法收敛。改用 CVXOPT。
prob.solve(solver=cp.CVXOPT, verbose=True)
print('求解器状态: {}'.format(prob.status))

# 检查错误。
if prob.status != cp.OPTIMAL:
    raise Exception("求解器未收敛！")

print("最优目标值: {}".format(obj.value))




fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)
# 绘制原始信号的估计趋势。

axs.plot(np.arange(1,n+1), y, 'k:', linewidth=1.0)
axs.plot(np.arange(1,n+1), np.array(x.value), 'b-', linewidth=2.0)
axs.set_xlabel('日期')
axs.set_ylabel('对数价格')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'L1TrendFiltering.eps', format='eps',  )
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



























































































































































































































































































































































































