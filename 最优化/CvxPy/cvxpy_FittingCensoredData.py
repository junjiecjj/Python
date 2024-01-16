#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:19:24 2024

@author: jack
"""

#=================================================================
#        数据拟合
# https://www.wuzao.com/document/cvxpy/examples/applications/censored_data.html
#=================================================================




import numpy as np
import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

fontpath = "/usr/share/fonts/truetype/windows/"

n = 30 # 变量数量
M = 50 # 截尾观测数量
K = 200 # 总观测数量

np.random.seed(n*M*K)
X = np.random.randn(K*n).reshape(K, n)
c_true = np.random.rand(n)

# 生成 y 变量
y = X@c_true + 0.3*np.sqrt(n)*np.random.randn(K)

# 基于 y 进行排序
order = np.argsort(y)
y_ordered = y[order]
X_ordered = X[order,:]

# 寻找边界
D = (y_ordered[M-1] + y_ordered[M])/2.

# 应用截尾
y_censored = np.concatenate((y_ordered[:M], np.ones(K-M)*D))



# 在 ipython 中内联显示图表。
# %matplotlib inline

def plot_fit(fit, fit_label):
    fig, axs = plt.subplots(1,1, figsize=(10, 6), constrained_layout=True)
    axs.grid()
    axs.plot(y_censored, 'bo', label = '截尾数据')
    axs.plot(y_ordered, 'co', label = '未截尾数据')
    axs.plot(fit, 'ro', label=fit_label)

    font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    legend1 = axs.legend(loc='best',  prop=font3,   borderaxespad=0,)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    # frame1.set_facecolor('none')  # 设置图例legend背景透明


    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
    axs.set_ylabel('y')
    axs.set_xlabel('观测值', fontproperties=font1);


    axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,   )
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels] #刻度值字号

    filepath2 = '/home/jack/snap/'
    out_fig = plt.gcf()
    out_fig.savefig(filepath2+'FittingData.eps', format='eps',  )
    #out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
    plt.show()


#%% 使用lstsq拟合全部数据
c_ols = np.linalg.lstsq(X_ordered, y_censored, rcond=None)[0]
fit_ols = X_ordered@c_ols
# plot_fit(fit_ols, 'OLS fit')

#%% 使用lstsq拟合截断数据
c_ols_uncensored = np.linalg.lstsq(X_ordered[:M], y_censored[:M], rcond=None)[0]
fit_ols_uncensored = X_ordered.dot(c_ols_uncensored)
plot_fit(fit_ols_uncensored, 'OLS fit with uncensored data only')

bad_predictions = (fit_ols_uncensored<=D) & (np.arange(K)>=M)
# plt.plot(np.arange(K)[bad_predictions], fit_ols_uncensored[bad_predictions], color='orange', marker='o', lw=0);


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 使用CVX拟合
import cvxpy as cp
X_uncensored = X_ordered[:M, :]
c = cp.Variable(shape=n)
objective = cp.Minimize(cp.sum_squares(X_uncensored@c - y_ordered[:M]))
constraints = [ X_ordered[M:,:]@c >= D]
prob = cp.Problem(objective, constraints)
result = prob.solve()

c_cvx = np.array(c.value).flatten()
fit_cvx = X_ordered.dot(c_cvx)
plot_fit(fit_cvx, 'CVX fit')























































































































































































































































































