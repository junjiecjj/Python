#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:59:22 2024
@author: jack

# 故障检测
# https://www.wuzao.com/document/cvxpy/examples/applications/fault_detection.html


"""

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

from matplotlib.pyplot import MultipleLocator
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

n = 2000
m = 200
p = 0.01
snr = 5

sigma = np.sqrt(p*n/(snr**2))
A = np.random.randn(m,n)

x_true = (np.random.rand(n) <= p).astype(int)
v = sigma*np.random.randn(m)

y = A@x_true + v


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 下面，我们展示x, AX和噪声v.
fig, axs = plt.subplots(2,1, figsize=(8, 10), constrained_layout=True)

axs[0].plot(range(n),x_true, color='b', linestyle='-', label='x_true',  )

font2  = {'family':'Times New Roman','style':'normal','size':22,  }
axs[0].set_xlabel(r'n',   fontdict = font2, labelpad=12.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。)
axs[0].set_ylabel(r'x',  fontdict = font2)
font2 = {'family':'Times New Roman','style':'normal','size':22,  }
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1].plot(range(m), A@x_true, color='b', linestyle='-', label='Ax',  )
axs[1].plot( range(m), v, color='r', linestyle='-', label='v',  )
font2  = {'family':'Times New Roman','style':'normal','size':22,  }
axs[1].set_xlabel(r'm',   fontdict = font2, labelpad=12.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。)
axs[1].set_ylabel(r'value',  fontdict = font2)
font2 = {'family':'Times New Roman','style':'normal','size':22,  }
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'FaultDetection.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 我们使用CVXPY解决放松的最大似然问题，然后将结果舍入得到布尔解。
import cvxpy as cp
x = cp.Variable(shape=n)
tau = 2*cp.log(1/p - 1)*sigma**2
obj = cp.Minimize(cp.sum_squares(A@x - y) + tau*cp.sum(x))
const = [0 <= x, x <= 1]
cp.Problem(obj, const).solve()
print("final objective value: {}".format(obj.value))

# relaxed ML estimate
x_rml = np.array(x.value).flatten()

# rounded solution
x_rnd = (x_rml >= .5).astype(int)





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib

def errors(x_true, x, threshold=.5):
    '''Return estimation errors.

    Return the true number of faults, the number of false positives, and the number of false negatives.
    '''
    n = len(x_true)
    k = sum(x_true)
    false_pos = sum(np.logical_and(x_true < threshold, x >= threshold))
    false_neg = sum(np.logical_and(x_true >= threshold, x < threshold))
    return (k, false_pos, false_neg)

def plotXs(x_true, x_rml, x_rnd, filename=None):
    '''Plot true, relaxed ML, and rounded solutions.'''
    matplotlib.rcParams.update({'font.size': 14})
    xs = [x_true, x_rml, x_rnd]
    titles = ['x_true', 'x_rml', 'x_rnd']

    n = len(x_true)
    k = sum(x_true)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 3))

    for i,x in enumerate(xs):
            ax[i].plot(range(n), x)
            ax[i].set_title(titles[i])
            ax[i].set_ylim([-0.1, 1.1])

    if filename:
        fig.savefig(filename, bbox_inches='tight')

    return errors(x_true, x_rml, 0.5)


err = plotXs(x_true, x_rml, x_rnd, filepath2+'fault.eps')




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




























































































































































































































































































