#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:33:50 2024

@author: jack
"""

#=================================================================
#                         最佳游行路线
#  https://www.wuzao.com/document/cvxpy/examples/applications/parade_route.html
#=================================================================


import numpy as np
import cvxpy as cp
import scipy as scipy
# import cvxopt as cvxopt
import matplotlib
# matplotlib.use('TkAgg')
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


## 在指定路线上等间距的生成n个点，最后形成路径。
def form_path(points, n):
    x, y = [], []
    pold = points[0]
    for p in points[1:]:
        x += list(np.linspace(pold[0], p[0], n))
        y += list(np.linspace(pold[1], p[1], n))
        pold = p

    path = np.array([x, y]).T
    return path

def form_grid(k):
    xs = list(np.linspace(0, 1, k))
    ys = list(np.linspace(0, 1, k))

    locations = []
    for x in xs:
        for y in ys:
            locations.append(np.array((x,y)))
    return np.array(locations).T

def guard_sets(k = 12, num = 4, noise = 0.2):
    guard_set = []
    grid = form_grid(k)
    for i in range(num):
        pert = noise*np.random.randn(*grid.shape)
        guard_set.append(grid + pert)
    return np.hstack(guard_set)

def inRect(p, rect):
    x, y, w, h = rect
    return x <= p[0] and p[0] <= x + w and y <= p[1] and p[1] <= y + h

def remove_guards(guards, buildings):
    '''Remove guards inside buildings and outside unit square.'''
    outside = []
    for i, guard in enumerate(guards.T):
        inside = False
        for build in buildings:
            if inRect(guard, build):
                inside = True
                break
            if not inRect(guard, (0, 0, 1, 1)):
                inside = True
                break
        if not inside:
            outside.append(i)
    return guards[:,outside]

def intersect(p1, p2, xmin, xmax, ymin, ymax):
    """ determine if a rectangle given by xy limits blocks the line of sight between p1 and p2 """

    block = False
    # if either point inside block
    for p in [p1, p1]:
        if xmin <= p[0] and p[0] <= xmax and ymin <= p[1] and p[1] <= ymax:
            return True
    # if the two points are equal at this stage, then they are outside the block
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return False
    if p2[0] != p1[0]:
        for x in [xmin, xmax]:
            alpha = (x - p1[0])/(p2[0] - p1[0])
            y = p1[1] + alpha*(p2[1] - p1[1])
            if 0 <= alpha and alpha <= 1 and ymin <= y and y <= ymax:
                return True
    if p2[1] != p1[1]:
        for y in [ymin, ymax]:
            alpha = (y - p1[1])/(p2[1] - p1[1])
            x = p1[0] + alpha*(p2[0] - p1[0])
            if 0 <= alpha and alpha <= 1 and xmin <= x and x <= xmax:
                return True
    return False

def p_evade(x, y, r = 0.5, minval = 0.1):
    d = np.linalg.norm(x-y)
    if d > r:
        return 1
    return (1-minval)*d/r + minval

def get_guard_effects(path, guards, buildings, evade_func, k = 12):
    guard_effects = []
    for guard in guards.T:
        guard_effect = []
        for p in path:
            prob = 1
            if not np.any([intersect(p, guard, x, x+w, y, y+h) for x, y, w, h in buildings]):
                prob = evade_func(p, guard)
            guard_effect.append(prob)
        guard_effects.append(guard_effect)
    return np.array(guard_effects).T

    # xs = list(np.linspace(0,1,k))
    # ys = list(np.linspace(0,1,k))
    # locations = []
    # for x in xs:
    #     for y in ys:
    #         point = np.array((x,y))
    #         detect_p = []
    #         for r in path:
    #             detect_p.append(p_evade(point,r,r=.5,m=0))
    #         locations.append((point,np.array(detect_p)))

#%%
buildings = [(0.1, 0.1, 0.4, 0.1),
             (0.6, 0.1, 0.1, 0.4),
             (0.1, 0.3, 0.4, 0.1),
             (0.1, 0.5, 0.4, 0.1),
             (0.4, 0.7, 0.4, 0.1),
             (0.8, 0.1, 0.1, 0.3),
             (0.8, 0.5, 0.2, 0.1),
             (0.2, 0.7, 0.1, 0.3),
             (0.0, 0.7, 0.1, 0.1),
             (0.6, 0.9, 0.1, 0.1),
             (0.9, 0.7, 0.1, 0.2)]

n = 10

points = [(0.05, 0), (0.05, 0.25), (0.55, 0.25), (0.55, 0.6), (0.75, 0.6), (0.75, 0.05),
          (0.95, 0.05), (0.95, 0.45), (0.75, 0.45), (0.75, 0.65), (0.85, 0.65), (0.85, 0.85),
          (0.35, 0.85),(0.35, 0.65),(0.15, 0.65),(0.15, 1)]

path = form_path(points, n)

g = guard_sets(12, 4, 0.02)
g = remove_guards(g, buildings)

guard_effects = get_guard_effects(path, g, buildings, p_evade)

A = 1 - np.log(guard_effects)

fig = plt.figure(figsize=(10,10), constrained_layout=True)
ax = plt.subplot(111, aspect='equal')
for x,y,w,h in buildings:
    rect = plt.Rectangle((x,y), w, h ,fc='y', alpha = 0.3)
    ax.add_patch(rect)

ax.plot(path[:,0], path[:,1], 'bo')
ax.plot(g[0,:], g[1,:],'ro',alpha=.3)

plt.tick_params(direction='in', axis='both', top=True, right=True, labelsize = 26, width = 3, )
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号


# ax.set_rasterized(True)
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'ParadeRoute.eps',  )
plt.close()


#%%
num_guards = 12
tau = 1e-2

m,n = A.shape

w = np.zeros(n)

for i in range(3):
    x = cp.Variable(shape=n)
    t = cp.Variable(shape=1)

    objective = cp.Maximize(t - x.T@w)
    constr = [0 <=x, x <= 1, t <= A@x, cp.sum(x) == num_guards]
    cp.Problem(objective, constr).solve(verbose=False)
    x = np.array(x.value).flatten()
    w = 2/(tau+np.abs(x))
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(x,'o')
    plt.close()

xsol = x
print("final objective value: {}".format(objective.value))

#%%


fig = plt.figure(figsize=(10,10), constrained_layout=True)
ax = plt.subplot(111,aspect='equal')
for x,y,w,h in buildings:
    rect = plt.Rectangle((x,y), w, h, fc='y', alpha=.3)
    ax.add_patch(rect)

ax.plot(path[:,0], path[:,1], 'o')

ax.plot(g[0,:], g[1,:], 'ro', alpha=.3)
ax.plot(g[0, xsol > 0.5], g[1, xsol > 0.5], 'go', markersize = 20, alpha = 0.5)

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'ParadeRoute.eps', format='eps',  )
plt.close()





































