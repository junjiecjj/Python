#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:54:56 2024

@author: jack
"""


import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = np.linspace(0, 8, 16)
y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

# plot
fig, ax = plt.subplots()

ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=2)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt

x = np.array([i for i in range(30)])
y = np.random.rand(30)

plt.plot(x, y)
plt.show()


##
plt.plot(x, y)   # 先将图画出来
plt.fill_between(x, 0, y, facecolor='green', alpha=0.3)
plt.show()


##
y1 = np.random.rand(30)  # 生成第一条曲线
y2 = y1 + 0.3            # 生成第二条曲线
plt.plot(x, y1, 'b')
plt.plot(x, y2, 'r')
plt.fill_between(x, y1, y2, facecolor='green', alpha=0.3)
plt.show()



##
x = np.array([i for i in range(30)])
y = np.random.rand(30)

# 设置想要高亮数据的位置
position = [[1, 6],
            [10, 12],
            [20, 23],
            [26, 28]]

# 画图
plt.plot(x, y, 'r')
for i in position:
    plt.fill_between(x[ i[0] : i[1] ], 0, 1, facecolor='green', alpha=0.3)
plt.show()











