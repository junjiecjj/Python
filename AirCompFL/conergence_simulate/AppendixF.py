#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:55:40 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate(T, delta1, delta2):
    S = 0
    J = 1
    for t in range(T, 0, -1):
        eta = 0.5/(t+1)
        S += J * eta**delta2
        J *= (1 - eta**delta1)
    return S


Delta1 = 1
Delta2 = [0, 1, 2, 3]
T = 1000

results = np.zeros((len(Delta2), T))
for row in range(len(Delta2)):
    results[row] = [calculate(t, Delta1, Delta2[row]) for t in range(T)]

X =  np.arange(T)

fig, ax = plt.subplots(figsize = (8,6))

colors = plt.cm.cool(np.linspace(0, 1, len(results)))
for i in range(len(results)):
    label =  r"$\delta_1$ = {}, $\delta_2$ = {}".format(Delta1, Delta2[i])
    ax.loglog(X, results[i], color = colors[i], label = label)


font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
ax.set_xlabel("Communication Round", fontdict = font, labelpad = 2)
ax.set_ylabel("value of sequences", fontdict = font, labelpad = 2)

font = {'weight' : 'normal', 'size': 20,}
ax.legend(loc='best', prop =  font )

ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

ax.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=25, labelfontfamily = 'Times New Roman', pad = 2)
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
ax.tick_params(which = 'minor', axis='x', direction='in', top=True,  width=2, length = 2,  )
ax.tick_params(which = 'minor', axis='y', direction='in', color = 'red',  width=2, length = 2,  )


ax.grid(color = 'black', alpha = 0.3, linestyle = (0, (5, 10)), linewidth = 1.5 )

plt.show()
plt.close()








































