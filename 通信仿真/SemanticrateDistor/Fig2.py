#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:30:40 2025

@author: jack
"""



# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)
#================================================
# Fig.2.3
#================================================

Kx = np.array([[11, 0, 0.5],
               [0, 3, -2],
               [0.5, -2, 2.35]])

H = np.array([[0.0701,  0.305,  0.457],
              [-0.0305, -0.220, 0.671]])


Kz = np.array([[0.701, -0.305],
               [-0.305, 0.220]])

n = Kx.shape[0]


def gaussRateDistortion(n,Ds, Do, Kx, H, Kz, ):
     # Define and solve the CVXPY problem.
     # Create a symmetric matrix variable.
     X = cp.Variable((n,n), symmetric=True)
     # The operator >> denotes matrix inequality.
     constraints = [0 << X,
                    X << Kx,
                    cp.trace(H @ X @H.T) <= Ds-cp.trace(Kz),
                    cp.trace(X) <= Do,
                    ]

     prob = cp.Problem(cp.Minimize(1.0/2*(cp.log_det(Kx)-cp.log_det(X))), constraints)
     prob.solve()

     if prob.status=='optimal':
          print(f"{Ds, Do,  prob.value}")
          return Ds, Do, prob.status, prob.value, X.value
     else:
           return Ds, Do, prob.status, 0, 0

M = 100
N = 100
DsStart = 1
DsEnd = 3.1
DoStart = 0.3
DoEnd = 16
DoArryX = np.linspace(DoStart, DoEnd ,M)
DsArryY = np.linspace(DsStart, DsEnd, N)
X, Y = np.meshgrid(DoArryX, DsArryY)
Z = np.zeros((DsArryY.shape[0] ,DoArryX.shape[0]))

for i, Do in enumerate(np.linspace(DoStart, DoEnd ,M)):
     for j, Ds in enumerate(np.linspace(DsStart, DsEnd, N)):
          ds, do, statu, res, X = gaussRateDistortion(n, Ds,Do, Kx, H, Kz)
          Z[j,i] = res
          #results.append([ds, do, res])

#========================================
fig = plt.figure(figsize=(10, 10))
axs = plt.axes(projection='3d')

# #生成表面， alpha 用于控制透明度
surf = axs.plot_surface(X, Y, Z,rstride=1, cstride=1, linewidth=1, antialiased=False, alpha=0.8, cmap='viridis')
C = axs.contour(X,Y,Z, levels=[1, 2, 3, 4, 5,6, 7],  zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
plt.clabel(C, inline = True, fontsize = 20)

fig.colorbar(surf, shrink=0.5, aspect=5)

axs.set_xlabel(r'$D_{O}$', )
axs.set_ylabel(r'$D_{S}$', )
axs.set_zlabel(r'$R(D_{S},D_{O})$', )

#设定显示范围
axs.set_xlim(0, DoEnd)  #拉开坐标轴范围显示投影
axs.set_ylim(DsEnd, DsStart)
#axs.set_zlim(0, 8)

axs.grid(True)

out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
plt.show()

#=======================

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

C = axs.contour(Y, X, Z, levels=[1, 2, 3, 4, 5,6,7  ])
plt.clabel(C, inline = True, fontsize = 25)

axs.set_xlabel(r'$D_{S}$', )
axs.set_ylabel(r'$D_{O}$', )
axs.set_title(r'$R(D_{s}, D_{o})$', )

out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
# out_fig.savefig(filepath2+'Figure2_4.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()

#=======================









