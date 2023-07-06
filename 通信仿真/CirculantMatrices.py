#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:58:17 2022

@author: jack
"""

import scipy.stats as st
import scipy.stats as stats
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

np.random.seed(1)


# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)


fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=24)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)

# gene[0] = 1
# gene[1] = 0.4
# gene[-1] = 0.4
# filepath2 = "/home/jack/文档/中山大学/20221209小组讨论/figures/"
filepath2 = "/home/jack/snap/Distortation/"
#print(np.roll(gene,-1))
#print(np.roll(gene,1))

# 产生傅里叶矩阵
def FFTmatrix(row,col):
     mat = np.zeros((row,col),dtype=complex)
     for i in range(row):
          for j in range(col):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/row)/(np.sqrt(row)*1.0)
     return mat

# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
          col = len(gen)
     elif type(gen) == np.ndarray:
          col = gen.size
     row = col

     mat = np.zeros((row, col))
     mat[0,:] = gen
     for i in range(1,row):
          mat[i,:] = np.roll(gen,i)

     return mat



N = 128
# 产生傅里叶矩阵
Q = FFTmatrix(N,N)

#=========================  产生H ===========================================
#  产生 生成向量，
gene = np.zeros(N)
gene[:4] = 0.3

# 生成 循环矩阵H
H = CirculantMatric(gene, N)


HTH = H.T@H  #  np.dot(np.mat(H).T, np.mat(H))

#=======================================
#==== 酉相似对角化 ===========
#=======================================

QHTHQ = Q.T.conjugate()@HTH@Q
A = np.diagonal(QHTHQ)
DiagQHTHQ = np.real(A)

#=========================  产生Kx ===========================================
#  产生 生成向量，
gene = np.zeros(N)
gene[0] = 1
gene[1] = 0.4
gene[-1] = 0.4

# 生成 循环矩阵Kx
Kx = CirculantMatric(gene, N)


#=======================================
#==== 酉相似对角化 ===========
#=======================================

QKxQ = Q.T.conjugate()@Kx@Q
DiagQKxQ1  = np.diagonal(QKxQ)
# DiagQKxQ1 = np.around(DiagQKxQ,decimals=2)
DiagQKxQ = np.real(DiagQKxQ1)

#========================================


Kz = np.zeros((N,N))



import cvxpy as cp

def gaussRateDistortion(n, Ds, Do, DiagKx, DiagHTH,   Kz, ):
     
     DiagKx = DiagKx.reshape(1,n)
     DiagHTH = DiagHTH.reshape(1,n)
     # Define and solve the CVXPY problem.
     # Create a symmetric matrix variable.
     X = cp.Variable(((1,n)))
     # The operator >> denotes matrix inequality.
     constraints = [0 <= X, X <= DiagKx,
                    X @ DiagHTH.T <= Ds-cp.trace(Kz),
                    cp.sum(X) <= Do,
                    ]
     obj =  cp.Minimize(cp.sum(cp.log(DiagKx) - cp.log(X)))
     
     prob = cp.Problem(obj, constraints)
     
     # print(cp.installed_solvers())
     # ['CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCIPY', 'SCS']
     try: 
          prob.solve()
     except  :
          print(f" Ds = {Ds}, Do = {Do} , no optimal answer")
          return Ds, Do, prob.status, -1, -1
     

     if prob.status=='optimal':
          #print(f"{Ds, Do,  prob.value}")
          return Ds, Do, prob.status, prob.value, X.value
     else:
           return Ds, Do, prob.status, 0, 0


dsA3 = 32.14285714285714
doA3 = 109.89795918367348


Ds, Do,  status, R, delta = gaussRateDistortion(N,dsA3,doA3,DiagQKxQ,DiagQHTHQ,Kz)


M1 = 100
N1 = 100
DsStart = 1
DsEnd = 110
DoStart = 1
DoEnd = 185
DoArryX = np.linspace(DoStart, DoEnd ,M1)
DsArryY = np.linspace(DsStart, DsEnd, N1)

Z = np.zeros((DsArryY.shape[0] ,DoArryX.shape[0]))

for i, Do in enumerate(np.linspace(DoStart, DoEnd ,M1)):
     for j, Ds in enumerate(np.linspace(DsStart, DsEnd, N1)):
          ds, do, statu, res, X = gaussRateDistortion(N,Ds,Do,DiagQKxQ,DiagQHTHQ,Kz)
          Z[j,i] = res
          #results.append([ds, do, res])



#========================================
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure(figsize=(10, 10))
axs = fig.gca(projection='3d')
#results = np.array(results)

#res = results[~np.isnan(results).any(axis=1), :]
X, Y = np.meshgrid(DoArryX, DsArryY)

#axs.view_init(elev=45, azim=45)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴


# #生成表面， alpha 用于控制透明度
surf = axs.plot_surface(X, Y, Z,rstride=1,cstride=1,linewidth=1, antialiased=False,alpha=0.8, cmap='viridis')
C = axs.contour(X,Y,Z,levels=[50], zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
plt.clabel(C, inline = True, fontsize = 20)
# axs.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
# axs.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
# plt.contour(X, Y, Z)   # 立体图上的曲线

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# axs.zaxis.set_rotate_label(False) #一定要先关掉默认的旋转设置
# axs.set_zlabel('Weight (kg)', rotation = 90)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 18)
axs.set_xlabel(r'$D_{O}$', fontproperties=font1)
axs.set_ylabel(r'$D_{S}$', fontproperties=font1)
axs.set_zlabel(r'$R(D_{S},D_{O})$', fontproperties=font1)

#设定显示范围
axs.set_xlim(0, DoEnd)  #拉开坐标轴范围显示投影
axs.set_ylim(DsEnd, 0)
#axs.set_zlim(0, 8)

axs.grid(True)

ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2); # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);   # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);    # 设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3)
labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(18) for label in labels] #刻度值字号



out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
out_fig.savefig(filepath2+'Figure9.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()


#=======================

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

C = axs.contour(Y, X, Z, levels=[50])
plt.clabel(C, inline = True, fontsize = 25)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 18)
axs.set_xlabel(r'$D_{S}$', fontproperties=font1)
axs.set_ylabel(r'$D_{O}$', fontproperties=font1)
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
axs.set_title(r'$R(D_{s}, D_{o})$', fontproperties=fontt)


ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2); # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);   # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2);  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);    # 设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20, width=3)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号



out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
out_fig.savefig(filepath2+'Figure9_1.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()


