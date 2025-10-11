#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:58:17 2022
@author: jack
"""

import scipy.stats as st
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
#filepath2 = "/home/jack/文档/中山大学/20221209小组讨论/figures/"
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


# 验证傅里叶矩阵的共轭转置乘以自身等于单位阵
QT_Q = Q.T.conjugate()@Q # 傅里叶矩阵的逆就是它的共轭转置。
QT_Q = np.around(QT_Q,decimals=2)
QT_Q1  = np.diagonal(QT_Q)
print(f"傅里叶矩阵的共轭转置与自身乘积:\n{QT_Q}")

#=========================  产生Kx ===========================================
#  产生 生成向量，
gene = np.zeros(N)
gene[0] = 1
gene[1] = 0.4
gene[-1] = 0.4

# 生成 循环矩阵Kx
Kx = CirculantMatric(gene, N)

#================================================
# Fig.7
#================================================

HTH = H.T@H  #  np.dot(np.mat(H).T, np.mat(H))
"""
HTH 既是实对称矩阵，也是循环矩阵，
对于实对称矩阵，可以正交相似对角化为实对角矩阵，假设用到的正交阵为P;
对于循环矩阵，可以酉相似(这里特殊，酉矩阵就是傅里叶矩阵)对角化为实对角矩阵,假设用到的傅里叶矩阵为Q；
那么上述两个P和Q是不同的，但是对角化后的两个对角阵如果不考虑元素的顺序，则是一样的；
下面的程序说明了这一结果；
"""
#=======================================
#==== 酉相似对角化 ===========
#=======================================

QHTHQ = Q.T.conjugate()@HTH@Q
A = np.diagonal(QHTHQ)
DiagQHTHQ = np.real(A)
AroundA = np.around(DiagQHTHQ,decimals=6)

# QHTHQ = np.around(QHTHQ,decimals=2)
# A是对角阵，取对角元素
QHTHQ1 = np.around(QHTHQ,decimals=6)
A1  = np.diagonal(QHTHQ1)


print(f"矩阵 QHTHQ 的秩={np.linalg.matrix_rank(QHTHQ)}")
print(f"矩阵 DiagQHTHQ 的秩={np.linalg.matrix_rank(np.diag(DiagQHTHQ))}")

# 验证循环矩阵的特征值就是循环矩阵的第一行的傅里叶变换的结果。
value = np.fft.fft(HTH[0,:])
value1 = np.around(value,decimals=6)
# value1与A1/AroundA完全一样
#=======================================
#==== 正交相似对角化 ===========
#=======================================

# 计算H^TH的特征值和特征向量
V, P = np.linalg.eigh(HTH)
B = np.linalg.inv(P)@HTH@P
# B = np.dot(np.dot(np.linalg.inv(P),H),P)
B1 = np.around(B,decimals=6)
print(f"相似对角阵 = \n{B}")
B2  = np.diagonal(B1)
Eigenvalue = np.real(V)

print(f"特征值：\n{V}\n特征向量：{P}")

#=========================
X = np.arange(N)

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
#============================= 0 =============================================
axs[0].plot(X, DiagQHTHQ,'b-',)
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs[0].set_xlabel(r'$j$', fontproperties=font1, )
axs[0].set_ylabel(r'$\alpha_{j}$', fontproperties=font1, )
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
#font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title(r'酉相似对角化', fontproperties=font1)


# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
# legend1 = axs[0].legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号


#============================= 1 =============================================

axs[1].plot(X, Eigenvalue,'b-',)
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs[1].set_xlabel(r'$j$', fontproperties=font1, )
axs[1].set_ylabel(r'$\alpha_{j}$', fontproperties=font1, )
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
#font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[1].set_title(r'正交相似对角化', fontproperties=font1)

# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
# legend1 = axs[1].legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明



axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

#fig.subplots_adjust(hspace=0.6)  # 调节两个子图间的距离
#plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.1 )
fig.tight_layout(pad=1, h_pad=1, w_pad=2)
out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
# out_fig.savefig(filepath2+'Figure7.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()



#================================================
# Fig.8
#================================================


#=======================================
#==== 酉相似对角化 ===========
#=======================================

QKxQ = Q.T.conjugate()@Kx@Q
DiagQKxQ1  = np.diagonal(QKxQ)
# DiagQKxQ1 = np.around(DiagQKxQ,decimals=2)
DiagQKxQ = np.real(DiagQKxQ1)


print(f"矩阵 QKxQ 的秩={np.linalg.matrix_rank(QKxQ)}")
print(f"矩阵 DiagQKxQ 的秩={np.linalg.matrix_rank(np.diag(DiagQKxQ))}")
print(f"矩阵 DiagQKxQ1 的秩={np.linalg.matrix_rank(np.diag(DiagQKxQ1))}")


#================================ A1区域的解 ========================================
dsA1 = 32
doA1 = 54
delta = []
dec = 0.0001
LambdaA1 = []
for k in np.linspace(1.0/DiagQKxQ.max(),1/DiagQKxQ.min() ,100000):
     delta = []
     for i in range(N):
          delta.append(min(DiagQKxQ[i], 1.0/k))
     Sum = sum(delta)
     if abs(Sum-doA1) <= 1e-3:
          LambdaA1.append(k)
print(f"满足条件的 LambdaA1 为:{LambdaA1}")
# 满足条件的lambda为:[2.1289816414974867]


solverA1 = []
for i in range(N):
     solverA1.append(min(DiagQKxQ[i], 1.0/LambdaA1[0]))


#================================ A2区域的解 ========================================
dsA2 = 18
doA2 = 76
a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 100
a = np.where(DiagQHTHQ<1e-9,100,DiagQHTHQ)
Min = a.min()

a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 0
a = np.where(DiagQHTHQ<1e-9,0,DiagQHTHQ)
Max = a.max()

LambdaA2 = []
# for k in np.arange(0.01,100,0.01):
for k in np.linspace(1.0/(Max*DiagQKxQ.max()), 1/(Min*DiagQKxQ.min()), 100000):
     delta = []
     for i in range(N):
          delta.append(DiagQHTHQ[i]*min(DiagQKxQ[i],1.0/(abs(DiagQHTHQ[i])*k)))
     Sum = sum(delta)
     if abs(Sum-dsA2) <= 1e-1:
          LambdaA2.append(k)
print(f"满足条件的 LambdaA2 为:{LambdaA2}")

solverA2 = []
for i in range(N):
     solverA2.append(min(DiagQKxQ[i],1.0/(abs(DiagQHTHQ[i])*LambdaA2[0])))


#================================ A3区域的解 ========================================
dsA3 = 19.5
doA3 = 58
a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 100
a = np.where(DiagQHTHQ<1e-9,100,DiagQHTHQ)
Min = a.min()

a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 0
a = np.where(DiagQHTHQ<1e-9,0,DiagQHTHQ)
Max = a.max()

LambdaA3 = []
MuA3 = []
# for k in np.arange(0.01,100,0.01):
for mu in np.linspace(1.0/DiagQKxQ.max(),1/DiagQKxQ.min() ,100):
     for lam in np.linspace(1.0/(Max*DiagQKxQ.max()), 1/(Min*DiagQKxQ.min()), 10000):
          deltaDo = []
          deltaDs = []
          for i in range(N):
               deltaDo.append(min(DiagQKxQ[i], 1.0/(mu + abs(DiagQHTHQ[i])*lam)))
               deltaDs.append(DiagQHTHQ[i]*min(DiagQKxQ[i],1.0/(mu + abs(DiagQHTHQ[i])*lam)))
          SumDs = sum(deltaDs)
          SumDo = sum(deltaDo)
          if abs(SumDs-dsA3) <= 1 and abs(SumDo-doA3) <= 1:
               LambdaA3.append(mu)
               MuA3.append(lam)
print(f"满足条件的 LambdaA3 为:{LambdaA3}")
print(f"满足条件的 MuA3 为:{MuA3}")
# 满足条件的 LambdaA3 为:[1.5432098765432116]
# 满足条件的 MuA3 为:[0.963733794931345]

solverA3 = []
for i in range(N):
     solverA3.append(min(DiagQKxQ[i],1.0/(MuA3[0]+abs(DiagQHTHQ[i])*LambdaA3[0])))



#================================ 画图 ==============================================


X = np.arange(N)
fig, axs = plt.subplots(1, 1, figsize=(10, 6))



axs.plot(X,solverA1, '--',linewidth=3,color='#BA55D3',label = r"$(D_{d},D_{o})=(D_{s,0},D_{o,1})$")
axs.plot(X,solverA2, '--',linewidth=3,color='#FFD700',label = r"$(D_{d},D_{o})=(D_{s,2},D_{o,2})$")
axs.plot(X,solverA3, '--',linewidth=3,color='#20B2AA',label = r"$(D_{d},D_{o})=(D_{s,3},D_{o,3})$")
axs.plot(X, DiagQKxQ,'b-',linewidth=2,label = r"$\delta_{j}^{*} = \sigma_{j}$")


font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs.set_xlabel(r'$j$', fontproperties=font1, )
axs.set_ylabel(r'$\delta_{j}^{*}$', fontproperties=font1, )


font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
legend1 = plt.legend(loc='upper center', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

ax=plt.gca()#获得坐标轴的句柄

ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号



out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
out_fig.savefig(filepath2+'Figure8.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()




#================================================
# Fig.6
#================================================


Dss = []
Dos = []
dec = 0.01
#for k in np.arange(1.0/DiagQKxQ.max()-dec,20/DiagQKxQ.min() ,dec):
for k in np.linspace(1.0/DiagQKxQ.max()-dec,40/DiagQKxQ.min(), 100000):
     ds = 0
     do = 0
     for i in range(N):
          ds += DiagQHTHQ[i]*min(DiagQKxQ[i],1.0/k)
          do += min(DiagQKxQ[i],1.0/k)
     Dss.append(ds)
     Dos.append(do)


Dso = []
Doo = []
a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 100
a = np.where(DiagQHTHQ<1e-9,100,DiagQHTHQ)
Min = a.min()

a = DiagQHTHQ.reshape(1,N).copy()
# a[a<1e-9] = 0
a = np.where(DiagQHTHQ<1e-9,0,DiagQHTHQ)
Max = a.max()

# for k in np.arange(0.01,100,0.01):
for k in np.linspace(1.0/(Max*DiagQKxQ.max()), 1/(Min*DiagQKxQ.min()), 100000):
     ds = 0
     do = 0
     for i in range(N):
          ds += DiagQHTHQ[i]*min(DiagQKxQ[i],1.0/(abs(DiagQHTHQ[i])*k))
          do += min(DiagQKxQ[i],1.0/(abs(DiagQHTHQ[i])*k))
     Dso.append(ds)
     Doo.append(do)



xmax1 = 110.0
ymax1 = 185.0

fig, axs = plt.subplots(1, 1, figsize=(10, 8))

plt.xlim(0,xmax1)
plt.ylim(0,ymax1)


axs.plot(Dss, Dos,'b-',label=r"$C_{s}$")
axs.plot(Dso, Doo,'-', color='#FF8C00',label=r"$C_{o}$")
axs.scatter(Dss[0], Dos[0],  s=100, c='k', marker='*',)

axs.axvline(x=Dss[0], ymin=Dos[0]/ymax1, ymax=1, ls='-', linewidth=1, color='g',)
axs.axhline(y=Dos[0], xmin=Dss[0]/xmax1, xmax=1, ls='-', linewidth=1, color='g',)

axs.scatter(dsA1, doA1,  s=300, color='#BA55D3', marker='.',)
axs.scatter(dsA2, doA2+5,  s=300, color='#FFD700', marker='.',)
axs.scatter(dsA3, doA3,  s=300, color='#20B2AA', marker='.',)

axs.text(dsA1,doA1, r'$(D_{s,1},D_{o,1})$',  fontsize=25)
axs.text(dsA2-10,doA2+8, r'$(D_{s,2},D_{o,2})$',  fontsize=25)
#axs.text(dsA3,doA3, r'$(D_{s,3},D_{o,3})$',  fontsize=25)
axs.annotate(r'$(D_{s,3},D_{o,3})$',
                xy=(dsA3, doA3), xycoords='data',
                xytext=(dsA3-5, doA3-30),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs.set_xlabel(r'$D_{s}$', fontproperties=font1, )
axs.set_ylabel(r'$D_{o}$', fontproperties=font1, )

axs.text(95,160, r'$A_{o}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=25)
axs.text(40,160, r'$A_{2}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=25)
axs.text(95,80, r'$A_{1}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=25)
axs.text(10,50, r'$A_{3}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=25)

axs.text(23,128, r'$(tr(HK_{X}H^{T}+K_{Z}),tr(K_{X}))$',   fontsize=25)

axs.text(2,5, r'$(tr(K_{Z}),0)$',   fontsize=25)

font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=22)
legend1 = plt.legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

ax=plt.gca()#获得坐标轴的句柄

ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=20,width=3)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号


out_fig = plt.gcf()
plt.tight_layout()#  使得图像的四周边缘空白最小化
out_fig.savefig(filepath2+'Figure6.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()



#================================================
# Fig.2.3
#================================================

# Import packages.
import cvxpy as cp
import numpy as np
np.random.seed(1)

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

Z = np.zeros((DsArryY.shape[0] ,DoArryX.shape[0]))

for i, Do in enumerate(np.linspace(DoStart, DoEnd ,M)):
     for j, Ds in enumerate(np.linspace(DsStart, DsEnd, N)):
          ds, do, statu, res, X = gaussRateDistortion(n, Ds,Do, Kx, H, Kz)
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
surf = axs.plot_surface(X, Y, Z,rstride=1, cstride=1, linewidth=1, antialiased=False, alpha=0.8, cmap='viridis')
C = axs.contour(X,Y,Z,levels=[1, 2, 3, 4, 5,6,7],  zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
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
axs.set_ylim(DsEnd, DsStart)
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
out_fig.savefig(filepath2+'Figure2_3.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()

#=======================

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

C = axs.contour(Y, X, Z, levels=[1, 2, 3, 4, 5,6,7  ])
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
out_fig.savefig(filepath2+'Figure2_4.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()

#=======================


#================================================
# Fig.4.5
#================================================

# Import packages.
import cvxpy as cp
import numpy as np
np.random.seed(1)

Kx = 2.0*np.eye(64)


H1 = np.random.randint(2,size=(16,64))
H1[H1==0] = -1
# 以0.95的概率生成0，0.05的概率生成1
p = np.random.choice([0, 1], size=(16,64), p=[0.95, 0.05])
# 将H1与p按对应元素位置的值相乘
H = np.multiply(H1,p)

Kz = 1.0*np.eye(16)
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
          return  prob.value
     else:
          return  0


M = 100
N = 100
DsStart = 18
DsEnd = 130
DoStart = 1
DoEnd = 125
DoArryX = np.linspace(DoStart, DoEnd, M)
DsArryY = np.linspace(DsStart, DsEnd, N)

Z = np.zeros((DsArryY.shape[0] ,DoArryX.shape[0]))

for i, Do in enumerate(np.linspace(DoStart, DoEnd, M)):
     for j, Ds in enumerate(np.linspace(DsStart, DsEnd, N)):
          res  = gaussRateDistortion(n, Ds,Do, Kx, H, Kz)
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
#X, Y = np.meshgrid(DoArryX, DsArryY)

# axs.view_init(elev=45, azim=45)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴

path = "/home/jack/snap/Distortation.npz"
DATA = np.load(path)
X = DATA['XX']
Y = DATA['YY']
Z = DATA['ZZ']

# #生成表面， alpha 用于控制透明度
surf = axs.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=False, alpha=0.8, cmap='viridis')
C = axs.contour(X,Y,Z, levels=[25, 50, 75, 100, 125, 150], zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
plt.clabel(C, inline = True, fontsize = 20)
# axs.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
# axs.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
#plt.contour(X, Y, Z)   # 立体图上的曲线

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# axs.zaxis.set_rotate_label(False) #一定要先关掉默认的旋转设置
# axs.set_zlabel('Weight (kg)', rotation = 90)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 18)
axs.set_xlabel(r'$D_{O}$', fontproperties=font1)
axs.set_ylabel(r'$D_{S}$', fontproperties=font1)
axs.set_zlabel(r'$R(D_{S},D_{O})$', fontproperties=font1)

## 设定显示范围
axs.set_xlim(0, DoEnd)  #拉开坐标轴范围显示投影
axs.set_ylim(DsEnd, DsStart)
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
out_fig.savefig(filepath2+'Figure4_1.pdf', format='pdf', dpi=1000, bbox_inches='tight')

plt.show()

#=======================

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

C = axs.contour(Y, X, Z, levels=[25, 50, 75, 100, 125, 150  ])
plt.clabel(C, inline = True, fontsize = 20)
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
out_fig.savefig(filepath2+'Figure4_2.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.show()



#=======================








