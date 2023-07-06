# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>



import numpy as np
import matplotlib.pyplot as plt

import math
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
#from matplotlib.pyplot import MultipleLocator
filepath2 = '/home/jack/snap/'

fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size = 22)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)


fontt  = {'family':'Times New Roman','style':'normal','size':17}
fonttX  = {'family':'Times New Roman','style':'normal','size':22}
fonttY  = {'family':'Times New Roman','style':'normal','size':22}
fonttitle = {'style':'normal','size':17 }
fontt2 = {'style':'normal','size':19,'weight':'bold'}
fontt3  = {'style':'normal','size':16,}

#普通阶乘
def fact(n):
    if n == 0:
        return 1
    else:
        return n*fact(n-1)
#普通Cmn
def Cmn(n,m):
    return fact(n)/(fact(n-m)*fact(m))



N = 1000
p = 0.25

A = []
B = []
C = []

# X的熵
Hs = -p*math.log2(p) - (1-p)*math.log2(1-p)


for i in range(N+1):
    tmp = (p**i) * ((1-p)**(N-i))
    c = math.comb(N,i)
    A.append(tmp)
    B.append(c*tmp)
    C.append(c)





print(f"=====================================================================================================")
print(f"=======  以下是正面朝上概率为{p},重复{N}次的抛硬币实验 X = [x1,x2,x3,...,xN ] 的典型集一些结果 ============")
print(f"=====================================================================================================\n")

print(f"X的熵为:{Hs}")

a = 2**(-1*N*Hs)
print(f"单个典型序列出现的概率 2**(-1 * {N} *{Hs} ) 为:{a}")


print(f"单个典型序列出现的概率 A[int(N*p)] 为:{a}")


print(f"典型集出现的概率的索引为 B.index(max(B)) :{B.index(max(B))}")

print(f"典型集出现的概率为 max(B) :{max(B)}")

b = 2**(N*Hs)
print(f"典型集的个数为 :{b}")

print(f"典型集的个数为 C_{N}^{int(N*p)} :{Cmn(N, int(N*p))}")



fig,axs = plt.subplots(3,1,figsize=(7,9))

X = np.linspace(0, len(A),  len(A))
axs[0].plot(X ,A,  c='r',linestyle='-',label=r'每种序列单独出现的概率',linewidth=2)

font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 16)
legend1 = axs[0].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

axs[0].tick_params(direction='in',axis='both',labelsize=16,width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

#==================================================================================
axs[1].plot(X ,C,  c='b',linestyle='-',label=r'每种序列出现的组合数',linewidth=2)

font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 16)
legend1 = axs[1].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

axs[1].tick_params(direction='in',axis='both',labelsize=16,width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

#=======================================================================================
axs[2].plot(X ,B,  c='g',linestyle='-',label=r'每种序列出现概率的总和',linewidth=2)
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 16)
legend1 = axs[2].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明
font3 = FontProperties(fname=fontpath+"simsun.ttf", size = 19)
axs[2].set_xlabel(r'正面朝上的次数',fontproperties = font3)

axs[2].tick_params(direction='in',axis='both',labelsize=16,width=3,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

#=======================================================================================
fig.subplots_adjust(hspace=0.2)#调节两个子图间的距离
out_fig = plt.gcf()
out_fig.savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

import numpy as np
import pandas as pd

L = 1000000
RandomNumber=np.random.choice([0,1],size=L, replace=True,p=[0.4,0.6])
pd.Series(RandomNumber).value_counts() # 计算频数分布value_counts()函数
pd.Series(RandomNumber).value_counts()/L  #计算概率分布





print(f"=====================================================================================================")
print(f"=======  验证随机过程习题第四题 ============")
print(f"=====================================================================================================\n")

n=10

p=0.4
q=1-p

sum=0

for i in range(n):
    P = math.comb(n+i-1,i)*(p**n * q**i + q**n * p**i)
    sum += P

print(f"sum = {sum}")


n=10

p=0.4
q=1-p

sum=0
for k in  range(n):
    P =  math.comb(n+k-1,k) * (p**n * q**k + q**n * p**k)
    sum += P

print(f"sum = {sum}")
