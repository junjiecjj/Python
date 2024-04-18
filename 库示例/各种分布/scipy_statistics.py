#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:43:24 2022

@author: jack

https://docs.scipy.org/doc/scipy/reference/stats.html


"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pylab import tick_params
import copy
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


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


# #二项分布
# from scipy.stats import binom

# #几何分布
# from scipy.stats import geom

# #泊松分布
# from scipy.stats import poisson

# #均匀分布
# from scipy.stats import uniform

# #指数分布
# from scipy.stats import expon

# #正太分布
# from scipy.stats import norm

# https://blog.csdn.net/sinat_39620217/article/details/117410871
#===========================================================================================================================
#                                             scipy模块   正态分布
#===========================================================================================================================



print("正太分布")
from IPython.display import Latex
Latex(r'$ f(x)= \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}  $')



# norm.rvs通过loc和scale参数可以指定随机变量的偏移和缩放参数，这里对应的是正态分布的期望和标准差。size得到随机数数组的形状参数。(也可以使用np.random.normal(loc=0.0, scale=1.0, size=None))

# 生成一个正态分布的 2 行 2 列随机数，均值为 5， 标准差为 1
print("stats.norm.rvs(loc=5, scale=1, size=[2,2]) = \n{}".format(stats.norm.rvs(loc=5, scale=1, size=[2,2] )))

print("stats.norm.rvs(loc=5, scale=1, size=[2,2]) = \n{}".format(stats.norm.rvs(loc=5, scale=1, size=(2, 2) )))

print("stats.norm.rvs(loc = 0,scale = 0.1,size =10) = \n{}".format(stats.norm.rvs(loc = 0,scale = 0.1,size = 10, random_state=None)))


loc = 1
scale = 2.0
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.norm.stats(loc = loc, scale = scale, moments='mvsk')
print(f"{mean}, {var}, {skew}, {kurt} ")

mean, var, skew, kurt = stats.norm(loc = loc, scale = scale,).stats(moments='mvsk')
print(f"{mean}, {var}, {skew}, {kurt} ")



#随机数生成：
from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.norm.rvs(loc = loc, scale = scale, size =10, random_state=1)
print(sample)

sample = stats.norm.rvs(size = 10, loc = loc, scale = scale, random_state=1)
print(sample)

sample = stats.norm(loc = loc, scale = scale).rvs(size = (2 ,3), random_state=1)
print(sample)

sample = stats.norm(loc = loc, scale = scale).rvs(size = [2 ,3], random_state=1)
print(sample)


#=========================================================================
# 画出 pdf, cdf
loc = 1
scale = 2.0


x = np.linspace(stats.norm.ppf(0.01, loc = loc, scale = scale,), stats.norm.ppf(0.99, loc = loc, scale = scale,), 100)
x = np.arange(stats.norm(loc = loc, scale = scale,).ppf(0.01, ), stats.norm.ppf(0.99, loc = loc, scale = scale,),0.1)

pdf = stats.norm.pdf(x, loc = loc, scale = scale,)
frozenpdf = stats.norm(loc = loc, scale = scale,).pdf(x, )
cdf = stats.norm.cdf(x, loc = loc, scale = scale,)
frozencdf = stats.norm(loc = loc, scale = scale,).cdf(x, )


fig, axs = plt.subplots(1, 1)
axs.plot(x, pdf, marker='o', markersize = 10, linestyle='-', label = "pdf")
axs.plot(x, cdf, marker='*', markersize = 10, linestyle='-', label = "cdf")
axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title("正态分布 ")
plt.show()

#=========================================================================

# 求概率密度函数指定点的函数值
print("stats.norm.pdf(0,loc = 0,scale = 1) = \n{}".format(stats.norm.pdf(0, loc = 0, scale = 1)))

print("stats.norm.pdf(0,loc = 0,scale = 1) = \n{}".format(stats.norm(loc = 0, scale = 1).pdf(0,)))

print("stats.norm.pdf(np.arange(3),loc = 0,scale = 1) = \n{}".format(stats.norm.pdf(np.arange(3), loc = 0, scale = 1)))


# 求累计分布函数指定点的函数值
print("stats.norm.cdf(0,loc=3,scale=1) = \n{}".format(stats.norm.cdf(0, loc=3, scale = 1)))

print("stats.norm.cdf(0,loc=3,scale=1) = \n{}".format(stats.norm(loc = 3, scale = 1).cdf(0)))

print("stats.norm.cdf(0,0,1) = \n{}".format(stats.norm.cdf(0, 0, 1)))


#stats.norm.ppf正态分布的累计分布函数的逆函数，即下分位点。
z05 = stats.norm.ppf(0.05, loc=0,scale=1)
print(z05)
print("st.norm.cdf(z05) = {}".format(stats.norm.cdf(z05)))



# Python 生成均值为 2 ，标准差为 3 的一维正态分布样本 500
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# 通过np生成正太分布
s=np.random.normal(1, 2, 50000)
# 最大最小值
s_fit = np.linspace(s.min(), s.max(), 100)
plt.plot(s_fit, stats.norm(1, 2).pdf(s_fit), lw = 2, c='b')
plt.show()


# 通过np生成正太分布
s=np.random.normal(1, 2, 50000)
# 最大最小值
s_fit = np.linspace(s.min(), s.max(), 100)
plt.plot(s_fit, stats.norm.pdf(s_fit, loc = 1, scale = 2), lw = 2, c = 'r')
plt.show()

from scipy import stats
# 产生正态分布随机数
s = stats.norm.rvs(loc=0, scale=1, size=50000)
s_fit = np.linspace(s.min(), s.max())
plt.plot(s_fit, stats.norm(0, 1).pdf(s_fit), lw=2, c='g')
plt.show()

#=================================================================
# https://zhuanlan.zhihu.com/p/35364867
# # 5.连续概率分布：正态分布（Normal Distribution）

import scipy.stats as st

# 第1步，定义随机变量
mu5 = 0   # 平均值
sigma = 1 # 标准差
X5 = np.arange(-5, 5, 0.1)


# 第2步，求概率密度函数（PDF）
y = stats.norm.pdf(X5, loc = mu5, scale = sigma, )


# 第3步，绘图
fontt  = {'family':'Times New Roman','style':'normal','size':22}
fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)

plt.plot(X5,y)
plt.xlabel('随机变量：x', fontproperties=fontt,)
plt.ylabel('概率：y', fontproperties=fontt,)

plt.title('正态分布：$\mu$=%.1f,$\sigma^2$=%.1f' % (mu5, sigma), fontproperties=fontt,)
plt.grid()
plt.show()
#==========================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import style

# style.use('fivethirtyeight')

mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(12,8))
for i in range(len(mu_params)):
    for j in range(len(sd_params)):
        mu = mu_params[i]
        sd = sd_params[j]
        y = stats.norm(mu, sd).pdf(x)
        y1 = stats.norm(mu, sd).cdf(x)
        ax[i, j].plot(x, y, label = 'pdf', alpha=1)
        ax[i, j].plot(x, y1, label = 'cdf', alpha=1)
        ax[i, j].plot(0, 0, label = 'mu={:3.2f}\nsigma={:3.2f}'.format(mu, sd), alpha=0)
        ax[i, j].legend(fontsize=10)
ax[2,1].set_xlabel('x', fontsize=16)
ax[1,0].set_ylabel('pdf(x)', fontsize=16)
plt.suptitle('Gaussian PDF', fontsize=16)
plt.tight_layout()
plt.show()


mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(12,8))
for i in range(len(mu_params)):
    for j in range(len(sd_params)):
        mu = mu_params[i]
        sd = sd_params[j]
        y = stats.norm.pdf(x, loc = mu, scale = sd)
        y1 = stats.norm.cdf(x, loc = mu, scale = sd)
        ax[i, j].plot(x, y, label = 'pdf', alpha=1)
        ax[i, j].plot(x, y1, label = 'cdf', alpha=1)
        ax[i, j].plot(0, 0, label = 'mu={:3.2f}\nsigma={:3.2f}'.format(mu, sd), alpha=0)
        ax[i, j].legend(fontsize=10)
ax[2,1].set_xlabel('x', fontsize=16)
ax[1,0].set_ylabel('pdf(x)', fontsize=16)
plt.suptitle('Gaussian PDF', fontsize=16)
plt.tight_layout()
plt.show()



#==========================================================

import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *

mu, sigma = 5, 0.7
lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)


figure(1)
subplot(2,1,1)
plt.hist(X.rvs(10000), bins=30)   # 截断正态分布的直方图
subplot(2,1,2)
plt.hist(N.rvs(10000),   bins=30)   # 常规正态分布的直方图
plt.show()

#==========================================================
# https://www.cnblogs.com/pinking/p/7898313.html
loc = 1
scale = 2.0
#平均值, 方差, 偏度, 峰度
mean,var,skew,kurt = st.norm.stats(loc, scale,moments='mvsk')
print(f" mean,var,skew,kurt")
#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.linspace(st.norm.ppf(0.01, loc, scale), st.norm.ppf(0.99, loc, scale), 100)
plt.plot(x, st.norm.pdf(x, loc, scale), 'b-', label = 'norm')

font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
plt.title(r'正太分布概率密度函数', fontproperties=font1)
plt.show()

# https://mp.weixin.qq.com/s?__biz=MjM5NzEyMzg4MA==&mid=2649458753&idx=1&sn=64f42fe483c1a187f2f1d9c245b13023&chksm=bec1ea0689b663101816ec28acee7b32399ebbea9dcb4eb4b0d7fed9dd480fda09c2dab89e28&mpshare=1&scene=24&srcid=0808KJeTjjiPTE0GaH9zxydA&sharer_sharetime=1659939715371&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=Aah%2Bw9Caj8%2B5NR4%2BcfuMr8Q%3D&acctmode=0&pass_ticket=CHQxjWvuBtwuL7rNWpEvCckUzLIEX0zizW%2B0hwsg1jJCO3y22VXGvJRUekW%2FEi9z&wx_header=0#rd
mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.subplots(figsize=(8, 5))
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.title("Normal Distribution")
plt.show()


#==========================================================
# https://blog.csdn.net/weixin_48964486/article/details/116083964

# 正态分布随机数的生成函数是normal()，其语法为：
# normal(loc=0.0, scale=1.0, size=None)

# 参数loc：表示正态分布的均值
# 参数scale：表示正态分布的标准差，默认为1
# 参数size：表示生成随机数的数量

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# 生成五个标准正态分布随机数
Norm = np.random.normal(size=5)
# 求生成的正态分布随机数的密度值
stats.norm.pdf(Norm)
# 求生成的正态分布随机数的累积密度值
stats.norm.cdf(Norm)

#绘制正态分布PDF,CDF
# 注意这里使用的pdf和cdf函数是norm包里的
u = 0
sigma = 1
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('X~N({},{})正态分布PDF'.format(u,sigma))
x = np.linspace(-5, 5, 100000)  # 设定分割区间
y1 = stats.norm.pdf(x, u, sigma**2)
plt.plot(x, y1, label = "pdf")
y2 = stats.norm.cdf(x,u,sigma**2)
plt.plot(x, y2, label = "cdf")
plt.legend()
plt.tight_layout()  # 自动调整子图，使之充满画布整个区域
plt.show()


#===========================================================================================================================
#                                             scipy模块   指数分布
#===========================================================================================================================
lambdaUse = 2
loc = 0
scale = 1.0/lambdaUse

#平均值, 方差, 偏度, 峰度
mean,var,skew,kurt = stats.expon.stats(loc = loc, scale = scale, moments='mvsk')
print( mean,var,skew,kurt)
mean,var,skew,kurt = stats.expon(loc = loc, scale = scale,).stats( moments='mvsk')
print( mean,var,skew,kurt)


#随机数生成：
from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.expon.rvs(loc = loc, scale = scale,  size=14, random_state=None)
print(sample)

sample = stats.expon.rvs(loc = loc, scale = scale, size=(2, 3), random_state=None)
print(sample)
sample = stats.expon(loc = loc, scale = scale,).rvs( size=(2, 3), random_state=None)
print(sample)


# https://www.cnblogs.com/pinking/p/7898313.html

from scipy import stats

fig, ax = plt.subplots(1,1)

lambdaUse = 2
loc = 0
scale = 1.0/lambdaUse


#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.linspace(0, stats.expon.ppf(0.99,loc,scale), 100)
ax.plot(x, stats.expon.pdf(x, loc = loc, scale = scale),'b-',label = 'PDF')
ax.plot(x, stats.expon.cdf(x, loc = loc, scale = scale),'r-',label = 'CDF')
font2  = {'family':'Times New Roman','style':'normal','size':17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 20)
legend1 = ax.legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,)
frame1 = legend1.get_frame()
plt.title(u'指数分布概率密度函数')
plt.show()

#================================================================================

#=========================================================================
# 画出 pdf, cdf
from scipy import stats

lambdaUse = 2
loc = 0
scale = 1.0/lambdaUse

x = np.linspace(stats.expon.ppf(0.1, loc = loc, scale = scale), stats.expon.ppf(0.99, loc = loc, scale = scale), 100)
x = np.arange(stats.expon.ppf(0.1, loc = loc, scale = scale), stats.expon.ppf(0.99, loc = loc, scale = scale), 0.01)

pmf = stats.expon.pdf(x, loc = loc, scale = scale)
frozenpdf = stats.expon(loc = loc, scale = scale,).pdf(x, )
cdf = stats.expon.cdf(x, loc = loc, scale = scale)
frozencdf = stats.expon(loc = loc, scale = scale,).cdf(x, )

fig, axs = plt.subplots(1, 1)
axs.plot(x, pmf, marker='o', linestyle='-', markersize = 10, label = "pdf")
axs.plot(x, cdf, marker='*', linestyle='-', markersize = 10, label = "cdf")
axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title('指数分布概率密度函数')
plt.show()


#=========================================================================

#================================================================================

X = np.linspace(0, 5, 5000)

exponetial_distribtuion = stats.expon.pdf(X, loc=0, scale=1)

plt.subplots(figsize=(8,5))
plt.plot(X, exponetial_distribtuion)
plt.title("Exponential Distribution")
plt.legend()
plt.show()



#随机数生成：
from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.expon.rvs(size=10, loc=0, scale=1, random_state=None)
print(sample)


#===========================================================================================================================
#                                             scipy模块   卡方 分布
#===========================================================================================================================


from scipy import stats



df2 = 40
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.chi2(df = df2,).stats(moments='mvsk')
print( mean,var,skew,kurt)
mean, var, skew, kurt = stats.chi2.stats(df = df2, moments='mvsk', )
print( mean,var,skew,kurt)


#随机数生成：
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
df = 1
sample = stats.chi2.rvs(df = df,  size=14, random_state=None)
print(sample)

sample = stats.chi2.rvs(df = df, size=(2, 3), random_state=None)
print(sample)
sample = stats.chi2(df = df,).rvs( size=(2, 3), random_state=None)
print(sample)

#=========================================================================
# 画出 pdf, cdf
df = 1
x = np.linspace(stats.chi2.ppf(0.2, df = df), stats.chi2.ppf(0.99, df = df), 100)
x = np.arange(stats.chi2(df = df).ppf(0.2 ),  stats.chi2.ppf(0.99, df = df), 0.1)

pdf = stats.chi2.pdf(x, df = df)
frozenpdf = stats.chi2(df = df,).pdf(x, )
cdf = stats.chi2.cdf(x, df = df)
frozencdf = stats.chi2(df = df,).cdf(x, )

fig, axs = plt.subplots(1, 1)
axs.plot(x, pdf, marker='o', linestyle='-', markersize = 10, label = "pdf")
axs.plot(x, cdf, marker='*', linestyle='-', markersize = 10, label = "cdf")
axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title('卡方分布概率密度函数')
plt.show()


#===========================================================================
"""
https://blog.csdn.net/weixin_48964486/article/details/116083964
若Z1, Z2, … Zn,为n个服从标准正态分布的随机变量，则变量：

X =Z_1^2+Z_2^2+\cdots+Z_n^2

因为n的取值可以不同，所以卡方分布是一族分布而不是一个单独的分布。根据X的表达式，服从卡方分布的随机变量值不可能取负值，其期望值为n，方差为2n。
"""
plt.plot(np.arange(0, 5, 0.002), stats.chi2.pdf(np.arange(0, 5, 0.002), 3))
plt.title('卡方分布PDF（自由度为3）')
plt.show()


X = np.arange(0, 6, 0.25)

plt.subplots(figsize=(8, 5))
plt.plot(X, stats.chi2.pdf(X, df=1), label="1 d.o.f")
plt.plot(X, stats.chi2.pdf(X, df=2), label="2 d.o.f")
plt.plot(X, stats.chi2.pdf(X, df=3), label="3 d.o.f")
plt.title("Chi-squared Distribution")
plt.legend()
plt.show()


#===========================================================================================================================
#                                             scipy模块   t分布
#===========================================================================================================================


x = np.arange(-4,4.004,0.004)
plt.plot(x, stats.norm.pdf(x), label='Normal')
plt.plot(x, stats.t.pdf(x, 5), label='df=5')
plt.plot(x, stats.t.pdf(x, 30), label='df=30')
plt.legend()
plt.show()




df2 = 40
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.t(df = df2,).stats(moments='mvsk')
print( mean,var,skew,kurt)
mean, var, skew, kurt = stats.t.stats(df = df2, moments='mvsk', )
print( mean,var,skew,kurt)



import seaborn as sns
from scipy import stats

X1 = stats.t.rvs(df=1, size=4)
X2 = stats.t.rvs(df=3, size=4)
X3 = stats.t.rvs(df=9, size=4)

plt.subplots(figsize=(8,5))
sns.kdeplot(X1, label = "1 d.o.f")
sns.kdeplot(X2, label = "3 d.o.f")
sns.kdeplot(X3, label = "6 d.o.f")
plt.title("Student's t distribution")
plt.legend()
plt.show()

#==============================================================================
from scipy import stats

#随机数生成：
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
df = 1
sample = stats.t.rvs(df = df,  size=14, random_state=None)
print(sample)

sample = stats.t.rvs(df = df, size=(2, 3), random_state=None)
print(sample)
sample = stats.t(df = df,).rvs( size=(2, 3), random_state=None)
print(sample)

#=========================================================================
# 画出 pdf, cdf

df = 1
x = np.linspace(stats.t.ppf(0.01, df = df), stats.t.ppf(0.99, df = df), 100)
x = np.arange(stats.t(df = df).ppf(0.01, ), stats.t.ppf(0.99, df = df), 0.1)

pdf = stats.t.pdf(x, df = df)
frozenpdf = stats.t(df = df,).pdf(x, )
cdf = stats.t.cdf(x, df = df)
frozencdf = stats.t(df = df,).cdf(x, )

fig, axs = plt.subplots(1, 1)
axs.plot(x, pdf, marker='o', markersize = 10, linestyle='-', label = "pdf")
axs.plot(x, cdf, marker='*', markersize = 10, linestyle='-', label = "cdf")
axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title("Student's t distribution")
plt.show()


#===========================================================================================================================
#                                             scipy模块   F分布
#===========================================================================================================================

from scipy import stats
import matplotlib.pyplot as plt


df1 = 4
df2 = 40
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.f(dfn = df1, dfd = df2,).stats(moments='mvsk')
print( mean,var,skew,kurt)
mean, var, skew, kurt = stats.f.stats(dfn = df1, dfd = df2, moments='mvsk', )
print( mean,var,skew,kurt)



from scipy import stats

#=========================================================================
#随机数生成：
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
df1 = 4
df2 = 40
sample = stats.f.rvs(dfn = df1, dfd = df2, size=14, random_state=None)
print(sample)

sample = stats.f.rvs(dfn = df1, dfd = df2, size=(2, 3), random_state=None)
print(sample)
sample = stats.f(dfn = df1, dfd = df2,).rvs( size=(2, 3), random_state=None)
print(sample)


#=========================================================================
# 画出 pdf, cdf

df1 = 4
df2 = 40
x = np.linspace(stats.f.ppf(0.01, dfn = df1, dfd = df2,), stats.f.ppf(0.99, dfn = df1, dfd = df2,), 100)
x = np.arange(stats.f.ppf(0.01, dfn = df1, dfd = df2,), stats.f.ppf(0.99, dfn = df1, dfd = df2,), 0.1)

pdf = stats.f.pdf(x, dfn = df1, dfd = df2,)
frozenpdf = stats.f(dfn = df1, dfd = df2,).cdf(x, )
cdf = stats.f.cdf(x, dfn = df1, dfd = df2,)
frozencdf = stats.f(dfn = df1, dfd = df2,).cdf(x, )

fig, axs = plt.subplots(1, 1)
axs.plot(x, pdf, marker='o', markersize = 10, linestyle='-', label = "pdf")
axs.plot(x, cdf, marker='*', markersize = 10, linestyle='-', label = "cdf")
axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title("F distribution") # F分布PDF(df1=4, df2=40)
plt.show()




#===========================================================================================================================
#                                             scipy模块   瑞利分布
#===========================================================================================================================


print("当一个随机二维向量的两个分量呈独立的、有着相同的方差、均值为0的正态分布时，这个向量的模呈瑞利分布。例如，当随机复数的实部和虚部独立同分布于0均值，同方差的正态分布时，该复数的绝对值服从瑞利分布。\
    \n瑞利分布的概率函数为：")
from IPython.display import Latex
Latex(r'$ f(x;\sigma)=\frac{x}{\sigma ^2} e^{-\frac{x^2}{2\sigma^2}}  $')

import matplotlib
matplotlib.use('TkAgg')
from scipy import stats
import matplotlib.pyplot as plt


loc = 0
scale = 2
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.rayleigh(loc = loc, scale = scale).stats(moments='mvsk')
print( mean,var,skew,kurt)
mean, var, skew, kurt = stats.rayleigh.stats(moments='mvsk', loc = loc, scale = scale)
print( mean,var,skew,kurt)

#随机数生成：
from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.rayleigh.rvs(loc = loc, scale = scale,  size=14, random_state=None)
print(sample)

sample = stats.rayleigh.rvs(loc = loc, scale = scale, size=(2, 3), random_state=None)
print(sample)
sample = stats.rayleigh(loc = loc, scale = scale,).rvs( size=(2, 3), random_state=None)
print(sample)

#=========================================================================
# 画出 pdf, cdf

x = np.linspace(stats.rayleigh.ppf(0.01, loc = loc, scale = scale,), stats.rayleigh.ppf(0.99, loc = loc, scale = scale,), 100)
x = np.arange(stats.rayleigh(loc = loc, scale = scale,).ppf(0.01, ), stats.rayleigh.ppf(0.99, loc = loc, scale = scale,),0.1)

pdf = stats.rayleigh.pdf(x, loc = loc, scale = scale,)
frozenpdf = stats.rayleigh(loc = loc, scale = scale,).pdf(x, )
cdf = stats.rayleigh.cdf(x, loc = loc, scale = scale,)
frozencdf = stats.rayleigh(loc = loc, scale = scale,).cdf(x, )


fig, axs = plt.subplots(1, 1)
axs.plot(x, pdf, marker='o', markersize = 10, linestyle='-', label = "pdf")
axs.plot(x, cdf, marker='*', markersize = 10, linestyle='-', label = "cdf")
# axs.plot(x, frozenpdf,   linestyle='--', label = "frozen pdf")
# axs.plot(x, frozencdf,  linestyle='--', label = "frozen cdf")
#axs.vlines(x, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 26}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('x')
plt.ylabel('概率')
plt.title("rayleigh distribution")

out_fig = plt.gcf()

out_fig.savefig("/home/jack/snap/rayleigh.eps")
plt.show()

#===========================================================================================================================
#                                             scipy模块   泊松分布
#===========================================================================================================================
from IPython.display import Latex
print("泊松分布的概率函数为：")

Latex(r'$P(X=k)=\frac{\lambda^{k}}{k !} e^{-\lambda}, k=0,1,2, \ldots $')

print("累积概率分布函数为：")
Latex(r'$ P(X \leq x)=\sum_{k=0}^{x} \frac{\lambda^{k} e^{-\lambda}}{k !} $')
# Latex(r"$\frac{{\partial {}}}{{\partial {}}}$".format(1, 2))

"""
https://blog.csdn.net/sinat_39620217/article/details/117410871

假设我每天喝水的次数服从泊松分布，并且经统计平均每天我会喝8杯水
请问：
1、我明天喝7杯水概率？
2、我明天喝9杯水以下的概率？

泊松分布的概率函数为：
    P(X=k)=\frac{\lambda^{k}}{k !} e^{-\lambda}, k=0,1,2, \ldots

累积概率分布函数为：
    P(X \leq x)=\sum_{k=0}^{x} \frac{\lambda^{k} e^{-\lambda}}{k !}

均值方差：泊松分布的均值和方差都是。（上述问题一：\lambda=8，k=7）

"""

#随机数生成：
from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.poisson.rvs(mu=8, size=14, random_state=None)
print(sample)

sample = stats.poisson.rvs(mu=8, size=(2, 3), random_state=None)
print(sample)
sample = stats.poisson(mu=8,).rvs( size=(2, 3), random_state=None)
print(sample)

mu = 9
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.poisson.stats(mu , moments='mvsk')
print(f"{mean} , {var}, {skew}, {kurt}")


from scipy import stats

#*离散分布的简单方法大多数与连续分布很类似，但是pdf被更换为密度函数pmf。
p = stats.poisson.pmf(np.arange(10), mu = 8)
print("喝7杯水概率：",p)

p = stats.poisson( mu = 8).pmf(np.arange(10),)
print("喝7杯水概率：",p)


p = stats.poisson.pmf(8, mu = 8)
print("喝8杯水概率：",p)

p = stats.poisson(mu = 8).pmf(8, )
print("喝8杯水概率：",p)


p = stats.poisson.cdf(9, mu = 8)
print("喝9杯水以下的概率：",p)
p = stats.poisson( mu = 8).cdf(9,)
print("喝9杯水以下的概率：",p)




#泊松分布概率密度函数和累计概率绘图
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.style as style

# 绘图配置
style.use('seaborn-bright')

plt.rcParams['figure.figsize'] = (15, 8)

plt.figure(dpi=120)

# 一段时间内发生的次数
data = np.arange(50)

# PMF 绘制泊松分布的概率密度函数
plt.plot(data, stats.poisson.pmf(data, mu=5), label='pmf(mu=5)')
plt.bar(data, stats.poisson.pmf(data, mu=5), alpha=.5)
# CDF 累积概率密度
plt.plot(data, stats.poisson.cdf(data, mu=5), label='cdf(mu=5)')

# PMF 绘制泊松分布的概率密度函数
plt.plot(data, stats.poisson.pmf(data, mu=15), label='pmf(mu=15)')
plt.bar(data, stats.poisson.pmf(data, mu=15), alpha=.5)
# CDF 累积概率密度
plt.plot(data, stats.poisson.cdf(data, mu=15), label='cdf(mu=15)')

# PMF 绘制泊松分布的概率密度函数
plt.plot(data, stats.poisson.pmf(data, mu=30), label='pmf(mu=30)')
plt.bar(data, stats.poisson.pmf(data, mu=30), alpha=.5)
# CDF 累积概率密度
plt.plot(data, stats.poisson.cdf(data, mu=30), label='cdf(mu=30)')

plt.legend(loc='upper left')
plt.title('poisson')

plt.show()

print('p(x<8)时的概率：{}'.format(stats.poisson.cdf(k=8, mu=15)))
print('p(8<x<20)时的概率：{}'.format(stats.poisson.cdf(k=20, mu=15) - stats.poisson.cdf(k=8, mu=15)))


# https://mp.weixin.qq.com/s?__biz=MjM5NzEyMzg4MA==&mid=2649458753&idx=1&sn=64f42fe483c1a187f2f1d9c245b13023&chksm=bec1ea0689b663101816ec28acee7b32399ebbea9dcb4eb4b0d7fed9dd480fda09c2dab89e28&mpshare=1&scene=24&srcid=0808KJeTjjiPTE0GaH9zxydA&sharer_sharetime=1659939715371&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=Aah%2Bw9Caj8%2B5NR4%2BcfuMr8Q%3D&acctmode=0&pass_ticket=CHQxjWvuBtwuL7rNWpEvCckUzLIEX0zizW%2B0hwsg1jJCO3y22VXGvJRUekW%2FEi9z&wx_header=0#rd
X = stats.poisson.rvs(mu=3, size=500)

plt.subplots(figsize=(8, 5))
plt.hist(X, density=True, edgecolor="black", bins = 10)
plt.title("Poisson Distribution")
plt.show()


#============================================================================
# https://www.51cto.com/article/612836.html
import scipy.stats as stats

for lambd in range(5, 35, 5):
    n = np.arange(0, 60)
    poissonpmf = stats.poisson.pmf(n, lambd)
    plt.plot(n, poissonpmf, '-o', label = "pmf(λ = {:f})".format(lambd))
    plt.bar(n, poissonpmf, alpha=.5)
    possioncdf = stats.poisson.cdf(n, mu=lambd)
    plt.plot(n, possioncdf, label="cdf(λ = {:f})".format(lambd))
    plt.xlabel('Number of Events', fontsize = 18)
    plt.ylabel('Probability', fontsize = 18)
    plt.title("Poisson Distribution varying λ")
    plt.legend()
plt.show()



import scipy.stats as stats

for lambd in range(200, 900, 200):
    n = np.arange(0, 1000)
    poisson = stats.poisson.pmf(n, lambd)
    #plt.plot(n, poisson, '-o', label="λ = {:f}".format(lambd))
    plt.plot(n, stats.poisson.cdf(n, mu=lambd), label="λ = {:f}".format(lambd))
    plt.xlabel('Number of Events', fontsize=22)
    plt.ylabel('CDF', fontsize=22)
    plt.title("Poisson Distribution varying λ")
    plt.legend()


#==================================================================
fig,ax = plt.subplots(1,1)
mu = 2
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = st.poisson.stats(mu,moments='mvsk')
print( mean,var,skew,kurt)
#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.arange(st.poisson.ppf(0.01, mu), st.poisson.ppf(0.99, mu), 0.2)
ax.plot(x, st.poisson.pmf(x, mu),'o')
plt.title(u'poisson分布概率质量函数')
plt.show()



#========================================================================
# https://zhuanlan.zhihu.com/p/35364867
# # 4.离散概率分布：泊松分布（Poisson Distribution）

# 案例：已知某路口发生事故的比率是每天2次，那么在此处一天内发生k次事故的概率是多少？

# 第1 步，定义随机变量
mu4 = 4  # 平均值：每天发生4次事故
k4 = 12   # 次数，现在想知道每天发生12次事故的概率
# 发生事故次数，包含0次，1次，2次，3次，4次事故
X4 = np.arange(0, k4 + 1, 1)


'''
第2步，求对应分布的概率：概率质量函数（PMF）
返回一个列表，列表中每个元素表示随机变量中对应值的概率
分别表示发生0次，1次，2次，3次，4次事故的概率
'''
pList4 = stats.poisson.pmf(X4, mu4)

# 第3步，绘图
plt.plot(X4, pList4, marker='o',linestyle='None')
plt.vlines(X4, 0, pList4)
plt.xlabel('某路口发生k次事故')
plt.ylabel('概率')
plt.title('泊松分布：平均值mu=%i' % mu4)
plt.show()

#===========================================================================================================================
#                                             scipy模块   伯努利概率分布
#===========================================================================================================================

#伯努利分布：伯努利试验单次随机试验，只有"成功（值为1）"或"失败（值为0）"这两种结果，
# 又名两点分布或者0-1分布。

#伯努利分布随机数生成
p=0.7#发生概率

b = stats.bernoulli.rvs(p = p, size=14, random_state=None)#random_state=None每次生成随机
print(b)
b = stats.bernoulli(p = p,).rvs( size=14, random_state=None)#random_state=None每次生成随机
print(b)


#伯努利分布随机数生成
p = 0.7#发生概率

b = stats.bernoulli.rvs(p=p, random_state=None, size=(2, 7))#random_state=None每次生成随机
print(b)
b = stats.bernoulli(p=p, ).rvs(random_state=None, size=(2, 7))#random_state=None每次生成随机
print(b)



p = 0.7
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.bernoulli.stats(p , moments='mvsk')
print(f"{mean}, {var}, {skew}, {kurt} ")
# 0.7, 0.21000000000000002, -0.8728715609439702, -1.2380952380952361

#=================================================================================
#  https://blog.csdn.net/sinat_39620217/article/details/117410871
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pylab import mpl
from matplotlib.font_manager import FontProperties

font  = FontProperties(fname = fontpath+"simsun.ttf", size=13, weight='bold')

p = 0.7#库里投三分命中率

plt.rcParams['axes.unicode_minus']=False #显示负号
X = np.arange(0, 3, 1)#[0, 1, 2)

pList = stats.bernoulli.pmf(X, p)#在离散分布中，请将pmf改为pdf
print(pList)
plt.plot(X, pList, marker='o', linestyle='None')
'''
vlines用于绘制竖直线（vertical lines）,
参数说明：vline(x坐标值，y坐标最小值，y坐标最大值)
我们传入的X是一个数组，是给数组中的每个x坐标值绘制直线，
数值线y坐标最小值是0，y坐标最大值是对应的pList中的值
'''
plt.vlines(X, (0, 0, 0), pList)

plt.xlabel('随机变量：库里投篮1次', fontproperties = font)
plt.ylabel('概率', fontproperties = font)
plt.title('伯努利分布：p=%.2f'% p, fontproperties = font)
plt.show()


#=======================================================================
#  https://zhuanlan.zhihu.com/p/35364867

# # 1.离散概率分布：伯努利分布（Bernoulli Distribution）

# 案例：玩抛硬币的游戏，只抛1次硬币，成功抛出正面朝上记录为1，反面朝上即抛硬币失败记录为0

# 导入包
import numpy as np
import matplotlib.pyplot as plt
# 统计计算包的统计模块
from scipy import stats

'''
第1步，定义随机变量：1次抛硬币
正面朝上记录为1，反面朝上记录为0
'''
# arange用于生成一个等差数组，arange([start, ]stop, [step, ]
X1 = np.arange(0,2,1)

'''
第2步，求对应分布的概率：概率质量函数（PMF）
返回一个列表，列表中每个元素表示随机变量中对应值的概率
'''
p1 = 0.3 # 硬币朝上的概率
pList1 = stats.bernoulli.pmf(X1, p1)


'''
第3步，绘图
plot默认绘制折线
marker：点的形状，值o表示点为圆圈标记（circle marker）
linestyle：线条的形状，值None表示不显示连接各个点的折线
'''
plt.plot(X1, pList1, marker='o', linestyle='None')

'''
vlines用于绘制竖直线(vertical lines),
参数说明：vline(x坐标值, y坐标最小值, y坐标值最大值)
我们传入的X是一个数组，是给数组中的每个x坐标值绘制竖直线，
竖直线y坐标最小值是0，y坐标值最大值是对应pList1中的值
'''
plt.vlines(X1,0,pList1)
plt.xlabel('随机变量：抛1次硬币')
plt.ylabel('概率')
plt.title('伯努利分布：p=%.2f' % p1)
plt.show()


#===========================================================================================================================
#                                             scipy模块   二项分布
#===========================================================================================================================
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

n = 8
p = 0.3
#平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = stats.binom.stats(n = n, p = p, moments='mvsk')
print(f"{mean}, {var}, {skew}, {kurt} ")
# 2.4, 1.68, 0.3086066999241839, -0.15476190476190463


#随机数生成：

# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.binom.rvs(size=12, n = 8, p = 0.5, random_state = 1 )
print(sample)

sample = stats.binom(n=8, p =0.5,).rvs(size=12,  random_state = 1 )
print(sample)
sample = stats.binom(n=8, p =0.5,).rvs(size= (2, 3),  random_state = 1 )
print(sample)




#=========================================================================
# 画出累计分布函数
n2 = 20    # 做某件事情的次数
p2 = 0.5  # 做某件事情成功的概率（抛硬币正面朝上的概率）
X2 = np.arange(stats.binom(n = n2, p = p2).ppf(0.001 ),  stats.expon.ppf(0.99, n2, p2), 1)
#X2 = np.arange(0, n2+1, 1) # 做某件事成功的次数（抛硬币正面朝上的次数）

fig, axs = plt.subplots(1, 1)
pdf = stats.binom.pmf(X2, n = n2, p = p2)
axs.plot(X2, pdf, marker='o', markersize = 12, linestyle='-', label = "PDF概率")
frozenpdf = stats.binom(n = n2, p = p2).pmf(X2, )
axs.plot(X2, frozenpdf,   linestyle='--', label = "froze PDF概率")
cdf = stats.binom.cdf(X2, n = n2, p = p2)
axs.plot(X2, cdf, marker='*', markersize = 12, linestyle='-', label = "CDF概率")
frozencdf = stats.binom(n = n2, p = p2).cdf(X2, )
axs.plot(X2, frozencdf,  linestyle='-', label = "frozen CDF概率")

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 12)
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



plt.xlabel('随机变量：抛硬币正面朝上的次数', fontproperties = font)
plt.ylabel('C概率',fontproperties=font)
plt.title('二项分布：n=%i,p=%.2f' % (n2, p2), fontproperties = font)
plt.show()

#=========================================================================




fig, ax = plt.subplots(1, 1)
n, p = 5, 0.4
x = np.arange(stats.binom.ppf(0.01, n, p),  stats.binom.ppf(0.99, n, p))
ax.plot(x, stats.binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
ax.vlines(x, 0, stats.binom.pmf(x, n, p), colors='g', lw=5, alpha=0.5, label='pmf')

ax.vlines(x, 0, stats.binom(n, p).pmf(x), colors='r', linestyles='-', lw=1, label='frozen pmf')
ax.legend(loc='best', frameon=False)
plt.show()

# 在Numpy库中可以使用binomial()函数来生成二项分布随机数。
# 形式为：binomial(n, p, size=None)
# 参数n是进行伯努利试验的次数，参数p是伯努利变量取值为1的概率，size是生成随机数的数量。
np.random.binomial(100, 0.4, 20)


# 求100次试验，20次成功的概率，p=0.5
stats.binom.pmf(20, 100, 0.4)
# 求100次试验，50次成功的概率,p=0.5
stats.binom.pmf(50, 100, 0.4)





#=========================================================================
#  https://blog.csdn.net/weixin_48964486/article/details/116083964
n = 15
p1 = 0.6

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(221)
plt.bar(['0','1'], [1-p1,p1], width=0.5)
plt.title("二点分布PMF")

plt.subplot(222)
plt.plot(['0','0','1','1',' '], [0, 0.4, 0.4, 1.0,1.0])
plt.title("二点分布CDF")

plt.subplot(223)
b = stats.binom.pmf(range(0, n+1), n, 0.6)
plt.bar([str(i) for i in range(0, n+1)], b)
plt.title('二项分布PMF')

plt.subplot(224)
plt.title("二项分布CDF")
c =  range(0, n+1)
d =  stats.binom.cdf(range(0, n+1), n, p1)
plt.plot(c,d)


plt.show()


#图像绘制
#传入时n+1是因为10次实验有十一种可能的结果组合。
n=10   # 十次试验
p=0.3
plt.rcParams['font.sans-serif'] = ['SimHei']
b = stats.binom.pmf(range(0, n+1), n, p)
plt.bar(range(0, len(b)), b)
plt.title('X~B({},{})二项分布PMF'.format(n,p))
plt.show()

#stats.binom.pmf函数很神奇，传入的第一个参数是数字(指定伯努利试验成功的次数)，
# 生成结果就也是一个数字。如果传入的第一个参数是数组，
#则会将会以该数组的shape输出其中每个数字成功次数的条件下，对应的概率。
dd = stats.binom.pmf(np.arange(0, 21, 1), 100, 0.5)
# 然后对数组求和即求出小于等于20次发生这一事件的概率
dd.sum()

# 依然100次试验成功20次，每次p=0.5
stats.binom.cdf(20, 100, 0.5)


n=10   # 十次试验
p=0.5

plt.title('X~B({},{})二项分布CDF'.format(n,p))
c =   range(0, n)

d = stats.binom.cdf(range(0, n), n, p)

plt.plot(c,d)
plt.show()


#==========================================================================
# https://zhuanlan.zhihu.com/p/35364867
# # 2.离散概率分布：二项分布（Binomial Distribution）

# 案例：继续玩抛硬币游戏，假如抛硬币5次，求抛出正面朝上次数的概率
# 导入包
import numpy as np
import matplotlib.pyplot as plt
# 统计计算包的统计模块
from scipy import stats
from matplotlib.font_manager import FontProperties
fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname = fontpath+"simsun.ttf", size=13, )

# 第1步，定义随机变量：5次抛硬币，正面朝上的次数
n2 = 20    # 做某件事情的次数
p2 = 0.4  # 做某件事情成功的概率（抛硬币正面朝上的概率）
X2 = np.arange(0, n2+1, 1) # 做某件事成功的次数（抛硬币正面朝上的次数）

# 第2步，求对应分布的概率：概率质量函数（PMF）
# 返回一个列表，列表中每个元素表示随机变量中对应值的概率
pList2 = stats.binom.pmf(X2,n2,p2)


'''
第3步，绘图
plot默认绘制折线
marker：点的形状，值o表示点为圆圈标记（circle marker）
linestyle：线条的形状，值None表示不显示连接各个点的折线
'''
plt.plot(X2, pList2, marker='o', linestyle='None')

'''
vlines用于绘制竖直线(vertical lines),
参数说明：vline(x坐标值, y坐标最小值, y坐标值最大值)
我们传入的X是一个数组，是给数组中的每个x坐标值绘制竖直线，
竖直线y坐标最小值是0，y坐标值最大值是对应pList2中的值
'''
plt.vlines(X2, 0, pList2)
plt.xlabel('随机变量：抛硬币正面朝上的次数', fontproperties = font)
plt.ylabel('概率', fontproperties = font)
plt.title('二项分布：n=%i,p=%.2f' % (n2, p2), fontproperties = font)
plt.show()


# 画出累计分布函数
n2 = 20    # 做某件事情的次数
p2 = 0.4  # 做某件事情成功的概率（抛硬币正面朝上的概率）
X2 = np.arange(0, n2+1, 1) # 做某件事成功的次数（抛硬币正面朝上的次数）
pList2 = stats.binom.cdf(X2, n2, p2)
plt.plot(X2, pList2, marker='*', linestyle='-')
plt.xlabel('随机变量：抛硬币正面朝上的次数', fontproperties = font)
plt.ylabel('CDF概率',fontproperties=font)
plt.title('二项分布：n=%i,p=%.2f' % (n2, p2), fontproperties = font)
plt.show()


#==========================================================================
# https://www.51cto.com/article/612836.html
# pmf(random_variable, number_of_trials, probability)
for prob in range(3,10,3):
    x=np.arange(0,25)
    binom=stats.binom.pmf(x, 20, 0.1*prob)
    plt.plot(x, binom, '-o', label="pdf={:f}".format(0.1*prob))
    plt.xlabel('Random Variable', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title("Binomial Distribution varying p")
    plt.legend()


#==========================================================================#
# https://www.cnblogs.com/pinking/p/7898313.html
from scipy.stats import binom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
fig,ax = plt.subplots(1,1)
n = 10
p = 0.5
#平均值, 方差, 偏度, 峰度
mean,var,skew,kurt = binom.stats(n, p, moments='mvsk')
print( mean,var,skew,kurt)
#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
ax.plot(x, binom.pmf(x, n, p),'o-')
ax.plot(x, binom.cdf(x, n, p),'*-')
plt.title(u'二项分布概率质量函数')
plt.show()



#===========================================================================================================================
#                                             scipy模块   几何分布
#===========================================================================================================================

from scipy import stats

#随机数生成：
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
sample = stats.geom.rvs(p=0.3, size=10220, random_state=None)
print(list(sample).count(1))

sample = stats.geom(p=0.3,).rvs( size=10220, random_state=None)
print(list(sample).count(1))

# ===============================================
# https://zhuanlan.zhihu.com/p/35364867
# # 3.离散概率分布：几何分布（Geometric Distribution）

# 案例：向一个喜欢的女孩表白，会存在表白成功和不成功的可能，如果向这个女孩表白，
# 直到表白成功为止，有可能表白1次、2次、3次，现在求首次表白成功的概率

'''
第1步，定义随机变量：
首次表白成功的次数，可能是1次，2次，3次
'''
from scipy import stats
# 第k次做某件事，才取得第1次成功
# 这里我们想知道5次表白成功的概率
k = 20
# p3表示做某件事成功的概率，这里假设每次表白成功的概率为60%
p3 = 0.6
X3 = np.arange(1, k+1, 1)
X3 = np.linspace(stats.geom.ppf(0.000001, p = p3), stats.geom.ppf(0.99999999, p = p3), 20)
X3 = np.arange(stats.geom.ppf(0.000001, p = p3), stats.geom.ppf(0.999999999, p = p3), 1)


pmf = stats.geom.pmf(X3, p = p3)
cdf = stats.geom.cdf(X3, p = p3)

fig, axs = plt.subplots(1, 1)
axs.plot(X3, pmf, marker='o', linestyle='-', label = "pmf")
axs.plot(X3, cdf, marker='*', linestyle='-', label = "cdf")
plt.vlines(X3, 0, pmf, colors='g')

# # 设置图例legend
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 16}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


plt.xlabel('随机变量：表白第k次才首次成功')
plt.ylabel('概率')
plt.title('几何分布：p=%.2f' % p3)
plt.show()

# =============================================================
# https://www.cnblogs.com/pinking/p/7898313.html
from scipy import stats as st
fig,ax = plt.subplots(1,1)
p = 0.5
#平均值, 方差, 偏度, 峰度
mean,var,skew,kurt = st.geom.stats(p, moments='mvsk')
print( mean,var,skew,kurt)
#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.arange(st.geom.ppf(0.01, p), st.geom.ppf(0.99, p))
ax.plot(x, st.geom.pmf(x, p),'o-')
ax.plot(x, st.geom.cdf(x, p),'*-')
plt.title(u'几何分布概率质量函数')
plt.show()




"""
https://blog.csdn.net/share727186630/article/details/107403761

https://baike.baidu.com/item/%E7%91%9E%E5%88%A9%E5%88%86%E5%B8%83/10284554

验证由两个均值为0，方差相同的独立同分布的正太分布分别为实部和虚部的复数向量分布符合瑞丽分布
"""
import numpy as np
mean, std = 0, 3
real = np.random.normal(loc =mean , scale= std, size = 100000000)
img = np.random.normal(loc =mean , scale= std, size = 100000000)

print(" real 所有元素的均值 = {}\n".format(np.mean(real)))  # real所有元素的均值 = 1.9994716916165884
print(" img 所有元素的均值 = {}\n".format(np.mean(img)))    # img所有元素的均值 = 1.9994716916165884

print(" real 所有元素的总体方差 = {}\n".format(real.var()))  #a所有元素的总体方差 = 8.998722858669622
print(" img 所有元素的总体方差 = {}\n".format(np.var(img)))  #a所有元素的总体方差 = 8.998722858669622


H = real  + img*1j
#H = real/np.sqrt(2) + img*1j/np.sqrt(2)

H_mod = np.abs(H)



print("H_mod mean = {}, var = {}\n".format( np.sqrt(np.pi/2)*std, (4-np.pi)*std**2/2 ))
# H_mod mean = 3.7599424119465006, var = 3.862833058845931
print(" H_mod 所有元素的均值 = {}\n".format(np.mean(H_mod)))    # H_mod 所有元素的均值 = 3.7598373748179172
print(" H_mod 所有元素的总体方差 = {}\n".format(np.var(H_mod))) # H_mod 所有元素的总体方差 = 3.863108295676793


















