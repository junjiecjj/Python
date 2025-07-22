




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_01.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a = 0
b = 10
num_data = 500

random_data = np.random.uniform(a,b,num_data)

fig, ax = plt.subplots()
# Plot the histogram
# sns.displot(random_data, bins=20)
sns.histplot(random_data, bins=20, ax = ax)
sns.rugplot(random_data, ax = ax)

plt.xlabel('x')
plt.ylabel('Count')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(a,b)
plt.xticks([0,2,4,6,8,10])


fig, ax = plt.subplots()
# Plot empirical cumulative distribution function
# sns.ecdfplot(random_data)

sns.histplot(random_data, bins=20, fill=True, cumulative=True, stat="density")

plt.xlabel('x')
plt.ylabel('Empirical CDF')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
plt.xlim(a,b)
plt.xticks([0,2,4,6,8,10])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_02

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


x = np.linspace(-3, 5, num=100)

# plot PDF
fig = plt.figure()

mean     = 0
variance = 1

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(0, 1)$")

# The location (loc) keyword specifies the mean.
# The scale (scale) keyword specifies the standard deviation.
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = 2
variance = 3

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = -1
variance = 0.5

plt.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

plt.xlabel('$x$')
plt.ylabel('PDF, $f(x)$')

plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=1)

# plot CDF curves
fig = plt.figure()

mean     = 0
variance = 1

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(0, 1)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = 2
variance = 3

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(2, 3)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

mean     = -1
variance = 0.5

plt.plot(x, norm.cdf(x, loc=mean, scale=np.sqrt(variance)), label="$N(-1, 0.5)$")
plt.axvline(x = mean, linestyle = '--', color = 'r')

plt.axhline(y = 0.5, linestyle = '--', color = 'r')

plt.xlabel('$x$')
plt.ylabel('CDF, $F(x)$')

plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=4)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_03.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic
from matplotlib import cm # Colormaps

x = np.linspace(start = -5, stop = 5, num = 200)

# plot PDF curves

fig, ax = plt.subplots()

Ss = np.arange(0.5,2.1,0.1)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Ss)))

for i in range(0,len(Ss)):
    s = Ss[i]
    plt.plot(x, logistic.pdf(x, loc = 0, scale = s), color = colors[int(i)], label = "s = %.1f" %s)

ax.axvline(x = 0, color = 'k', linestyle = '--')

plt.ylim((0, 0.5))
plt.xlim((-5,5))
plt.title("PDF of logistic distribution")
plt.ylabel("PDF")
plt.legend()
plt.show()

# plot CDF curves
fig, ax = plt.subplots()
for i in range(0,len(Ss)):
    s = Ss[i]
    plt.plot(x, logistic.cdf(x, loc = 0, scale = s), color = colors[int(i)], label = "s = %.1f" %s)

ax.axvline(x = 0, color = 'k', linestyle = '--')
plt.ylim((0, 1))
plt.xlim((-5,5))
plt.title("CDF of logistic distribution")
plt.ylabel("CDF")
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_04.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from scipy.stats import norm
from matplotlib import cm # Colormaps

x = np.linspace(start = -5, stop = 5, num = 200)

# plot PDF curves

fig, ax = plt.subplots()
DFs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 30]
colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(DFs)))
for i in range(0,len(DFs)):
    df = DFs[i]
    plt.plot(x, t.pdf(x, df = df, loc = 0, scale = 1), color = colors[int(i)], label = "\u03BD = " + str(df))
ax.axvline(x = 0, color = 'k', linestyle = '--')
# compare with normal
plt.plot(x,norm.pdf(x,loc = 0, scale = 1), color = 'k', label = 'Normal')
plt.ylim((0, 0.5))
plt.xlim((-5,5))
plt.title("PDF of student's t")
plt.ylabel("PDF")
plt.legend()
plt.show()

# plot CDF curves
fig, ax = plt.subplots()
for i in range(0,len(DFs)):
    df = DFs[i]
    plt.plot(x, t.cdf(x, df = df, loc = 0, scale = 1), color = colors[int(i)], label = "\u03BD = " + str(df))

ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0.5, color = 'k', linestyle = '--')

# compare with normal
plt.plot(x,norm.cdf(x,loc = 0, scale = 1), color = 'k', label = 'Normal')
plt.ylim((0, 1))
plt.xlim((-5,5))
plt.title("CDF of student's t")
plt.ylabel("CDF")
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_05.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 500)

# plot PDF curves

fig, ax = plt.subplots()

STDs = np.arange(0.5,2.1,0.1)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(STDs)))

for i in range(0,len(STDs)):
    std = STDs[i]
    plt.plot(x, lognorm.pdf(x, loc = 0, s = std), color = colors[int(i)], label = "\u03C3= %.1f" %std)

plt.ylim((0, 1.5))
plt.xlim((0,10))
plt.title("PDF of lognormal")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()

for i in range(0,len(STDs)):
    std = STDs[i]
    plt.plot(x, lognorm.cdf(x, loc = 0, s = std), color = colors[int(i)], label = "\u03C3= %.1f" %std)

plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("CDF of lognormal")
plt.ylabel("CDF")
plt.legend()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_06.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from scipy.stats import norm, lognorm

def logn_pdf(x,mu,sigma):
    scaling = 1/x/sigma/np.sqrt(2*np.pi)
    exp_part = np.exp(-(np.log(x) - mu)**2/2/sigma**2)
    pdf = scaling*exp_part
    return pdf


width = 4
X = np.linspace(-3,3,200)
Y = np.linspace(0.01,10,200)

mu    = 0
sigma = 1
pdf_X = norm.pdf(X, mu, sigma)
pdf_Y = lognorm.pdf(Y,s = sigma, scale = np.exp(mu))

mu_Y, var_Y, skew_Y, kurt_Y = lognorm.stats(s = sigma, scale = np.exp(mu), moments='mvsk')
# mu_Y = np.exp(mu + sigma**2/2)

pdf_Y_2 = logn_pdf(Y,mu,sigma)

# Plot the conditional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

# # gs.update(wspace=0., hspace=0.)
# plt.suptitle('Marginal distributions', y=0.93)

# Plot surface on top left
ax1 = plt.subplot(gs[0])

# Plot bivariate normal
ax1.plot(X, np.exp(X))
ax1.axvline(x = 0, color = 'k', linestyle = '--')
ax1.axhline(y = 1, color = 'k', linestyle = '--')

ax1.axvline(x = mu, color = 'r', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax1.axhline(y = np.exp(mu), color = 'k', linestyle = '--')

ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(X.min(),X.max())
ax1.set_ylim(Y.min(),Y.max())

# Plot Y marginal
ax2 = plt.subplot(gs[1])

ax2.plot(pdf_Y, Y, 'b', label='$f_{Y}(y)$')

ax2.plot(pdf_Y_2, Y, 'k')
ax2.axhline(y = mu_Y, color = 'r', linestyle = '--')
ax2.axhline(y = np.exp(mu), color = 'r', linestyle = '--')

ax2.fill_between(pdf_Y,Y, edgecolor = 'none', facecolor = '#DBEEF3')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(Y.min(),Y.max())
ax2.set_xlim(0, pdf_Y.max()*1.1)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X marginal
ax3 = plt.subplot(gs[2])

ax3.plot(X, pdf_X, 'b', label='$f_{X}(x)$')
ax3.axvline(x = mu, color = 'r', linestyle = '--')

ax3.fill_between(X,pdf_X, edgecolor = 'none', facecolor = '#DBEEF3')
ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(X.min(),X.max())
ax3.set_ylim(0, pdf_X.max()*1.1)


ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_07.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 500)
# plot PDF curves
fig, ax = plt.subplots()
lambdas = np.arange(0.1,1.1,0.1)
colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(lambdas)))
for i in range(0,len(lambdas)):
    lambda_i = lambdas[i]
    plt.plot(x, expon.pdf(x, loc = 0, scale = 1/lambda_i), color = colors[int(i)], label = "\u03BB= %.2f" %lambda_i)
plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("PDF of exponential distribution")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves
fig, ax = plt.subplots()
for i in range(0,len(lambdas)):
    lambda_i = lambdas[i]
    plt.plot(x, expon.cdf(x, loc = 0, scale = 1/lambda_i), color = colors[int(i)], label = "\u03BB= %.2f" %lambda_i)
plt.ylim((0, 1))
plt.xlim((0,10))
plt.title("CDF of exponential distribution")
plt.ylabel("CDF")
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_08.py 卡方分布：若干 IID 标准正态分布平方和


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from matplotlib import cm # Colormaps

x = np.linspace(start = 0, stop = 10, num = 200)

# plot PDF curves

fig, ax = plt.subplots()

DFs = range(1,10)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(DFs)))

for df in DFs:
    plt.plot(x, chi2.pdf(x, df = df), color = colors[int(df)-1], label = "k = " + str(df))

plt.ylim((0, 1))
plt.xlim((0, 10))
plt.title("PDF of $\\chi^2_k$")
plt.ylabel("PDF")
plt.legend()
plt.show()


# plot CDF curves

fig, ax = plt.subplots()

DFs = range(1,10)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(DFs)))

for df in DFs:
    plt.plot(x, chi2.cdf(x, df = df), color = colors[int(df)-1], label = "k = " + str(df))

plt.ylim((0, 1))
plt.xlim((0, 10))
plt.title("CDF of $\\chi^2_k$")
plt.ylabel("CDF")
plt.legend()
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk5_Ch07_09.py F-分布：和两个服从卡方分布的独立随机变量有关

from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0, 4, 100)

dfn_array = [1, 2, 5, 20, 100]
dfd_array = [1, 2, 5, 20, 100]
dfn_array_, dfd_array_ = np.meshgrid(dfn_array, dfd_array)

#%% PDF of F Distributions
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
for dfn_idx, dfd_idx,ax in zip(dfn_array_.ravel(), dfd_array_.ravel(), axs.ravel()):
    title_idx = '$d_1$ = ' + str(dfn_idx) + '; $d_2$ = ' + str(dfd_idx)
    ax.plot(x_array, f.pdf(x_array, dfn_idx, dfd_idx),
            'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,4)
    ax.set_ylim(0,2)
    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([0,1,2])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%% CDF of F Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
for dfn_idx, dfd_idx,ax in zip(dfn_array_.ravel(), dfd_array_.ravel(), axs.ravel()):
    title_idx = '$d_1$ = ' + str(dfn_idx) + '; $d_2$ = ' + str(dfd_idx)
    ax.plot(x_array, f.cdf(x_array, dfn_idx, dfd_idx),
            'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')



#%% Bk5_Ch07_10.py Beta 分布：概率的概率

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0,1,200)
alpha_array = [0.1, 0.5, 1, 2, 4]
beta_array = [0.1, 0.5, 1, 2, 4]
alpha_array_, beta_array_ = np.meshgrid(alpha_array, beta_array)

#%% PDF of Beta Distributions
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
for alpha_idx, beta_idx,ax in zip(alpha_array_.ravel(), beta_array_.ravel(), axs.ravel()):
    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.pdf(x_array, alpha_idx, beta_idx), 'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,1)
    ax.set_ylim(0,4)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,2,4])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%% CDF of Beta Distributions

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
for alpha_idx, beta_idx,ax in zip(alpha_array_.ravel(), beta_array_.ravel(), axs.ravel()):
    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.cdf(x_array, alpha_idx, beta_idx), 'b', lw=1)
    ax.set_title(title_idx)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_11.py Dirichlet 分布：多元 Beta 分布

import numpy as np
import scipy.stats as st
import scipy.interpolate as si
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# alpha = np.array([1, 1, 1])
# alpha = np.array([2, 2, 2])
# alpha = np.array([4, 4, 4])

# alpha = np.array([1, 4, 4])
# alpha = np.array([4, 1, 4])
# alpha = np.array([4, 4, 1])

# alpha = np.array([4, 2, 2])
# alpha = np.array([2, 4, 2])
# alpha = np.array([2, 2, 4])

# alpha = np.array([1, 2, 4])
# alpha = np.array([2, 1, 4])
alpha = np.array([4, 2, 1])

rv = st.dirichlet(alpha)

x1 = np.linspace(0,1,201)
x2 = np.linspace(0,1,201)

xx1, xx2 = np.meshgrid(x1, x2)

xx3 = 1.0 - xx1 - xx2
xx3 = np.where(xx3 > 0.0, xx3, np.nan)

PDF_ff = rv.pdf(np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])))
PDF_ff = np.reshape(PDF_ff, xx1.shape)

# PDF_ff = np.nan_to_num(PDF_ff)

#%% 2D contour

fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx1, xx2, PDF_ff, 20, cmap='RdYlBu_r')

#%% 3D contour

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1, xx2, PDF_ff, color = [0.7,0.7,0.7], linewidth = 0.25, rstride=10, cstride=10)
ax.contour(xx1, xx2, PDF_ff, levels = 20,  cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,20])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,20])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(True)
plt.show()

#%% 3D visualization

x1_ = np.linspace(0,1,51)
x2_ = np.linspace(0,1,51)

xx1_, xx2_ = np.meshgrid(x1_, x2_)

xx3_ = 1.0 - xx1_ - xx2_
xx3_ = np.where(xx3_ > 0.0, xx3_, np.nan)

PDF_ff_ = rv.pdf(np.array(([xx1_.ravel(), xx2_.ravel(), xx3_.ravel()])))
PDF_ff_ = np.reshape(PDF_ff_, xx1_.shape)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

# Creating plot
PDF_ff_ = np.nan_to_num(PDF_ff_)
ax.scatter3D(xx1_.ravel(), xx2_.ravel(), xx3_.ravel(), c=PDF_ff_.ravel(), marker='.', cmap = 'RdYlBu_r')
ax.contour(xx1_, xx2_, PDF_ff_, 15, zdir='z', offset=0, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=-30, elev=20)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()

#%% Marginal distributions

from scipy.stats import beta

x_array = np.linspace(0,1,200)

alpha_array = alpha
beta_array = alpha.sum() - alpha

# PDF of Beta Distributions
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
for alpha_idx, beta_idx, ax in zip(alpha_array.ravel(), beta_array.ravel(), axs.ravel()):
    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.pdf(x_array, alpha_idx, beta_idx), lw=1)

    ax.set_xlim(0,1)
    ax.set_ylim(0,4)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,2,4])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')
    ax.set_box_aspect(1)
    ax.set_title(title_idx)

#%% Scatter plot of random data
random_data = np.random.dirichlet(alpha, 500).T

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

ax.scatter3D(random_data[0,:], random_data[1,:], random_data[2,:], marker='.')

ax.set_proj_type('ortho')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=30, elev=20)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))
ax.grid()
plt.show()

























































































































































































































































































