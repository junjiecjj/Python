#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:23:10 2024

@author: jack
"""

import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import scipy.stats as st
import matplotlib.gridspec as gridspec

# visualize surface using 3D and 2D
def plot_surface(xx1, xx2, surface, z_height, title_txt, data = None ):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, surface, color = [0.7,0.7,0.7], linewidth = 0.25)
    ax.contour3D(xx1, xx2, surface,20, cmap = 'RdYlBu_r')
    ax.set_proj_type('ortho')

    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    ax.set_zlabel('PDF')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # 3D坐标区的背景设置为白色
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xticks([4,5,6,7,8])
    ax.set_yticks([1,2,3,4,5])
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())
    ax.set_zlim3d([0, z_height])
    ax.view_init(azim=-120, elev=30)
    ax.set_title(title_txt)
    ax.grid(False)

    ax = fig.add_subplot(1, 2, 2)
    # Contourf plot
    # cfset = ax.contourf(xx1, xx2, surface, 12, cmap='RdYlBu_r')
    # cset = ax.contour(xx1, xx2, surface, 12, colors='w')
    cset = ax.contour(xx1, xx2, surface, 20, cmap='RdYlBu_r')
    if  data is not None:
        sns.scatterplot(data = data, x="sepal_length", y="sepal_width", hue = 'species', ax = ax, palette={'setosa': '#FF3300','versicolor': '#0099FF','virginica':'#8A8A8A'})
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())
    # Label plot
    # ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title(title_txt)
    plt.show()
    plt.close()
    return

# visualize 2D PDF contour and marginal
def plot_joint_marginal(xx1, xx2, surface, x1, f_x1, x2, f_x2, x1_s, x2_s, color, title_txt):
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

    # # gs.update(wspace=0., hspace=0.)
    # plt.suptitle('Marginal distributions', y=0.93)

    # Plot surface on top left
    ax1 = plt.subplot(gs[0])

    # Plot bivariate normal
    ax1.contour(xx1, xx2, surface, 20, cmap='RdYlBu_r')
    ax1.scatter(x1_s, x2_s, c=color)

    ax1.set_xlabel('Sepal length, $x_1$')
    ax1.set_ylabel('Sepal width, $x_2$')
    ax1.yaxis.set_label_position('right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(title_txt)

    # Plot Y marginal
    ax2 = plt.subplot(gs[1])
    ax2.plot(f_x2, x2, color = color)
    ax2.fill_between(f_x2, x2, edgecolor = 'none', facecolor = color, alpha = 0.2)
    ax2.set_xlabel('PDF')
    ax2.set_ylim(1, 5)
    ax2.set_xlim(0, 1.5)
    ax2.set_xticks([0, 0.5, 1, 1.5])
    ax2.set_yticks([1,2,3,4,5])
    ax2.invert_xaxis()
    ax2.yaxis.tick_right()

    # Plot X marginal
    ax3 = plt.subplot(gs[2])
    ax3.plot(x1, f_x1, color = color)
    ax3.fill_between(x1, f_x1, edgecolor = 'none', facecolor = color, alpha = 0.2)
    ax3.set_ylabel('PDF')
    ax3.yaxis.set_label_position('left')
    ax3.set_xlim(4,8)
    ax3.set_xticks([4,5,6,7,8])
    ax3.set_ylim(0, 1.5)
    ax3.set_yticks([0, 0.5, 1, 1.5])
    ax4 = plt.subplot(gs[3])
    ax4.set_visible(False)

    plt.show()
    plt.close()
    return

# Initialization
plt.close('all')
iris_sns = sns.load_dataset("iris")
iris = load_iris()
# A copy from Sklearn
X_1_to_4 = iris.data
y = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']
X_df = pd.DataFrame(X_1_to_4, columns=feature_names)
y_df = pd.DataFrame(y, columns=['label'])
y_df = y_df.replace({0: 'C_1', 1: 'C_2', 2: 'C_3'})

X1_2_df = X_df[['Sepal length, $X_1$','Sepal width, $X_2$']]
x1 = np.linspace(4, 8, 161)
x2 = np.linspace(1, 5, 161)
xx1, xx2 = np.meshgrid(x1, x2) #  (161, 161)
positions = np.vstack([xx1.ravel(), xx2.ravel()]) #   (2, 25921)


#%% likelihood PDF, given class Y
# 图 1. 似然概率 PDF 曲面 fX1,X2|Y(x1,x2|C1)
# given C1 (y = 0)
kernel = st.gaussian_kde(X1_2_df[y==0].values.T)
f_x1_x2_given_C1 = np.reshape(kernel(positions).T, xx1.shape) #  (161, 161)

z_height = 1
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_1)$, likelihood'
plot_surface(xx1, xx2, f_x1_x2_given_C1, z_height, title_txt)

x1_s_C1 = X1_2_df['Sepal length, $X_1$'][y==0]
KDE_x1_given_C1 = sm.nonparametric.KDEUnivariate(x1_s_C1)
KDE_x1_given_C1.fit(bw=0.1)
f_x1_given_C1 = KDE_x1_given_C1.evaluate(x1)

x2_s_C1 = X1_2_df['Sepal width, $X_2$'][y==0]
KDE_x2_given_C1 = sm.nonparametric.KDEUnivariate(x2_s_C1)
KDE_x2_given_C1.fit(bw=0.1)
f_x2_given_C1 = KDE_x2_given_C1.evaluate(x2)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_1)$'
plot_joint_marginal(xx1, xx2, f_x1_x2_given_C1, x1, f_x1_given_C1, x2, f_x2_given_C1, x1_s_C1, x2_s_C1, '#FF3300',title_txt)

# given C2 (y = 1)
# 图 2. 似然概率 PDF 曲面 fX1,X2|Y(x1,x2|C2)
kernel = st.gaussian_kde(X1_2_df[y==1].values.T)
f_x1_x2_given_C2 = np.reshape(kernel(positions).T, xx1.shape)

z_height = 1
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_2)$, likelihood'
plot_surface(xx1, xx2, f_x1_x2_given_C2, z_height, title_txt)

x1_s_C2 = X1_2_df['Sepal length, $X_1$'][y==1]
KDE_x1_given_C2 = sm.nonparametric.KDEUnivariate(x1_s_C2)
KDE_x1_given_C2.fit(bw=0.1)
f_x1_given_C2 = KDE_x1_given_C2.evaluate(x1)

x2_s_C2 = X1_2_df['Sepal width, $X_2$'][y==1]
KDE_x2_given_C2 = sm.nonparametric.KDEUnivariate(x2_s_C2)
KDE_x2_given_C2.fit(bw=0.1)
f_x2_given_C2 = KDE_x2_given_C2.evaluate(x2)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_2)$'
plot_joint_marginal(xx1,xx2,f_x1_x2_given_C2, x1,f_x1_given_C2, x2,f_x2_given_C2, x1_s_C2,x2_s_C2, '#0099FF',title_txt)

# given C3 (y = 2)
# 图 3. 似然概率 PDF 曲面 fX1,X2|Y(x1,x2|C3)
kernel = st.gaussian_kde(X1_2_df[y==2].values.T)
f_x1_x2_given_C3 = np.reshape(kernel(positions).T, xx1.shape)

z_height = 1
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_3)$, likelihood'
plot_surface(xx1, xx2, f_x1_x2_given_C3, z_height, title_txt)

x1_s_C3 = X1_2_df['Sepal length, $X_1$'][y==2]
KDE_x1_given_C3 = sm.nonparametric.KDEUnivariate(x1_s_C3)
KDE_x1_given_C3.fit(bw=0.1)
f_x1_given_C3 = KDE_x1_given_C3.evaluate(x1)

x2_s_C3 = X1_2_df['Sepal width, $X_2$'][y==2]
KDE_x2_given_C3 = sm.nonparametric.KDEUnivariate(x2_s_C3)
KDE_x2_given_C3.fit(bw=0.1)
f_x2_given_C3 = KDE_x2_given_C3.evaluate(x2)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_3)$'
plot_joint_marginal(xx1,xx2,f_x1_x2_given_C3, x1,f_x1_given_C3, x2,f_x2_given_C3, x1_s_C3,x2_s_C3, '#8A8A8A',title_txt)

#%% compare three likelihood PDF surfaces
# 图 4. 比较三个似然概率曲面，平面等高线
fig, ax = plt.subplots()

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())

ax.contour(xx1, xx2, f_x1_x2_given_C1, 15, colors='#FF3300')
ax.contour(xx1, xx2, f_x1_x2_given_C2, 15, colors='#0099FF')
ax.contour(xx1, xx2, f_x1_x2_given_C3, 15, colors='#8A8A8A')

# Label plot
# ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('Sepal width, $x_2$')
plt.gca().set_aspect('equal', adjustable='box')
ax.set_title('Compare three likelihood PDF')
plt.show()
plt.close()
#%% prior probability

y_counts = y_df.value_counts()

#Plot the data:
my_colors = ['#FF3300', '#0099FF', '#8A8A8A']

fig, ax = plt.subplots()
y_counts.plot.bar(color=my_colors)
plt.ylabel('Count')
plt.show()
plt.close()

fig, ax = plt.subplots()
y_prob = y_counts/y_df.count().values[0]
y_prob.plot.bar(color=my_colors)
plt.ylabel('Prior probability')
plt.show()
plt.close()
#%% Joint PDF

f_x1_x2_joint_C1 = f_x1_x2_given_C1*y_prob['C_1']
f_x1_x2_joint_C2 = f_x1_x2_given_C2*y_prob['C_2']
f_x1_x2_joint_C3 = f_x1_x2_given_C3*y_prob['C_3']

# C1 (y = 0) 图 5. 联合概率 PDF 曲面 fX1,X2,Y(x1,x2,C1)
z_height = 1
title_txt = '$f_{X1, X2, Y}(x_1, x_2, C_1)$, joint'
plot_surface(xx1, xx2, f_x1_x2_joint_C1, z_height, title_txt)


# C2 (y = 1) 图 6. 联合概率 PDF 曲面 fX1,X2,Y(x1,x2,C2)
title_txt = '$f_{X1, X2, Y}(x_1, x_2, C_2)$, joint'
plot_surface(xx1, xx2, f_x1_x2_joint_C2, z_height, title_txt)

# C3 (y = 2) 图 7. 联合概率 PDF 曲面 fX1,X2,Y(x1,x2,C3)
title_txt = '$f_{X1, X2, Y}(x_1, x_2, C_3)$, joint'
plot_surface(xx1, xx2, f_x1_x2_joint_C3, z_height, title_txt)

#%% Evidence fX_1, X_2(x_1, x_2)
# 图 9. fX1,X2(x1,x2) 曲面及平面等高线
f_x1_x2 = f_x1_x2_joint_C1 + f_x1_x2_joint_C2 + f_x1_x2_joint_C3

z_height = 0.5
title_txt = '$f_{X1, X2}(x_1, x_2)$, evidence'
plot_surface(xx1, xx2, f_x1_x2, z_height, title_txt, data = iris_sns)

#%% Posterior

f_C1_given_x1_x2 = f_x1_x2_joint_C1/f_x1_x2
f_C2_given_x1_x2 = f_x1_x2_joint_C2/f_x1_x2
f_C3_given_x1_x2 = f_x1_x2_joint_C3/f_x1_x2

# C1 (y = 0) 图 10. 后验概率 fY|X1,X2(C1 | x1,x2) 对应曲面和平面等高线
z_height = 1
title_txt = '$f_{Y|X1, X2}(C_1|x_1, x_2)$, posterior'
plot_surface(xx1, xx2, f_C1_given_x1_x2, z_height, title_txt)

# C2 (y = 1) 图 11. 后验概率 fY|X1,X2(C2 | x1,x2) 对应曲面和平面等高线
title_txt = '$f_{Y|X1, X2}(C_2|x_1, x_2)$, posterior'
plot_surface(xx1, xx2, f_C2_given_x1_x2, z_height, title_txt)

# C3 (y = 2) 图 12. 后验概率 fY|X1,X2(C3 | x1,x2) 对应曲面和平面等高线
title_txt = '$f_{Y|X1, X2}(C_3|x_1, x_2)$, posterior'
plot_surface(xx1, xx2, f_C3_given_x1_x2, z_height, title_txt)


#%% evidence, independence
# 图 15. X1和 X2独立时，估算得到的联合概率 fX1,X2(x1,x2) 曲面和曲面等高线
x1_s = X1_2_df['Sepal length, $X_1$']

KDE_X1 = sm.nonparametric.KDEUnivariate(x1_s)
KDE_X1.fit(bw=0.1)
f_x1 = KDE_X1.evaluate(x1)

x2_s = X1_2_df['Sepal width, $X_2$']
KDE_X2 = sm.nonparametric.KDEUnivariate(x2_s)
KDE_X2.fit(bw=0.1)
f_x2 = KDE_X2.evaluate(x2)

f_x1_x2_indp = np.outer(f_x2, f_x1)

title_txt = '$f_{X1,X2}(x_1,x_2)$, independence'
plot_joint_marginal(xx1, xx2, f_x1_x2_indp, x1, f_x1, x2, f_x2, x1_s, x2_s, '#00448A',title_txt)

z_height = 0.8
title_txt = '$f_{X1, X2}(x_1, x_2)$, independence'
plot_surface(xx1, xx2, f_x1_x2_indp, z_height, title_txt)

#%% surface projected along Y to X-Z plane
# 图 16. fX1,X2(x1,x2) 曲面在两个不同平面的投影，假设特征独立
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1, xx2, f_x1_x2_indp, rstride=2, cstride=0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f_x1_x2_indp, levels = 80, zdir='y', offset=xx2.max(), cmap='rainbow')

ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('Sepal width, $x_2$')
ax.set_title(title_txt)
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.set_zlim(0, 0.7)
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()


#%% surface projected along X to Y-Z plane
# 图 16. fX1,X2(x1,x2) 曲面在两个不同平面的投影，假设特征独立
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1, xx2, f_x1_x2_indp, rstride=0, cstride=2, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f_x1_x2_indp, levels = 80, zdir='x', offset=xx1.max(), cmap='rainbow')

ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('Sepal width, $x_2$')
ax.set_title(title_txt)
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.set_zlim(0, 0.7)
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()



#%% conditional independence
# C1, conditional independence
# 图 17. 在 Y = C1的条件下，X1和 X2条件独立，估算得到的似然概率 fX1,X2|Y(x1,x2|C1)
f_x1_x2_condi_indp_given_C1 = np.outer(f_x2_given_C1, f_x1_given_C1) #  (161, 161)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_1)$, conditional independence'
plot_joint_marginal(xx1,xx2,f_x1_x2_condi_indp_given_C1, x1,f_x1_given_C1, x2,f_x2_given_C1, x1_s_C1,x2_s_C1, '#FF3300',title_txt)

z_height = 1.5
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_1)$, conditional independence'
plot_surface(xx1, xx2, f_x1_x2_condi_indp_given_C1, z_height, title_txt)

# C2, conditional independence
# 图 18. 在 Y = C2的条件下，X1和 X2条件独立，估算得到的似然概率 fX1,X2|Y(x1,x2|C2)
f_x1_x2_condi_indp_given_C2 = np.outer(f_x2_given_C2, f_x1_given_C2)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_2)$, conditional independence'
plot_joint_marginal(xx1, xx2, f_x1_x2_condi_indp_given_C2, x1, f_x1_given_C2, x2, f_x2_given_C2, x1_s_C2, x2_s_C2, '#0099FF', title_txt)

z_height = 1.5
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_2)$, conditional independence'
plot_surface(xx1, xx2, f_x1_x2_condi_indp_given_C2, z_height, title_txt)


# C3, conditional independence
# 图 19. 在 Y = C3的条件下，X1和 X2条件独立，估算得到的似然概率 fX1,X2|Y(x1,x2|C3)
f_x1_x2_condi_indp_given_C3 = np.outer(f_x2_given_C3, f_x1_given_C3)

title_txt = '$f_{X1,X2|Y}(x_1,x_2|C_3)$, conditional independence'
plot_joint_marginal(xx1,xx2,f_x1_x2_condi_indp_given_C3, x1,f_x1_given_C3, x2,f_x2_given_C3, x1_s_C3,x2_s_C3, '#8A8A8A',title_txt)

z_height = 1.5
title_txt = '$f_{X1, X2|Y}(x_1, x_2|C_3)$, conditional independence'
plot_surface(xx1, xx2, f_x1_x2_condi_indp_given_C3, z_height, title_txt)

#%% Evidence fX_1, X_2(x_1, x_2), conditional independence
# 图 21. 假设条件独立，证据因子 fX1,X2(x1,x2) 曲面和平面等高线
f_x1_x2_condi_indp = (f_x1_x2_condi_indp_given_C1*y_prob['C_1'] + f_x1_x2_condi_indp_given_C2*y_prob['C_2'] + f_x1_x2_condi_indp_given_C3*y_prob['C_3'])

z_height = 0.5
title_txt = '$f_{X1, X2}(x_1, x_2)$, evidence, conditional independence'
plot_surface(xx1, xx2, f_x1_x2_condi_indp, z_height, title_txt)

# 图 22. 假设条件独立，证据因子 fX1,X2(x1,x2) 等高线，和边缘概率密度 fX1(x1)、fX2(x2) 曲线关系
title_txt = '$f_{X1,X2}(x_1,x_2)$, conditional independence'
plot_joint_marginal(xx1, xx2, f_x1_x2_condi_indp, x1,f_x1, x2,f_x2, x1_s,x2_s, '#00448A',title_txt)


#%% surface projected along Y to X-Z plane
# 图 23. 假设条件独立，证据因子 fX1,X2(x1,x2) 曲面在两个平面投影曲线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1, xx2, f_x1_x2_condi_indp, rstride=2, cstride=0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f_x1_x2_condi_indp, levels = 80, zdir='y', offset=xx2.max(), cmap='rainbow')

ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('Sepal width, $x_2$')
ax.set_title(title_txt)
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.set_zlim(0, 0.7)
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()


#%% surface projected along X to Y-Z plane
# 图 23. 假设条件独立，证据因子 fX1,X2(x1,x2) 曲面在两个平面投影曲线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1, xx2, f_x1_x2_condi_indp, rstride=0, cstride=2, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f_x1_x2_condi_indp, levels = 40, zdir='x', offset=xx1.max(), cmap='rainbow')

ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('Sepal width, $x_2$')
ax.set_title(title_txt)
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.set_zlim(0, 0.7)
plt.show()











