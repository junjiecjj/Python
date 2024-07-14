

#%% Bk5_Ch17_01
# 图 3. 鸢尾花四个特征的高斯 KDE 曲线
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris data
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

# KDE

fig, axes = plt.subplots(2,2)

sns.kdeplot(data=X_df, fill=True, x = feature_names[0], ax = axes[0][0])
axes[0][0].set_xlim([0,8]); axes[0][0].set_ylim([0,1])
sns.kdeplot(data=X_df, fill=True, x = feature_names[1], ax = axes[0][1])
axes[0][1].set_xlim([0,8]); axes[0][1].set_ylim([0,1])
sns.kdeplot(data=X_df, fill=True, x = feature_names[2], ax = axes[1][0])
axes[1][0].set_xlim([0,8]); axes[1][0].set_ylim([0,1])
sns.kdeplot(data=X_df, fill=True, x = feature_names[3], ax = axes[1][1])
axes[1][1].set_xlim([0,8]); axes[1][1].set_ylim([0,1])

plt.show()


#%% Bk5_Ch17_02
# 图 7. 鸢尾花四个特征数据的概率密度函数曲线
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

plt.close('all')

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
axs = [ax1, ax2, ax3, ax4]
for name, ax in zip(feature_names, axs):
    df = X_df[name]
    KDE = sm.nonparametric.KDEUnivariate(df)
    KDE.fit(bw=0.5) # 0.1, 0.2, 0.4
    ax.fill_between(KDE.support, KDE.density, facecolor = '#DBEEF4')
    ax.plot(KDE.support, KDE.density)
    ax.fill_between(KDE.support, KDE.cdf, facecolor = 'gray')
    ax.plot(KDE.support, KDE.cdf, color = 'r')
    ax.scatter(df, 0.03*np.abs(np.random.randn(df.size)),marker = 'x')

    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,8])
    ax.set_xlabel(name)
plt.show()


# 图 8. 鸢尾花四个特征数据的累积概率密度函数曲线
# Cumulative distribution
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

axs = [ax1, ax2, ax3, ax4]

for name, ax in zip(feature_names, axs):
    df = X_df[name]
    KDE = sm.nonparametric.KDEUnivariate(df)
    KDE.fit(bw=0.5) # 0.1, 0.2, 0.4
    ax.fill_between(KDE.support, KDE.cdf, facecolor = '#DBEEF4')
    ax.plot(KDE.support, KDE.cdf)
    ax.plot(KDE.support, KDE.density)

    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,8])
    ax.set_xlabel(name)



#%% Bk5_Ch17_03
#  八个不同核函数

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import kernel_switch
from itertools import islice

list(kernel_switch.keys())

# Create a figure
fig = plt.figure(figsize=(12, 5))

# Enumerate every option for the kernel
for i, (ker_name, ker_class) in enumerate(islice(kernel_switch.items(),8)):
    # Initialize the kernel object
    kernel = ker_class()

    # Sample from the domain
    domain = kernel.domain or [-3, 3]
    x_vals = np.linspace(*domain, num=2**10)
    y_vals = kernel(x_vals)

    # Create a subplot, set the title
    ax = fig.add_subplot(2, 4, i + 1)
    ax.set_title('Kernel function "{}"'.format(ker_name))
    ax.plot(x_vals, y_vals, lw=3, label='{}'.format(ker_name))
    ax.scatter([0], [0], marker='x', color='red')
    plt.grid(True, zorder=-5)
    ax.set_xlim(domain)
plt.tight_layout()
plt.show()


# 图 13. 八个不同核函数得到的不同的概率密度估计
data = [-3,-2,0,2,2.5,3,4]
kde = sm.nonparametric.KDEUnivariate(data)

# Create a figure
fig = plt.figure(figsize=(12, 5))

# Enumerate every option for the kernel
for i, kernel in enumerate(islice(kernel_switch.keys(),8)):
    # Create a subplot, set the title
    ax = fig.add_subplot(2, 4, i + 1)
    ax.set_title('Kernel function "{}"'.format(kernel))

    # Fit the model (estimate densities)
    kde.fit(kernel=kernel, fft=False, bw=1.5)

    ax.fill_between(kde.support, kde.density, facecolor = '#DBEEF4')
    # Create the plot
    ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)
    ax.scatter(data, np.zeros_like(data), marker='x', color='red')
    plt.grid()
    ax.set_xlim([-6, 6])
    ax.set_ylim([0, 0.3])

plt.tight_layout()
plt.show()

#%% Bk5_Ch17_04
# 图 15. 鸢尾花花萼长度和花萼宽度两个特征数据的 KDE 曲面
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

plt.close('all')

iris = load_iris()
# A copy from Sklearn

X = iris.data
x = X[:, 0]
y = X[:, 1]


xmin, xmax = 4, 8
ymin, ymax = 1, 5

# Perform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
PDF_xy = np.reshape(kernel(positions).T, xx.shape)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, PDF_xy, rstride=4, cstride=4, color = [0.5,0.5,0.5], linewidth = 0.25)
colorbar = ax.contour(xx,yy, PDF_xy,20, cmap = 'RdYlBu_r')
fig.colorbar(colorbar, ax=ax)

ax.set_xlabel('Sepal length, $X_1$')
ax.set_ylabel('Sepal width, $X_2$')
ax.set_zlabel('$f_{X1,X2}(x_1,x_2)$')

ax.set_proj_type('ortho')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.view_init(azim=-135, elev=30)
ax.grid(False)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
# ax.set_zlim(0, 0.7)
plt.tight_layout()
plt.show()

# 图 16. 鸢尾花花萼长度和花萼宽度两个特征数据的 KDE 曲面等高线图
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Contourf plot
cfset = ax.contourf(xx, yy, PDF_xy, cmap='Blues')
cset = ax.contour(xx, yy, PDF_xy, colors='k')
plt.scatter(x,y,marker = 'x')

# Label plot
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Sepal length, $X_1$')
ax.set_ylabel('Sepal width, $X_2$')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%% Bk5_Ch17_05
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import scipy.stats as st

#%% use seaborn to visualize the data

import seaborn as sns
# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",hue = 'species', palette={'setosa': '#FF3300','versicolor': '#0099FF','virginica':'#8A8A8A'})


sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width",hue = 'species', kind="kde", palette={'setosa': '#FF3300','versicolor': '#0099FF','virginica':'#8A8A8A'})































































































































































































































































































