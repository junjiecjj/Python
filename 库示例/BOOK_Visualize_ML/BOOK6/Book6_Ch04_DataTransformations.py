

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 数据转换




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2', 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%%%%%%%%%%%%%%%%%%%%% visualize original data
sns.set_style("ticks")
X = X_df.to_numpy();

# Visualize the heatmap of X
fig, ax = plt.subplots()
ax = sns.heatmap(X, cmap='RdYlBu_r', xticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=-1, vmax=9)
plt.title('X')

# distribution of column features of X
fig, ax = plt.subplots()
sns.kdeplot(data=X, fill=True, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.title('Distribution of X columns')

# violin plot of data
fig, ax = plt.subplots()
sns.violinplot(data=X_df, palette="Set3", bw=.2, cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates
fig, ax = plt.subplots()
# Make the plot
pd.plotting.parallel_coordinates(iris_sns, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

#%%%%%%%%%%%%%%%%%%%%%  Demean, centralize
X_demean = X_df.sub(X_df.mean())

# distribution of column features of X
fig, ax = plt.subplots()
ax = sns.heatmap(X_demean, cmap='RdYlBu_r', xticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=-3, vmax=3)
plt.title('$X_{demean}$')


fig, ax = plt.subplots()
sns.kdeplot(data=X_demean,fill=True, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of centralized data
fig, ax = plt.subplots()
sns.violinplot(data=X_demean, palette="Set3", bw=.2, cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates
iris_df_demean = X_demean.copy()
iris_df_demean['species'] = iris_sns['species']

fig, ax = plt.subplots()
pd.plotting.parallel_coordinates(iris_df_demean, 'species', colormap=plt.get_cmap("Set2"))
plt.show()


#%%%%%%%%%%%%%%%%%%%%%   标准化：Z 分数
Z_score = (X_df - X_df.mean()) /X_df.std()

fig, ax = plt.subplots()
ax = sns.heatmap(Z_score, cmap='RdYlBu_r', xticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=-3, vmax=3)
plt.title('Z')

# KDE plot of normalized data
fig, ax = plt.subplots()
sns.kdeplot(data=Z_score,fill=True, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of normalized data
fig, ax = plt.subplots()
sns.violinplot(data=Z_score, palette="Set3", bw=.2, cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates
iris_df_z_scores = Z_score.copy()
iris_df_z_scores['species'] = iris_sns['species']

fig, ax = plt.subplots()
pd.plotting.parallel_coordinates(iris_df_z_scores, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

#%%%%%%%%%%%%%%%%%%%%%  归一化：取值在 0 和 1 之间
# similar function: sklearn.preprocessing.minmax_scale
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(X_df)
# X_normalized = scaler.transform(X_df)
X_normalized = (X_df - X_df.min()) /(X_df.max() - X_df.min())
fig, ax = plt.subplots()
ax = sns.heatmap(X_normalized, cmap='RdYlBu_r', xticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=0, vmax=1)
plt.title('Normalized')

# KDE plot of normalized data
fig, ax = plt.subplots()
sns.kdeplot(data=X_normalized,fill=False, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.title('Distribution of $X_{demean}$ columns')

# violin plot of normalized data
fig, ax = plt.subplots()
sns.violinplot(data=X_normalized, palette="Set3", bw=.2, cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# parallel coordinates
iris_df_normalized = X_normalized.copy()
iris_df_normalized['species'] = iris_sns['species']

fig, ax = plt.subplots()
pd.plotting.parallel_coordinates(iris_df_normalized, 'species', colormap=plt.get_cmap("Set2"))
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 广义幂转换


import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# original data: exponential distribution
original_X = np.random.exponential(size = 1000)

# Box-Cox tpower transformation
new_X, fitted_lambda = stats.boxcox(original_X)

# Yeo-Johnson power transformation
# new_X, fitted_lambda = stats.yeojohnson(original_X)


fig, ax = plt.subplots(1, 2)
sns.histplot(original_X, kde = True, label = "Original", ax = ax[0])
sns.histplot(new_X, kde = True, label = "Original", ax = ax[1])


# QQ plot
fig, ax = plt.subplots(1, 2)

stats.probplot(original_X, dist=stats.norm, plot=ax[0])
ax[0].set_xlabel('Normal')
ax[0].set_ylabel('Original data')
ax[0].set_title('')

stats.probplot(new_X, dist=stats.norm, plot=ax[1])
ax[1].set_xlabel('Normal')
ax[1].set_ylabel('Transformed data')
ax[1].set_title('')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 经验累积分布函数

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

# Histograms PDF
fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis",fill = True, binwidth = 0.15,element="step",stat="density", cumulative=False, common_norm=False)

### CDF plot
fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis",fill = False, binwidth = 0.15,element="step",stat="density", cumulative=True, common_norm=False)

fig, ax = plt.subplots()
sns.ecdfplot(data=X_df, palette = "viridis")

#### convert data to emperical CDF
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles=len(X_df), random_state=0)
ecdf = qt.fit_transform(X_df)
ecdf_df = pd.DataFrame(ecdf, columns = X_df.columns)

g = sns.jointplot(data=ecdf_df, x=feature_names[0], y=feature_names[1], xlim = [0,1],ylim = [0,1])
g.plot_joint(sns.kdeplot, cmap="Blues_r", zorder=0, levels=10, fill = True)


### Pairplot of the emperical data

# with no class labels
g = sns.pairplot(ecdf_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_r")
g.map_diag(sns.histplot, kde=False, color = 'b')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 常见插值方法

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

x_known = np.linspace(0, 6, num=7, endpoint=True)
y_known = np.sin(x_known)
# y_known = np.array([-1, -1, -1, 0, 1, 1, 1])

x_fine  = np.linspace(0, 6, num=300, endpoint=True)
y_fine  = np.sin(x_fine)

methods = ['previous', 'next', 'nearest', 'linear', 'cubic']

for kind in methods:
    f_prev = interp1d(x_known, y_known, kind = kind)

    fig, axs = plt.subplots()
    plt.plot(x_known, y_known, 'or')
    plt.plot(x_fine,  y_fine, 'r--',  linewidth = 0.25)
    plt.plot(x_fine,  f_prev(x_fine), linewidth = 1.5)

    for xc in x_known:
        plt.axvline(x=xc, color = [0.6, 0.6, 0.6], linewidth = 0.25)

    plt.axhline(y=0, color = 'k', linewidth = 0.25)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlabel('x'); plt.ylabel('y')
    plt.ylim([-1.1,1.1])



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 拉格朗日插值
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import numpy as np

x_known = np.linspace(0, 6, num=7, endpoint=True)
y_known = np.sin(x_known)
# y_known = np.array([-1, -1, -1, 0, 1, 1, 1])

x_fine  = np.linspace(0, 6, num=300, endpoint=True)
y_fine  = np.sin(x_fine)

x = np.array([0, 1, 2])
y = x**3
poly = lagrange(x_known, y_known)


Polynomial(poly).coef

fig, axs = plt.subplots()
plt.plot(x_known, y_known, 'or')
plt.plot(x_fine,  y_fine, 'r--',  linewidth = 0.25)
plt.plot(x_fine,  poly(x_fine), linewidth = 1.5)

for xc in x_known:
    plt.axvline(x=xc, color = [0.6, 0.6, 0.6], linewidth = 0.25)

plt.axhline(y=0, color = 'k', linewidth = 0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
plt.xlabel('x'); plt.ylabel('y')
plt.ylim([-1.1,1.1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二元插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D

def surface(x1,x2):
    v = (x1 + x2)*np.exp(-2*(x1**2 + x2**2))
    return v

x1_data = np.linspace(-1, 1, 5)
x2_data = np.linspace(-1, 1, 5)
xx1_data, xx2_data = np.meshgrid(x1_data, x2_data) # (5, 5)
yy_data = surface(xx1_data,xx2_data) # (5, 5)

x1_grid = np.linspace(-1.1, 1.1, 23)
x2_grid = np.linspace(-1.1, 1.1, 23)
xx1_grid, xx2_grid = np.meshgrid(x1_grid, x2_grid) # (23, 23)

fig = plt.figure()
ax = plt.axes(projection ="3d")

ax.scatter(xx1_data, xx2_data, yy_data, marker = 'x', c = 'k')
ax.plot_wireframe(xx1_data, xx2_data, yy_data)

methods = ['linear', 'cubic']

for kind in methods:
    f_interp = interp2d(xx1_data, xx2_data, yy_data, kind=kind)
    yy_interp_2D = f_interp(x1_grid, x2_grid) # (23, 23)

    plt.figure()

    lims = dict(cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    # plt.pcolormesh(xx1_grid, xx2_grid, yy_interp_2D, shading='flat', **lims)
    plt.scatter(xx1_data.ravel(),xx2_data.ravel(), marker = 'x', c = 'k')
    plt.axis('scaled')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)

    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    ax.scatter(xx1_data, xx2_data, yy_data, marker = 'x', c = 'k')
    ax.plot_wireframe(xx1_grid, xx2_grid, yy_interp_2D)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 非网格数据插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def surface(x1,x2):
    v = (x1 + x2)*np.exp(-(x1**2 + x2**2))
    return v

X_scatter = np.random.uniform(-1,1,(25,2))
yy_scatter = surface(X_scatter[:,0],X_scatter[:,1])

x1_grid = np.linspace(-1, 1, 100)
x2_grid = np.linspace(-1, 1, 100)
xx1_grid, xx2_grid = np.meshgrid(x1_grid, x2_grid)

methods = ['nearest','linear', 'cubic']

for method in methods:
    yy_interp_2D = griddata(X_scatter, yy_scatter, (xx1_grid, xx2_grid), method=method)

    plt.figure()

    lims = dict(cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    # plt.pcolormesh(xx1_grid, xx2_grid, yy_interp_2D, shading='flat', **lims)
    plt.scatter(X_scatter[:,0],X_scatter[:,1], marker = 'x', c = 'k')
    plt.axis('scaled')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)

    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    ax.scatter(X_scatter[:,0],X_scatter[:,1], yy_scatter, marker = 'x', c = 'k')
    ax.plot_wireframe(xx1_grid, xx2_grid, yy_interp_2D)
























































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 常见二元插值方法


import matplotlib.pyplot as plt
import numpy as np

methods = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']


# Fixing random state for reproducibility
def surface(x1,x2):
    v = (x1 + x2)*np.exp(-2*(x1**2 + x2**2))
    return v

x1_data = np.linspace(-1, 1, 5)
x2_data = np.linspace(-1, 1, 5)
xx1_data, xx2_data = np.meshgrid(x1_data, x2_data)
yy_data = surface(xx1_data,xx2_data)


fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(yy_data, interpolation=interp_method, cmap='RdBu_r')
    ax.set_title(str(interp_method))

plt.tight_layout()
plt.show()



















































































































































































































































