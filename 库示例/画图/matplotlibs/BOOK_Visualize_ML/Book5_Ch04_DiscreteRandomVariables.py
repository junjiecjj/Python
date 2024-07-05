



#%% 代码 Bk5_Ch04_01.py 绘制表 1 中图像。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# utility functions

def plot_stem(df, column):
    pmf_df = df[column].value_counts().to_frame().sort_index()/len(df)

    fig, ax = plt.subplots()
    centroid = np.sum(pmf_df[column].astype(float) * pmf_df.index)
    plt.stem(pmf_df.index, pmf_df[column].astype(float))
    plt.vlines(centroid, 0, 0.3, colors = 'r', linestyles = '--')
    plt.ylim(0, 0.3)
    plt.xlim(-5,40)
    plt.ylabel('Probability, PMF')
    plt.title(column + '; average = ' + '{0:.2f}'.format(centroid))
    return

def plot_contour(XX1, XX2, df, column, XX1_fine, XX2_fine, YY):
    XX1_ = XX1.ravel()
    XX2_ = XX2.ravel()

    levels = np.sort(df[column].unique())
    # print(list(levels)) # test only

    fig, ax = plt.subplots()
    plt.scatter(XX1_,XX2_)
    CS = plt.contour(XX1_fine, XX2_fine, YY, levels = levels, cmap = 'rainbow')

    # plt.contour(XX1_fine, XX2_fine, YY, levels = levels, cmap = 'rainbow')
    ax.clabel(CS, inline=True, fontsize=12, fmt="%.2f")
    ax.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlim(1 - 0.5, 6 + 0.5)
    plt.ylim(1 - 0.5, 6 + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis="y", direction='in', length=8)
    ax.tick_params(axis="x", direction='in', length=8)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.title(column)
    return

#%% prepare data
X1_array = np.arange(1, 6 + 1)
X2_array = np.arange(1, 6 + 1)
XX1, XX2 = np.meshgrid(X1_array, X2_array)

X1_fine_array = np.linspace(0.5, 6.5, 100)
X2_fine_array = np.linspace(0.5, 6.5, 100)
XX1_fine, XX2_fine = np.meshgrid(X1_fine_array, X2_fine_array)

XX1_ = XX1.ravel()
XX2_ = XX2.ravel()
df = pd.DataFrame(np.column_stack((XX1_, XX2_)), columns = ['X1', 'X2'])

#%% X1 only
df['X1_sq'] = df['X1'] ** 2
YY_X1_only = XX1_fine
plot_contour(XX1, XX2, df, 'X1', XX1_fine, XX2_fine, YY_X1_only)
plot_stem(df, 'X1')

#%% X1 only squared
df['X1_sq'] = df['X1'] ** 2
YY_X1_sq = XX1_fine ** 2

plot_contour(XX1, XX2, df, 'X1_sq', XX1_fine, XX2_fine, YY_X1_sq)
plot_stem(df, 'X1_sq')

#%% sum: (X1 + X2)
df['sum'] = (df['X1'] + df['X2'])
YY_sum = XX1_fine + XX2_fine

plot_contour(XX1, XX2, df, 'sum', XX1_fine, XX2_fine, YY_sum)
plot_stem(df, 'sum')

#%% mean： (X1 + X2)/2
df['mean'] = (df['X1'] + df['X2'])/2
YY_mean = (XX1_fine + XX2_fine)/2

plot_contour(XX1, XX2, df, 'mean', XX1_fine, XX2_fine, YY_mean)
plot_stem(df, 'mean')

#%% mean： (X1 + X2 - 7)/2
df['mean_centered'] = (df['X1'] + df['X2'] - 7)/2
YY_mean_centered = (XX1_fine + XX2_fine - 7)/2

plot_contour(XX1, XX2, df, 'mean_centered', XX1_fine, XX2_fine, YY_mean_centered)
plot_stem(df, 'mean_centered')

#%% product of X1 and X2
df['product'] = df['X1'] * df['X2']
YY_product = XX1_fine * XX2_fine

plot_contour(XX1,XX2, df, 'product', XX1_fine, XX2_fine, YY_product)
plot_stem(df, 'product')

#%% devision, X1 over X2
df['devision'] = df['X1']/df['X2']
YY_devision = XX1_fine/XX2_fine

plot_contour(XX1,XX2, df, 'devision', XX1_fine, XX2_fine, YY_devision)
plot_stem(df, 'devision')

#%% difference
df['difference'] = df['X1'] - df['X2']
YY_difference = XX1_fine - XX2_fine
plot_contour(XX1,XX2, df, 'difference', XX1_fine, XX2_fine, YY_difference)
plot_stem(df, 'difference')

#%% abs_difference
df['abs_difference'] = np.abs(df['X1'] - df['X2'])
YY_abs_difference = np.abs(XX1_fine - XX2_fine)
plot_contour(XX1,XX2, df, 'abs_difference', XX1_fine, XX2_fine, YY_abs_difference)
plot_stem(df, 'abs_difference')

#%% (X1 - 3)**2 + (X2 - 3.5)**2
df['circle'] = (df['X1'] - 3.5) ** 2 + (df['X2'] - 3.5) ** 2
YY_circle = (XX1_fine - 3.5) ** 2 + (XX2_fine - 3.5) ** 2
plot_contour(XX1,XX2, df, 'circle', XX1_fine, XX2_fine, YY_circle)
plot_stem(df, 'circle')







#%% Bk5_Ch04_02.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
pd.options.mode.chained_assignment = None  # default='warn'

# Load the iris data
X_df = sns.load_dataset("iris")

#%% self-defined function

def heatmap_sum(data, i_array, j_array, title, vmin, vmax, cmap):
    fig, ax = plt.subplots(figsize=(10, 10))
    #'YlGnBu', # YlGnBu
    ax = sns.heatmap(data, cmap = cmap, cbar_kws = {"orientation": "horizontal"}, yticklabels=i_array, xticklabels=j_array, ax = ax, annot = True, linewidths=0.25, linecolor='grey', vmin = vmin, vmax = vmax)

    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')

    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0)

#%% Prepare data
X_df.sepal_length = round(X_df.sepal_length*2)/2
X_df.sepal_width  = round(X_df.sepal_width*2)/2

# print(X_df.sepal_length.unique())
sepal_length_array = X_df.sepal_length.unique()
sepal_length_array = np.sort(sepal_length_array)
# print(X_df.sepal_length.nunique())

# print(X_df.sepal_width.unique())
sepal_width_array = X_df.sepal_width.unique()
sepal_width_array = np.sort(sepal_width_array)
# sepal_width_array = np.flip(sepal_width_array)
# print(X_df.sepal_width.nunique())

#%% scatter plot, 图 28. “离散”的鸢尾花花萼长度、花萼宽度散点图
fig, ax = plt.subplots()
# scatter plot of iris data
ax = sns.scatterplot(data = X_df, x = 'sepal_length', y = 'sepal_width')

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

#%% frequency count,
X_df_12 = X_df[['sepal_length','sepal_width']]
X_df_12['count'] = 1
frequency_matrix = X_df_12.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix.columns = frequency_matrix.columns.droplevel(0)
frequency_matrix = frequency_matrix.fillna(0)
frequency_matrix = frequency_matrix.iloc[::-1]

probability_matrix = frequency_matrix/150

#%% 图 29. 频数和概率热图，全部样本点，不考虑分类
title = 'No species label, frequency'
heatmap_sum(frequency_matrix, sepal_width_array, sepal_length_array,title, 0, 50, 'plasma_r')

title = 'No species label, probability'
heatmap_sum(probability_matrix, sepal_width_array, sepal_length_array,title,0, 0.4, 'viridis_r')

#%% 图 30. 花萼长度的边缘频数和概率热图，不考虑分类

freq_sepal_length = frequency_matrix.sum(axis = 0).to_numpy().reshape((1,-1))
prob_sepal_length = probability_matrix.sum(axis = 0).to_numpy().reshape((1,-1))

title = 'Marginal count, frequency, sepal length'
heatmap_sum(freq_sepal_length,[],sepal_length_array, title, 0, 50,'plasma_r')

title = 'Marginal count, probability, sepal length'
heatmap_sum(prob_sepal_length,[],sepal_length_array, title, 0, 0.4,'viridis_r')

#%% 期望值 of X1
E_X1 = prob_sepal_length @ sepal_length_array.reshape(-1,1)
E_X1_ = X_df['sepal_length'].mean() # test only

#%% 方差 of X1
E_X1_sq = prob_sepal_length @ (sepal_length_array**2).reshape(-1,1)
var_X1 = E_X1_sq - E_X1**2
var_X1_ = X_df['sepal_length'].var()*149/150 # test only


#%% Marginal, sepal width, X2, 图 31. 花萼宽度的边缘频数和概率热图，不考虑分类
freq_sepal_width = frequency_matrix.sum(axis = 1).to_numpy().reshape((-1,1))
prob_sepal_width = probability_matrix.sum(axis = 1).to_numpy().reshape((-1,1))

title = 'Marginal count, frequency, sepal width'
heatmap_sum(freq_sepal_width,sepal_width_array,[],title,0,50,'plasma_r')

title = 'Marginal count, probability, sepal width'
heatmap_sum(prob_sepal_width,sepal_width_array,[],title,0,0.4,'viridis_r')


#%% Expectation of X2
E_X2 = sepal_width_array.reshape(1,-1) @ prob_sepal_width
E_X2_ = X_df['sepal_width'].mean() # test only



#%% assumption: independence, 图 32. 联合概率，假设独立
title = 'Assumption: independence'
# joint probability
heatmap_sum(prob_sepal_width@prob_sepal_length, sepal_width_array, sepal_length_array, title,0,0.4,'viridis_r')


#%% conditional probability, given sepal length, 给定花萼长度，花萼宽度的条件概率
given_sepal_length = 5

prob_given_length = probability_matrix[given_sepal_length]
prob_given_length = prob_given_length/prob_given_length.sum()
prob_given_length = prob_given_length.to_frame()
title = 'No species label, probability given sepal length'
heatmap_sum(prob_given_length,sepal_width_array,[],title,0,0.4,'viridis_r')

#%% Matrix, 图 36. 给定花萼长度，花萼宽度的条件概率 pX2 | X1(x2 | x1)
probability_matrix_ = probability_matrix.to_numpy()
conditional_X2_given_X1_matrix = probability_matrix_/(np.ones((6,1))@np.array([probability_matrix_.sum(axis = 0)]))

title = 'X2 given X1'
heatmap_sum(conditional_X2_given_X1_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

#%% conditional probability, given sepal width, 给定花萼宽度，花萼长度的条件概率 pX1 | X2(x1 | x2)
given_sepal_width = 2.5

prob_given_width = probability_matrix.loc[given_sepal_width,:]
prob_given_width = prob_given_width/prob_given_width.sum()
prob_given_width = prob_given_width.to_frame().T
title = 'No species label, probability given sepal width'
heatmap_sum(prob_given_width,[],sepal_length_array,title,0,0.4,'viridis_r')


#%% Matrix, 图 39. 给定花萼宽度，花萼长度的条件概率 pX1 | X2(x1 | x2)
conditional_X1_given_X2_matrix = probability_matrix_/(probability_matrix_.sum(axis = 1).reshape(-1,1)@np.ones((1,8)))

title = 'X1 given X2'
heatmap_sum(conditional_X1_given_X2_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')


#%% Given Y
Given_Y = 'virginica' # 'setosa', 'versicolor', 'virginica'

fig, ax = plt.subplots()

# scatter plot of iris data
ax = sns.scatterplot(data = X_df.loc[X_df.species == Given_Y], x = 'sepal_length', y = 'sepal_width')

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=0.5))
ax.set_yticks(np.arange(0, 6 + 1, step=0.5))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

# joint probability of X1, X2 given Y
X_df_12_given_Y = X_df[['sepal_length','sepal_width','species']]
X_df_12_given_Y['count'] = 1

X_df_12_given_Y.loc[~(X_df_12_given_Y.species == Given_Y),'count'] = np.nan

X_df_12_given_Y = X_df_12_given_Y[['sepal_length','sepal_width','count']]
frequency_matrix_given_Y = X_df_12_given_Y.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix_given_Y.columns = frequency_matrix_given_Y.columns.droplevel(0)
frequency_matrix_given_Y = frequency_matrix_given_Y.fillna(0)
frequency_matrix_given_Y = frequency_matrix_given_Y.iloc[::-1]

probability_matrix_given_Y = frequency_matrix_given_Y/frequency_matrix_given_Y.sum().sum()

# 图 43. 频数和条件概率 pX1,X2 | Y(x1, x2 | y = C3) 热图，给定分类标签 Y = C3 (virginica)
title = 'Given Y, frequency'
heatmap_sum(frequency_matrix_given_Y,sepal_width_array,sepal_length_array,title,0,50,'plasma_r')

title = 'Given Y, probability'
heatmap_sum(probability_matrix_given_Y,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

# Conditional Marginal, sepal length
freq_sepal_length_given_Y = frequency_matrix_given_Y.sum(axis = 0).to_numpy().reshape((1,-1))
prob_sepal_length_given_Y = probability_matrix_given_Y.sum(axis = 0).to_numpy().reshape((1,-1))

title = 'Conditional Marginal count, frequency, sepal length'
heatmap_sum(freq_sepal_length_given_Y,[],sepal_length_array,title,0,50,'plasma_r')

title = 'Conditional Marginal count, probability, sepal length'
heatmap_sum(prob_sepal_length_given_Y,[],sepal_length_array,title,0,0.4,'viridis_r')


# Conditional Marginal, sepal width
freq_sepal_width_given_Y = frequency_matrix_given_Y.sum(axis = 1).to_numpy().reshape((-1,1))
prob_sepal_width_given_Y = probability_matrix_given_Y.sum(axis = 1).to_numpy().reshape((-1,1))

title = 'Conditional Marginal count, frequency, sepal width'
heatmap_sum(freq_sepal_width_given_Y,sepal_width_array,[],title,0,50,'plasma_r')

title = 'Conditional Marginal count, probability, sepal width'
heatmap_sum(prob_sepal_width_given_Y,sepal_width_array,[],title,0,0.4,'viridis_r')

# conditional independence
title = 'Assumption: conditional independence'
heatmap_sum(prob_sepal_width_given_Y@prob_sepal_length_given_Y,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')



























































































































































































































































































