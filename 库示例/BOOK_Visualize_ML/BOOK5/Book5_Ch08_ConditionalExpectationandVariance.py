





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
pd.options.mode.chained_assignment = None  # default='warn'


### self-defined function
def heatmap_sum(data, i_array, j_array, title, vmin, vmax, cmap):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(data, cmap= cmap, cbar_kws={"orientation": "horizontal"}, yticklabels=i_array, xticklabels=j_array, ax = ax, annot = True, linewidths=0.25, linecolor='grey', vmin = vmin, vmax = vmax)
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0)

### Load the iris data
X_df = sns.load_dataset("iris")


### Prepare data
X_df.sepal_length = round(X_df.sepal_length*2)/2
X_df.sepal_width  = round(X_df.sepal_width*2)/2

print(X_df.sepal_length.unique())
sepal_length_array = X_df.sepal_length.unique()
sepal_length_array = np.sort(sepal_length_array)
print(X_df.sepal_length.nunique())

print(X_df.sepal_width.unique())
sepal_width_array = X_df.sepal_width.unique()
sepal_width_array = -np.sort(-sepal_width_array)
# sepal_width_array = np.flip(sepal_width_array)
print(X_df.sepal_width.nunique())

X_df_12 = X_df[['sepal_length','sepal_width']]
X_df_12['count'] = 1
frequency_matrix = X_df_12.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix.columns = frequency_matrix.columns.droplevel(0)
frequency_matrix = frequency_matrix.fillna(0)
frequency_matrix = frequency_matrix.iloc[::-1]

probability_matrix = frequency_matrix/150
probability_matrix_ = probability_matrix.to_numpy()

X1_array = np.sort(X_df.sepal_length.unique()).reshape(1,-1)
X2_array = np.sort(X_df.sepal_width.unique())[::-1].reshape(1,-1)

### marginal distributions
marginal_X1 = probability_matrix.sum(axis = 0).to_numpy().reshape((1,-1))
marginal_X2 = probability_matrix.sum(axis = 1).to_numpy().reshape((-1,1))

# marginal_X2.sum() # test only
# marginal_X1.sum() # test only

#%% conditional PMF X2 given X1
# 图 8. 给定花萼长度 X1，花萼宽度 X2的条件概率 pX2 | X1(x2 | x1) 热图
conditional_X2_given_X1_matrix = probability_matrix_/(np.ones((6,1))@np.array([probability_matrix_.sum(axis = 0)]))
title = '$p_{X_2 | X_1}(x_2 | x_1)$'
heatmap_sum(conditional_X2_given_X1_matrix, sepal_width_array, sepal_length_array, title, 0, 0.4, 'viridis_r')

#%% X2 * X2 given X1
# 图 8. 给定花萼长度 X1，花萼宽度 X2的条件概率 pX2 | X1(x2 | x1) 热图，x2 × pX2 | X1(x2 | x1) 热图
title = '$x_2$ * $p_{X_2 | X_1}(x_2 | x_1)$'
heatmap_sum(X2_array.T * conditional_X2_given_X1_matrix, sepal_width_array, sepal_length_array, title, 0, 3, 'RdYlBu_r')

#%% Conditional Expectation, X2 given X1
# 图 9. 矩阵乘法视角看条件期望 E(X2 | X1 = x1)
E_X2_given_X1 = X2_array@conditional_X2_given_X1_matrix
# 图 10. 给定花萼长度 X1，花萼宽度 X2的条件期望 E(X2 | X1 = x1)，和边缘 PMF pX1(x1)
fig, ax = plt.subplots(figsize=(10, 8))
ax.stem(X1_array.T, E_X2_given_X1.T, )
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$E(X_2 | X_1 = x_1)$')
ax.set_xlim(4.5, 8.0)
ax.set_ylim(0, 4)
plt.grid()
plt.show()

#%% Law of total expectation, 根据 (5) 的全期望定理
E_X2 = E_X2_given_X1 @ marginal_X1.T
E_X2_ = X_df_12['sepal_width'].mean()

#%% X2 sq * X2 given X1
# 图 8. 给定花萼长度 X1，花萼宽度 X2的条件概率 pX2|X1(x2|x1) 热图, x_2^2*px2|x1(x2|x1) 热图
title = '$x_2^2$ * $p_{X_2 | X_1}(x_2 | x_1)$'
heatmap_sum((X2_array**2).T * conditional_X2_given_X1_matrix, sepal_width_array,sepal_length_array,title,0,16,'RdYlBu_r')

#%% Conditional variance, X2 ^ 2 given X1
E_X2_sq_given_X1 = (X2_array**2)@conditional_X2_given_X1_matrix
# 图 11. 给定花萼长度 X1，花萼宽度平方值 X2^2的条件期望 E(X_2^2|X1 = x)
fig, ax = plt.subplots(figsize=(10, 8))
ax.stem(X1_array.T, E_X2_sq_given_X1.T,  )
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$E(X_2^2 | X_1 = x_1)$')
ax.set_xlim(4.5, 8.0)
ax.set_ylim(0, 16)
plt.grid()
plt.show()

#%% Conditional Variance, X2 given X1
var_X2_given_X1 = E_X2_sq_given_X1 - E_X2_given_X1**2
# 图 12. 给定花萼长度 X1，花萼宽度的条件方差 var(X2 | X1 = x1)
fig, ax = plt.subplots(figsize=(10, 8))
plt.stem(X1_array.T, var_X2_given_X1.T, )
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$var(X_2 | X_1 = x_1)$')
ax.set_xlim(4.5, 8.0)
ax.set_ylim(0, 0.35)
plt.grid()
plt.show()

#%% law of total variance, 全方差定理
import pandas as pd

# part A: expectation of conditional variances
E_var_X2_given_X1 = var_X2_given_X1 @ marginal_X1.T
E_var_X2_given_X1_each = var_X2_given_X1*marginal_X1

# part B: variance of conditional expectation
var_E_X2_sq_given_X1 = (E_X2_given_X1 - E_X2)**2 @ marginal_X1.T

var_X2_total = E_var_X2_given_X1 + var_E_X2_sq_given_X1
var_X2 = np.var(X_df_12['sepal_width'])


index = ['X1 = ' + str(x1) for x1 in X1_array[0]]
index.append('var of E_X2_given_X1')

var_X2_drill_down_df = pd.DataFrame({'var':np.append(E_var_X2_given_X1_each, var_E_X2_sq_given_X1)}, index = index)

#%% 图 13. 各个不同成分对花萼宽度 X2的方差 var(X2) 的贡献
var_X2_drill_down_df.sort_values('var',inplace=True)
var_X2_drill_down_df.plot.pie(y = 'var', autopct='%1.1f%%', legend = False, cmap='rainbow_r')
var_X2_drill_down_df.plot.barh(y = 'var')

#%% Conditional standard deviation, X2 given X1
# 图 14. 给定花萼长度 X1，花萼宽度 X2的条件期望 E(X2 | X1 = x1) ± std(X2 | X1 = x1)
std_X2_given_X1 = np.sqrt(var_X2_given_X1)
fig, ax = plt.subplots()
ax.errorbar(X1_array[0], E_X2_given_X1[0], yerr=std_X2_given_X1[0], capsize = 5, fmt='--o')

ax.set_xlim(4.5, 8.0)
ax.set_ylim(0, 4)
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$\mu_{X_2 | X_1 = x_1} \pm \sigma_{X_2 | X_1 = x_1} $')
plt.grid()
plt.show()

#%% conditional PMF X1 given X2
# 图 15. 给定花萼宽度，花萼长度的条件概率 pX1 | X2(x1 | x2)
conditional_X1_given_X2_matrix = probability_matrix_/(probability_matrix_.sum(axis = 1).reshape(-1,1)@np.ones((1,8)))
title = '$p_{X_1 | X_2}(x_1 | x_2)$'
heatmap_sum(conditional_X1_given_X2_matrix,sepal_width_array,sepal_length_array,title,0,0.4,'viridis_r')

#%% X1 * X1 given X2
# 图 15. 给定花萼宽度，花萼长度的条件概率 pX1 | X2(x1 | x2)
title = '$x_1$ * $p_{X_1 | X_2}(x_1 | x_2)$'
heatmap_sum(X1_array * conditional_X1_given_X2_matrix,sepal_width_array,sepal_length_array,title,0,5.5,'RdYlBu_r')

#%% Conditional Expectation, X1 given X2
# 图 16. 给定花萼宽度 X2，花萼宽度的条件期望 E(X1 | X2 = x2)
E_X1_given_X2 = conditional_X1_given_X2_matrix@X1_array.T

fig, ax = plt.subplots(figsize=(10, 8))
ax.stem(X2_array.T, E_X1_given_X2, )
ax.set_xlim(2, 4.5)
ax.set_ylim(0, 6.5)
ax.set_xlabel('Sepal width, $x_2$')
ax.set_ylabel('$E(X_1 | X_2 = x_2)$')
ax.grid()
plt.show()

#%% X1 sq * X1 given X2
# 图 15. 给定花萼宽度，花萼长度的条件概率 pX1 | X2(x1 | x2)
title = '$x_1^2$ * $p_{X_1 | X_2}(x_1 | x_2)$'
heatmap_sum((X1_array**2) * conditional_X1_given_X2_matrix, sepal_width_array, sepal_length_array, title, 0, 30, 'RdYlBu_r')

#%% Conditional variance, X1 ^ 2 given X2
# 图 17. 给定花萼长度 X2，花萼宽度平方值 X1^2的条件期望 E(X1^2|X_2 = x2)
E_X1_sq_given_X2 = conditional_X1_given_X2_matrix @ (X1_array**2).T

fig, ax = plt.subplots(figsize=(10, 8))
ax.stem(X2_array.T, E_X1_sq_given_X2, )
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$E(X_1^2 | X_2 = x_2)$')

ax.set_xlim(2, 4.5)
ax.set_ylim(0, 40)
plt.grid()
plt.show()

#%%  Conditional Variance, X1 given X2
# 图 18. 给定花萼宽度 X2，花萼宽度的条件方差 var(X1 | X2 = x2)
var_X1_given_X2 = E_X1_sq_given_X2 - E_X1_given_X2**2

fig, ax = plt.subplots(figsize=(10, 8))
ax.stem(X2_array.T, var_X1_given_X2, )
ax.set_ylabel('$var(X_1 | X_2 = x_2)$')
ax.set_xlim(2, 4.5)
ax.set_ylim(0, 1)
plt.grid()
plt.show()

#%% Conditional standard deviation, X1 given X2
# 图 19. 给定花萼宽度 X2，花萼长度 X1的条件期望 E(X1 | X2 = x2) ± std(X1 | X2 = x2)
std_X1_given_X2 = np.sqrt(var_X1_given_X2)
fig, ax = plt.subplots()
ax.errorbar(X2_array[0], E_X1_given_X2.T[0], yerr=std_X1_given_X2.T[0], capsize = 5, fmt='--o')
ax.set_xlim(2, 4.5)
ax.set_ylim(0, 7)
ax.set_xlabel('Sepal length, $x_1$')
ax.set_ylabel('$\mu_{X_1 | X_2 = x_2} \pm \sigma_{X_1 | X_2 = x_2} $')
plt.grid()
plt.show()

#%% Given Y, iris class label
# 图 20. 给定鸢尾花标签 Y，花萼长度的条件 PMF
# 图 21. 给定鸢尾花标签 Y，花萼宽度的条件 PMF
Y_array = ['setosa', 'versicolor', 'virginica']
# iris class labels

for Given_Y in Y_array:
    print('====================')
    print('Iris class label:')
    print(Given_Y)

    X_df_12_given_Y = X_df[['sepal_length','sepal_width','species']]
    X_df_12_given_Y['count'] = 1
    X_df_12_given_Y.loc[~(X_df_12_given_Y.species == Given_Y),'count'] = np.nan
    X_df_12_given_Y = X_df_12_given_Y[['sepal_length','sepal_width','count']]
    frequency_matrix_given_Y = X_df_12_given_Y.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
    frequency_matrix_given_Y.columns = frequency_matrix_given_Y.columns.droplevel(0)
    frequency_matrix_given_Y = frequency_matrix_given_Y.fillna(0)
    frequency_matrix_given_Y = frequency_matrix_given_Y.iloc[::-1]

    probability_matrix_given_Y = frequency_matrix_given_Y/frequency_matrix_given_Y.sum().sum()

    # Conditional Marginal, sepal length X1 given Y
    prob_sepal_length_given_Y = probability_matrix_given_Y.sum(axis = 0).to_numpy().reshape((1,-1))

    title = 'Conditional Marginal count, probability, sepal length'
    heatmap_sum(prob_sepal_length_given_Y, [], sepal_length_array, title, 0, 0.4, 'viridis_r')

    E_X1_given_Y = prob_sepal_length_given_Y@sepal_length_array.reshape(-1,1)
    print('E_X1_given_Y: ' + str(E_X1_given_Y))

    E_X1_sq_given_Y = prob_sepal_length_given_Y@(sepal_length_array**2).reshape(-1,1)
    print('E_X1_sq_given_Y: ' + str(E_X1_sq_given_Y))

    var_X1_given_Y = E_X1_sq_given_Y - E_X1_given_Y**2
    print('var_X1_given_Y: ' + str(var_X1_given_Y))

    std_X1_given_Y = np.sqrt(var_X1_given_Y)
    print('std_X1_given_Y: ' + str(std_X1_given_Y))

    # Conditional Marginal, sepal width X2 given Y
    prob_sepal_width_given_Y = probability_matrix_given_Y.sum(axis = 1).to_numpy().reshape((-1,1))

    title = 'Conditional Marginal count, probability, sepal width'
    heatmap_sum(prob_sepal_width_given_Y,sepal_width_array,[],title,0,0.4,'viridis_r')

    E_X2_given_Y = sepal_width_array.reshape(1,-1) @ prob_sepal_width_given_Y
    print('E_X2_given_Y: ' + str(E_X2_given_Y))

    E_X2_sq_given_Y = (sepal_width_array**2).reshape(1,-1) @ prob_sepal_width_given_Y
    print('E_X2_sq_given_Y: ' + str(E_X2_sq_given_Y))

    var_X2_given_Y = E_X2_sq_given_Y - E_X2_given_Y**2
    print('var_X2_given_Y: ' + str(var_X2_given_Y))

    std_X2_given_Y = np.sqrt(var_X2_given_Y)
    print('std_X2_given_Y: ' + str(std_X2_given_Y))

    print('====================')



































































































































































































































































































