





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pandas数据帧索引和切片

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 导入数据
# iris_df_original = sns.load_dataset("iris")
# 从Seaborn中导入鸢尾花数据帧
from sklearn.datasets import load_iris

# 鸢尾花数据
# iris_df = sns.load_dataset("./seaborn-data/iris")
iris_df_original = pd.read_csv("/home/jack/seaborn-data/iris.csv")
# 从Seaborn中导入鸢尾花数据帧
# iris_df.to_csv('iris_df.csv')


iris_df_original.columns

iris_df = iris_df_original.copy()
iris_df = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_df = iris_df.rename(columns={'sepal_length': 'X1', 'sepal_width':  'X2', 'petal_length': 'X3', 'petal_width':  'X4'})


############################## 提取列
# 取出某一列
##### 结果为数据帧
iris_df[['X1']]

iris_df.loc[:,['X1']]

iris_df.iloc[:,[0]]

iris_df.columns == 'X1'


##### 结果为Series
iris_df.loc[:,'X1']
# 结果为Pandas series
iris_df['X1']

# 结果为Pandas series
iris_df.X1

iris_df.iloc[:,0]


###### 提取多列
# 连续
iris_df[['X1', 'X2', 'X3']]

iris_df.loc[:,['X1','X2','X3']]

iris_df.iloc[:,0:3]

iris_df.iloc[:,:3]


# 不连续
iris_df[['X1', 'X4']]

iris_df.loc[:,['X1', 'X4']]

iris_df.iloc[:,[0, 3]]

# iris_df['X1', 'X2']
# 会报错 KeyError

# 等间隔
iris_df.iloc[:,::2]



############# 提取行
iris_df_ = iris_df.rename(lambda x: f'idx_{x}')

# 提取结果为 Pandas Series
iris_df_.loc['idx_0']

iris_df_.iloc[0]


type(iris_df_.iloc[0]) #  <class 'pandas.core.series.Series'>

# 提取结果为 Pandas DataFrame
iris_df_.iloc[[0]]

iris_df_.loc[['idx_0']]

# 布尔索引
condition = (iris_df.index == 1)
iris_df[condition]

############  提取几行
# 连续
iris_df_.iloc[[0,1,2]]

iris_df_.iloc[0:3]
# 不包含整数位置索引3，只选取位置0、1、2的行

iris_df_.iloc[:3]
# 不包含整数位置索引3，只选取位置0、1、2的行


iris_df_.loc[['idx_0', 'idx_1', 'idx_2']]


# 不连续
iris_df_.iloc[[0,50,100]]

iris_df_.loc[['idx_0','idx_50','idx_100']]


# 提取元素
# 某一元素
iris_df.iloc[::10,[0]]

iris_df.iloc[1:5, 2:4]

iris_df.iloc[[1, 3, 5], [1, 3]]

iris_df.iloc[1:3, :]

iris_df.iloc[:, 1:3]

iris_df.iloc[1,1]

######### 使用at、iat
# 元素
iris_df_.head()
iris_df_.at['idx_0', 'X1']
iris_df_.iat[0, 0]

iris_df_.loc['idx_0', 'X1']

iris_df_.iloc[0,0]

# Dataframe
iris_df_.iloc[[0],[0]]

iris_df_.loc[['idx_0'], ['X1']]



# 条件索引

Boolean_df = (iris_df > 6) | (iris_df < 1.5)
Boolean_df

fig,ax = plt.subplots(figsize = (3,5))
sns.heatmap(Boolean_df, cmap = 'Blues', ax = ax, cbar=False, xticklabels = [], yticklabels = [], cbar_kws = {'orientation':'vertical'}, annot=False)

# fig.savefig('鸢尾花数据dataframe_布尔条件.svg', format='svg')

# 取出满足条件的行
condition = iris_df['X1'] >= 7
iris_df[condition]

iris_df[condition].shape

fig,ax = plt.subplots(figsize = (6,8))
sns.heatmap(iris_df[condition], cmap = 'RdYlBu_r', ax = ax, vmax = 0, vmin = 8, cbar_kws = {'orientation':'vertical'}, annot=False)

# fig.savefig('鸢尾花数据dataframe_满足条件的行.svg', format='svg')

iris_df.loc[:,'X1'] >= 7

iris_df.loc[iris_df.loc[:,'X1'] >= 7, :]

iris_df_original.loc[iris_df_original.loc[:,'species'] == 'versicolor', :]

iris_df_original.loc[iris_df_original.species == 'versicolor', :]

# 或，非
iris_df_original.loc[(iris_df_original.sepal_length < 6.5) & (iris_df_original.sepal_length > 6)]

iris_df_original.loc[(iris_df_original.sepal_length < 6.5) & (iris_df_original.sepal_length > 6), ['petal_length', 'petal_width']]

iris_df_original.loc[iris_df_original['species'] != 'virginica']

iris_df_original.loc[iris_df_original['species'].isin(['virginica','setosa'])]

iris_df_original.loc[~iris_df_original['species'].isin(['virginica','setosa']), ['petal_length', 'petal_width']]

iris_df_original.loc[iris_df_original.species.isin(['setosa', 'virginica'])]


# query()
iris_df_original.query('sepal_length > 2*sepal_width')

iris_df_original.query("petal_length + petal_width < sepal_length + sepal_width")

iris_df_original.query("species == 'versicolor'")
iris_df_original.query("not (sepal_length > 7 and petal_width > 0.5)")


iris_df_original.query("species != 'versicolor'")


iris_df_original.query("abs(sepal_length-6) > 1")

iris_df_original.query("species in ('versicolor','virginica')")


iris_df_original.query("sepal_length >= 6.5 or sepal_length <= 4.5")

iris_df_original.query("sepal_length <= 6.5 and sepal_length >= 4.5")


################################### 多层索引
# 多层行
import pandas as pd
import numpy as np
# 示例数据
index_arrays = [['A','A','B','B','C','C','D','D'], range(1,9)]
data = np.random.randint(0,9,size=(8,4))

# 创建多层行索引
multi_row_idx = pd.MultiIndex.from_arrays(index_arrays, names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, index=multi_row_idx, columns=['X1','X2','X3','X4'])
df

multi_row_idx

df.loc[('A', 1):('C',1)]

df.loc[('A', 1)]
df.loc[[('A', 1)]]


df.loc[('A', 1),'X1']

df.loc['A']

df.reset_index()
df.index

list(df.index)


######### 两组列表
categories = ['A','B','C','D']
types = ['X', 'Y']

# 创建多层行索引
multi_index = pd.MultiIndex.from_product([categories, types], names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, index=multi_index, columns=['X1','X2','X3','X4'])
df

df.index = df.index.map('_'.join)
df


# 创建多层行索引
multi_index = pd.MultiIndex.from_product([types, categories], names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, index=multi_index, columns=['X1','X2','X3','X4'])
df
df.index = df.index.map('_'.join)
df


# index_tuples = [('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y')]

# # 创建多层行索引
# multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['I', 'II'])

# # 创建DataFrame
# df = pd.DataFrame(data, index=multi_index, columns=['X1','X2'])
# df

# Index_data = {
#     'Category': ['A', 'A', 'B', 'B'],
#     'Type': ['X', 'Y', 'X', 'Y']
# }
# df_index = pd.DataFrame(Index_data)


# # 创建多层行索引
# multi_index = pd.MultiIndex.from_frame(df_index, names=['I', 'II'])

# # 创建DataFrame
# df = pd.DataFrame(data, index=multi_index, columns=['X1','X2'])
# df

# 多层列
# 创建两层列标签列表
col_arrays = [['A', 'A', 'B', 'B'], ['X1', 'X2', 'X3', 'X4']]

# 创建两层列索引
multi_index = pd.MultiIndex.from_arrays(col_arrays, names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, columns=multi_index)
df

df.A.X1
# df.loc[:,'A'].X1
# df.loc[:,('A','X1')]
# df.loc[:,'A'].loc[:,'X1']

df.loc[:,'A']

df.loc[:,[('A','X1')]]

df.loc[:,'A'].loc[:,['X1']]


df.columns

df.columns = df.columns.map('_'.join)
df

########################### 示例数据 (1)
categories = ['A', 'B']
types = ['X', 'Y']

# 创建多层列索引
multi_index = pd.MultiIndex.from_product([categories, types], names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, columns=multi_index)
df

############################ 示例数据 (2 )
index_tuples = [('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y')]
# 创建多层列索引
multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, columns=multi_index)
df

############################# 示例数据 (3)
dict_index = {
    'Category': ['A', 'A', 'B', 'B'],
    'Type': ['X', 'Y', 'X', 'Y']
}
df_index = pd.DataFrame(dict_index)

# 创建多层列索引
multi_index = pd.MultiIndex.from_frame(df_index, names=['I', 'II'])

# 创建DataFrame
df = pd.DataFrame(data, columns=multi_index)
df


################################## 多层列、多层行
#
index_arrays = [['A','A','B','B','C','C','D','D'], range(1,9)]
# 创建多层行索引
multi_row = pd.MultiIndex.from_arrays(index_arrays, names=['I', 'II'])

# 创建两层列标签列表
col_arrays = [['A', 'A', 'B', 'B'], ['X1', 'X2', 'X3', 'X4']]

# 创建两层列索引
multi_col = pd.MultiIndex.from_arrays(col_arrays, names=['I', 'II'])

data = np.random.randint(0,9,size=(8,4))

# 创建DataFrame
df = pd.DataFrame(data, index=multi_row, columns=multi_col)
df


# 以下内容移动到第四章，本版块

# 鸢尾花数据为例
iris_df_multi_col = iris_df_original.copy()

index_arrays = [['Quant',  'Quant',  'Quant', 'Quant', 'Categ'],
                ['Sepal',  'Sepal',  'Petal', 'Petal', 'Species'],
                ['Length', 'Width', 'Length', 'Width', 'Species']]
# 创建多层行索引
iris_df_multi_col.columns = pd.MultiIndex.from_arrays(index_arrays, names=['I', 'II', 'III'])
iris_df_multi_col

iris_df_multi_col.columns

iris_df_multi_col.loc[:,'Categ']

iris_df_multi_col.loc[:,'Sepal'].loc[2,'Length']

iris_df_multi_col.loc[:,('Sepal','Length')]


iris_df_multi_row = iris_df_original.reset_index()

# 定义函数将花萼长度映射为等级
def map_sepal_length_to_category(sepal_length):
    if sepal_length < 5:
        return 'D'
    elif 5 <= sepal_length < 6:
        return 'C'
    elif 6 <= sepal_length < 7:
        return 'B'
    else:
        return 'A'

# 使用 apply 函数将 sepal_length 映射为等级并添加新列
iris_df_multi_row['category'] = iris_df_multi_row['sepal_length'].apply(map_sepal_length_to_category)

iris_df_multi_row

# 以 'species' 列为第一级索引，'category'为二级索引，'index' 列（之前的索引）为第三级索引
iris_df_multi_row.set_index(['species', 'category','index']) # , inplace=True


df = iris_df_multi_row.sort_values(by=['category', 'species']).set_index(['category', 'species', 'index']) # , inplace=True
df

df.reset_index()
df.index

df.loc[('A', 'versicolor',  50)]

df.loc[('A', 'versicolor',  50):('A',  'virginica', 129)]

df.loc['A']

df.loc['A'].loc['virginica']

# pip install pandas_datareader
# 时间序列indexing
# 创建时间序列
import pandas_datareader as pdr
import datetime

start_date = datetime.datetime(2020, 1, 1)
end_date   = datetime.datetime(2022, 12, 31)

ticker_list = ['SP500']
SP500_df = pdr.DataReader(ticker_list, 'fred', start_date, end_date)
SP500_df


SP500_df.loc['2022-06-01':]

SP500_df.query("DATE.dt.year > 2021")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用列表构造多层行标签

import pandas as pd
import numpy as np

# 创建列表、数据
index_arrays = [['A','A','B','B','C','C','D','D'], range(1,9)]
data = np.random.randint(0,9,size=(8,4))

# 创建多层行索引
row_idx = pd.MultiIndex.from_arrays(index_arrays, names=['I','II'])


# 创建DataFrame
df = pd.DataFrame(data, index=row_idx, columns=['X1','X2','X3','X4'])
df


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用多个可迭代对象的笛卡尔积构造多层行标签

import pandas as pd
import numpy as np


# 示例数据
data = np.random.randint(0,9,size=(8,4))


# 两组列表
categories = ['A','B','C','D']
types = ['X', 'Y']

# 创建多层行索引，先categories，再types
idx_1 = pd.MultiIndex.from_product([categories, types], names=['I', 'II'])
idx_1
df_1 = pd.DataFrame(data, index=idx_1, columns=['X1','X2','X3','X4'])
df_1

# 创建多层行索引，先types，再categories
idx_2 = pd.MultiIndex.from_product([types, categories], names=['I', 'II'])
idx_2
df_2 = pd.DataFrame(data, index=idx_2, columns=['X1','X2','X3','X4'])
df_2

# 将第0级索引的名称设置为 'Level_0_idx'
df_2.index.set_names('Level_0_idx', level=0, inplace=True)
df_2

# 将第1级索引的名称设置为 'Level_1_idx'
df_2.index.set_names('Level_1_idx', level=1, inplace=True)
df_2

# 获取 DataFrame 中多级索引的第0级别（level=0）的所有标签值
df_2.index.get_level_values(0)
# 获取 DataFrame 中多级索引的第1级别（level=1）的所有标签值
df_2.index.get_level_values(1)

df_2.xs('X', level='Level_0_idx')
# df_2.xs('X')
# 获取 Level_0_idx 等于 'X' 的所有行

df_2.xs('A', level='Level_1_idx')
# 获取 Level_1_idx 等于 'A' 的所有行

df_2.xs(('X', 'A'), level=['Level_0_idx','Level_1_idx'])
# df_2.xs(('X', 'A'))
# 获取 Level_0_idx 等于 'X' 且 Level_1_idx 等于 'A' 的所有行


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用列表构造多层列标签

# 示例数据
data = np.random.randint(0,9,size=(8,4))

# 创建两层列标签列表
col_arrays = [['A',  'A',  'B',  'B'],
              ['X1', 'X2', 'X3', 'X4']]


# 创建两层列索引
multi_col = pd.MultiIndex.from_arrays(col_arrays, names=['I','II'])


# 创建DataFrame
df = pd.DataFrame(data, columns=multi_col)
df


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 蒙特卡罗模拟时间序列切片

# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建日期范围，假设为2年（365*2天）
date_range = pd.date_range(start='2023-01-01', periods=365*2, freq='D')


# 创建一个空的 DataFrame，用于存储随机行走数据
df = pd.DataFrame(index=date_range)
df.head()

# 模拟50个随机行走
num_path = 50
# 设置随机种子以保证结果可重复
np.random.seed(0)
for i in range(num_path):
    # 生成随机步长，每天行走步长服从标准正态分布
    step_idx = np.random.normal(loc=0.0, scale=1.0, size=len(date_range) - 1)
    # 增加初始状态
    step_idx = np.append(0, step_idx)

    # 计算累积步数
    walk_idx = step_idx.cumsum()

    # 将行走路径存储在DataFrame中，列名为随机行走编号
    df[f'Walk_{i + 1}'] = walk_idx
# 请大家想办法去掉for循环

# 绘制所有随机行走轨迹
df.plot(legend = False)
plt.grid(True)
plt.ylim(-80,80)

# 绘制前两条随机行走
# df.iloc[:, [1, 0]].plot(legend = True)
df[['Walk_1', 'Walk_2']].plot(legend = True)
plt.grid(True)
plt.ylim(-80,80)

# 绘制前两条随机行走，特定时间段
df.loc['2023-01-01':'2023-08-08', ['Walk_1', 'Walk_2']].plot(legend = True)
# df.iloc[0:220, 0:2].plot(legend = True)
plt.grid(True)
plt.ylim(-80,80)



