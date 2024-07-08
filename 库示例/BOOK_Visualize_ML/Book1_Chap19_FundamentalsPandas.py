


# Bk1_Ch19_01


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# p = plt.rcParams
# p["font.sans-serif"] = ["Roboto"]
# p["font.weight"] = "light"
# p["ytick.minor.visible"] = True
# p["xtick.minor.visible"] = True
# p["axes.grid"] = True
# p["grid.color"] = "0.5"
# p["grid.linewidth"] = 0.5


# 创建数据帧
# 定义数据帧
# 从字典
# 采用默认行索引，Zero-based numbering
dict_eg = {'Positive integer': [1, 2, 3, 4, 5],
           'Greek letter': ['alpha', 'beta', 'gamma', 'delta', 'epsilon']}
df_from_dict = pd.DataFrame(data=dict_eg)
df_from_dict


df_from_dict2 = df_from_dict.set_index('Positive integer')
df_from_dict2


df_from_dict2.reset_index()


# 修改行索引
df_from_dict3 = pd.DataFrame(data=dict_eg, index = ['a', 'b', 'c', 'd', 'e'])
df_from_dict3


# 从列表
list_fruits = [['apple',  11],
               ['banana', 22],
               ['cherry', 33],
               ['durian', 44]]
df_from_list1 = pd.DataFrame(list_fruits)
df_from_list1

# 设定行索引
df_from_list1.set_axis(['a', 'b', 'c', 'd'], axis='index')

# 设定行标签
df_from_list1.set_axis(['Fruit', 'Number'], axis='columns')


# 改变row index
df_from_list2 = pd.DataFrame(list_fruits, columns=['Fruit', 'Number'], index = ['a', 'b', 'c', 'd'])
df_from_list2

# 将numpy数组转化为数据帧
numpy_array = np.random.normal(size = (10,4))
# NumPy库中的random.normal函数生成一个形状为(10, 4)的二维数组（矩阵），其中的元素是从正态分布（高斯分布）中随机抽取的数据。
df_from_np = pd.DataFrame(numpy_array, columns=['X1', 'X2', 'X3', 'X4'])
df_from_np

# for循环生成数据帧
np_data = []
# 创建一个空list
for idx in range(10):
    data_idx = np.random.normal(size = (1,4)).tolist()
    np_data.append(data_idx[0])

# 注意，用list.append() 速度相对较快
df_for_loop = pd.DataFrame(np_data, columns = ['X1','X2','X3','X4'])
df_for_loop




# 鸢尾花数据
iris_df = sns.load_dataset("iris")
# 从Seaborn中导入鸢尾花数据帧
# iris_df.to_csv('iris_df.csv')


fig,ax = plt.subplots(figsize = (6,8))
sns.heatmap(iris_df.iloc[:, 0:4],
            cmap = 'RdYlBu_r',
            ax = ax,
            vmax = 0, vmin = 8,
            cbar_kws = {'orientation':'vertical'},
            annot=False)

# fig.savefig('鸢尾花数据dataframe.svg', format='svg')


# 打印整个数据集
print(iris_df.to_string())


iris_df.to_numpy() # 将数据帧转化为NumPy array
# , dtype=object
# 在 NumPy 中，dtype=object 表示数组中的元素类型为 Python 对象。
# 这意味着数组的每个元素都可以是任意类型的 Python 对象，
# 例如整数、浮点数、字符串、列表、字典等。
# 与其他 NumPy 数组不同，dtype=object 数组允许每个元素具有不同的数据类型。

# 当你创建一个 NumPy 数组并指定 dtype=object，
# NumPy 将会把数组视为一个包含 Python 对象的数组，
# 而不是传统的数值类型数组。这种数组的灵活性较高，
# 但也会导致一些性能上的损失，
# 因为在处理数组时无法利用 NumPy 的优化和并行计算功能。



# 查询
# pandas.DataFrame.index
iris_df.index

row_index_list = list(iris_df.index)
# row_index_list


# pandas.DataFrame.columns
iris_df.columns

list(iris_df.columns)


# pandas.DataFrame.axes
iris_df.axes


# 判断数据类型
type(iris_df)



# pandas.DataFrame.values
iris_df.values


# pandas.DataFrame.describe
iris_df.describe()


# 小数点后一位
iris_df.describe().round(1)



# pandas.DataFrame.nunique
iris_df.nunique()



iris_df['species'].unique()


# 打印数据帧前5行
iris_df.head()


# 打印数据帧后5行
iris_df.tail()


# 形状
# 数据帧本质上就是一个表格
# 获取数据帧形状
# pandas.DataFrame.shape
iris_df.shape



# 获取表格元素总数
# pandas.DataFrame.size
iris_df.size

# 每一列非缺失值的数量
# pandas.DataFrame.count
iris_df.count()



print(iris_df.count(axis = 1))

iris_df.count() * 100 / len(iris_df)

# pandas.DataFrame.isnull
iris_df.isnull()

iris_df.isnull().sum() * 100 / len(iris_df)



# 获取数据帧行数，几种不同方法
print(iris_df.shape[0])
print(len(iris_df))
print(len(iris_df.index))
print(iris_df[iris_df.columns[0]].count())
num_rows = len(iris_df.axes[0])


# 获取数据帧列数，几种不同方法
print(iris_df.shape[1])
print(len(iris_df.T))
print(len(iris_df.columns))
print(len(iris_df.axes[1]))

# 循环
# iterate rows
for idx, row_idx in iris_df.iterrows():
    print('=================')
    print('Row index =',str(idx))
    print(row_idx['sepal_length'], row_idx['sepal_width'])


for column_idx in iris_df:
    print(column_idx)
    print(iris_df[column_idx])
for column_idx in iris_df.iteritems():
    print(column_idx)
for column_idx in iris_df.items():
    print(column_idx)


# 转置
iris_df.T

# 将数据帧转化为numpy array
iris_df_2_array = iris_df.to_numpy()


# 数据帧前四列转化为numpy array
iris_df_2_array_numeric = iris_df[iris_df.columns[:4]].to_numpy()
iris_df_2_array_numeric.shape


# 指定column名称，转化成numpy array
iris_df_2_array_sepal = iris_df[['sepal_length','sepal_width']].to_numpy()
iris_df_2_array_sepal


# 更改表头
iris_df.rename(columns={'sepal_length': 'X1',
                        'sepal_width':  'X2',
                        'petal_length': 'X3',
                        'petal_width':  'X4',
                        'species':      'Y'})
# 注意，函数输入增加 inplace=True，直接修改原数据帧表头

# 另外两种新方法：
iris_df.rename({'sepal_length': 'X1',
                'sepal_width':  'X2',
                'petal_length': 'X3',
                'petal_width':  'X4',
                'species':      'Y'},
               axis = 1)

iris_df.rename({'sepal_length': 'X1',
                'sepal_width':  'X2',
                'petal_length': 'X3',
                'petal_width':  'X4',
                'species':      'Y'},
               axis = 'columns')

# 加“根、缀”
iris_df_suffix = iris_df.add_suffix('_col')
iris_df_suffix.head()


# 修改行索引
iris_df.rename(lambda x: f'idx_{x}')
iris_df.rename(lambda x: f'{x}_idx')
iris_df_suffix.rename(columns = lambda x: x.strip('_col'))
iris_df_prefix = iris_df.add_prefix('col_').head()
iris_df_prefix

iris_df_prefix.rename(columns = lambda x: x.strip('col_'))



# 更改列顺序
# 当前列顺序
iris_df.columns.tolist()
# 顺序调转
new_col_order = iris_df.columns.tolist()[::-1]
new_col_order

iris_df[new_col_order]

# 自定义顺序
new_col_order = ['species',
                 'sepal_length', 'petal_length',
                 'sepal_width', 'petal_width']
iris_df[new_col_order]

iris_df.loc[:, new_col_order]

iris_df.iloc[:, [4,0,2,1,3]]

iris_df.set_axis(new_col_order, axis=1)


# 修改行顺序
# 取出前5行，并修改行索引
iris_df_ = iris_df.iloc[:5,:].rename(lambda x:
                                     f'idx_{x}')


new_order = ['idx_4','idx_2','idx_0','idx_3','idx_1']
iris_df_.reindex(new_order)


iris_df_.loc[new_order]


new_order_int = [4, 2, 0, 3, 1]
iris_df_.iloc[new_order_int]



iris_df_.sort_index(ascending=False)



# 删除特定行
iris_df.drop(index=[0,1])
# 将inplace参数设置为True，可以在原地修改DataFrame，而不返回一个新的DataFrame


# 删除特定列
iris_df.drop(columns='species')


# 视图 vs 副本
df = pd.DataFrame({'A': [1, 2], 'B': [11, 22]})

df_view = df[['A']]
df_view



# 时间序列
import pandas_datareader as pdr
import datetime

start_date = datetime.datetime(2014, 1, 1)
end_date   = datetime.datetime(2022, 12, 31)

ticker_list = ['SP500']
df = pdr.DataReader(ticker_list,
                    'fred',
                    start_date,
                    end_date)
# df.to_csv('SP500_' + str(start_date.date()) + '_' + str(end_date.date()) + '.csv')
# df.to_pickle('SP500_' + str(start_date.date()) + '_' + str(end_date.date()) + '.pkl')
# pandas.read_csv
# pandas.read_pickle


# 导入、导出文件
# 导出数据
# 将数据帧存成CSV文件
csv_filename = 'iris_df.csv'
iris_df.to_csv(csv_filename,
               index = False)


# 将数据帧存成Excel文件
xlsx_filename = 'iris_df.xlsx'
iris_df.to_excel(xlsx_filename,
                 sheet_name='iris_data')

# 导入文件
# 读入CSV文件
iris_df_from_CSV = pd.read_csv(csv_filename)
# 参考：
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

iris_df_from_CSV.head()

# 读入Excel文件
iris_df_from_Excel = pd.read_excel(xlsx_filename)
# 参考：
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

iris_df_from_Excel.head()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk1_Ch19_02
import seaborn as sns
import pandas as pd
import numpy as np


iris_df = sns.load_dataset("iris")
# 从Seaborn中导入鸢尾花数据帧




# 加减乘除
# Load the iris data
X_df = iris_df.copy()

X_df.rename(columns = {'sepal_length':'X1',
                       'sepal_width':'X2'},
            inplace = True)

X_df_ = X_df[['X1','X2', 'species']]

#%% data transformation

X_df_['X1 - E(X1)'] = X_df_['X1'] - X_df_['X1'].mean()
X_df_['X2 - E(X2)'] = X_df_['X2'] - X_df_['X2'].mean()
X_df_['X1 + X2'] = X_df_['X1'] + X_df_['X2']
X_df_['X1 - X2'] = X_df_['X1'] - X_df_['X2']
X_df_['X1 * X2'] = X_df_['X1'] * X_df_['X2']
X_df_['X1 / X2'] = X_df_['X1'] / X_df_['X2']
X_df_.drop(['X1','X2'], axis=1, inplace=True)

# 可视化

sns.pairplot(X_df_, corner=True, hue="species")



# 统计汇总
# 数据帧统计汇总
# 注意很多统计运算只针对数值，比如自动忽略'species'一列
iris_df.describe()



# 对于特定列的汇总
iris_df.sepal_length.describe()



# 等价于
iris_df['sepal_length'].describe()


# 最大、最小值
iris_df.max()
iris_df.min()


# 均值、中位数、众数
# 均值
iris_df.mean()

# 中位数
iris_df.median()

# 众数
iris_df.mode(numeric_only = True, dropna=True)


# 独特值
# 鸢尾花种类 'species' 一列独特值
iris_df['species'].unique()
# 常用来获得标签


# 将数组转化为列
iris_df['species'].unique().tolist()


# 鸢尾花种类 'species' 独特值数量
iris_df['species'].nunique()
# 常用来获得标签数量

# 排序
# 根据花萼长度排序，默认从小到大
iris_df.sort_values(['sepal_length'])


# 分位

iris_df.quantile([0.05, 0.95])


# 根据花萼长度排序，默认从大到小
iris_df.sort_values(['sepal_length'], ascending=False)



# 样本方差
iris_df.var()


# 样本标准差
iris_df.std()


# 协方差矩阵 注意，默认分母n - 1
iris_df.cov()

# 相关性系数矩阵
iris_df.corr()

iris_df.skew()
iris_df.kurt()
# 也可以用 iris_df.kurtosis()


iris_df_rounded = np.ceil(iris_df[iris_df.columns[:4]])
# 向上取整
iris_df_rounded.sepal_length.unique()

iris_df_rounded.sepal_width.unique()

iris_df_rounded.value_counts()

iris_df_rounded[['sepal_length','sepal_width']].value_counts()


iris_sepal = iris_df_rounded[['sepal_length','sepal_width']]
iris_sepal['count'] = 1



frequency_matrix =iris_sepal.groupby(['sepal_length','sepal_width']).count().unstack(level=0)
frequency_matrix


## 将连续转为分类
pd.cut(iris_df.sepal_length, bins = 3, labels = ['short', 'medium', 'long'])


# groupby
iris_df = sns.load_dataset("iris")
iris_df.groupby(['species']).mean()

iris_df.groupby(['species']).std()



iris_df.groupby(['species']).var()



three_cov_matrics = iris_df.groupby(['species']).cov()
three_cov_matrics


three_cov_matrics.index

three_cov_matrics.columns.get_level_values(0)


three_cov_matrics.index.get_level_values(0).unique().to_list()


three_cov_matrics.index.get_level_values('species').unique().to_list()

three_cov_matrics.index.get_level_values(1).unique().to_list()


three_cov_matrics.loc['setosa']


three_cov_matrics.xs('setosa')


iris_df.loc[iris_df['species'] == 'setosa'].cov()


# agg
iris_df.iloc[:,0:4].agg(['sum', 'min', 'max', 'std', 'var', 'mean', np.mean])


iris_df.iloc[:,0:3].agg(['sum', 'min', 'max', 'std', 'var', 'mean'], axis = 'columns')

# map
import pandas as pd
import seaborn as sns

# 加载鸢尾花数据集
iris = sns.load_dataset('iris')


# 定义映射函数
def map_sepal_length(sepal_length):
    if sepal_length < 5:
        return "短"
    elif 5 <= sepal_length < 6:
        return "中等"
    else:
        return "长"

# 使用 map 方法将花萼长度映射为分类值
iris['sepal_length_category'] = iris['sepal_length'].map(map_sepal_length)
iris


# 计算鸢尾花各类花瓣平均宽度
mean_X2_by_species = iris_df.groupby(
    'species')['petal_width'].mean()
mean_X2_by_species




# 定义映射函数
def map_petal_width(petal_width, species):
    if petal_width > mean_X2_by_species[species]:
        return "YES"
    else:
        return "NO"

# 使用 map 方法将花瓣宽度映射为是否超过平均值
iris_df['greater_than_mean'] = iris_df.apply(lambda
       row: map_petal_width(row['petal_width'],
                            row['species']), axis=1)

iris_df


# apply

import seaborn as sns
import pandas as pd

iris_df = sns.load_dataset("iris")
# 从Seaborn中导入鸢尾花数据帧
# 定义函数将花萼长度映射为等级
def sepal_length_to_category(sepal_length):
    if sepal_length < 5:
        return 'D'
    elif 5 <= sepal_length < 6:
        return 'C'
    elif 6 <= sepal_length < 7:
        return 'B'
    else:
        return 'A'

# 使用 apply 函数将 sepal_length 映射为等级并添加新列
iris_df['category'] = iris_df['sepal_length'].apply(sepal_length_to_category)
iris_df


# 使用apply和lambda函数计算每个类别中最小的花瓣宽度
iris_df.groupby('species')['sepal_length'].apply(
    lambda x: x.min())



iris_df.groupby('species')['sepal_length'].min()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 从CSV读入时间序列数据
# 导入包
import pandas as pd

# CSV 文件名称
csv_file_name = 'SP500_2014-01-01_2022-12-31.csv'

# 读入CSV文件
df = pd.read_csv(csv_file_name)



# 将输入的数据转换为日期时间对象
df["DATE"] = pd.to_datetime(df["DATE"])


# 将名为"DATE"的列设置为索引
df.set_index('DATE', inplace = True)
# 快速可视化
df.plot()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 从网页下载数据


import pandas as pd
import requests


# 设置数据源的URL
url = 'https://fred.stlouisfed.org/data/SP500.txt'


# 发送GET请求并获取数据
response = requests.get(url)


# 检查是否成功获取数据
if response.status_code == 200:
    # 数据以制表符分隔
    df = pd.read_csv(url,skiprows=44, sep='\s+')
else:
    print("Failed to fetch data from the source")



df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)




df.head()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 利用pandas_datareader从FRED下载数据


# 导入包
import pandas_datareader as pdr
# 需要安装 pip install pandas-datareader
import pandas as pd
import datetime




# 从FRED下载标普500 (S&P 500)
start_date = datetime.datetime(2014, 1, 1)
end_date   = datetime.datetime(2022, 12, 31)



# 下载数据
ticker_list = ['SP500']
df = pdr.DataReader(ticker_list, 'fred', start_date, end_date)
























































































































































































































































































































































































































































































































































































































































