

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html







# Pandas快速可视化


# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# pip install pandas_datareader
import seaborn as sns





# 下载数据
df = pdr.data.DataReader(['DGS6MO','DGS1',
                          'DGS2','DGS5',
                          'DGS7','DGS10',
                          'DGS20','DGS30'],
                          data_source='fred',
                          start='01-01-2022',
                          end='12-31-2022')
df.to_csv('IR_data.csv')
# 如果不能下载，请用pandas.read_csv() 读取数据
df = df.dropna()


# 修改列标签
df = df.rename(columns={'DGS6MO': '0.5 yr',
                        'DGS1': '1 yr',
                        'DGS2': '2 yr',
                        'DGS5': '5 yr',
                        'DGS7': '7 yr',
                        'DGS10': '10 yr',
                        'DGS20': '20 yr',
                        'DGS30': '30 yr'})



# 绘制利率走势线图
df.plot(xlabel="Time", ylabel="IR level",
        legend = True,
        xlim = (df.index.min(), df.index.max()))

# 美化线图

fig, ax = plt.subplots(figsize = (5,5))
df.plot(ax = ax, xlabel="Time", legend = True)
ax.set_xlim((df.index.min(), df.index.max()))
ax.set_ylim((0,5))
ax.set_xticks([])
ax.set_xlabel('Time')
ax.set_ylabel('IR level')



# 绘制利率走势线图，子图布置
df.plot(subplots=True, layout=(2,4),
        sharex = True, sharey = True,
        xticks = [],yticks =[],
        xlim = (df.index.min(), df.index.max()))




# 绘制利率走势线面积图，子图布置
df.plot.area(subplots=True, layout=(2,4),
             sharex = True, sharey = True,
             xticks = [],yticks =[],
             xlim = (df.index.min(), df.index.max()),
             ylim = (0,5), legend = False)


# 计算日收益率
r_df = df.pct_change()



# 绘制利率日收益率，子图布置
r_df.plot(subplots=True, layout=(2,4),
          sharex = True, sharey = True,
          xticks = [],yticks =[],
          xlim = (df.index.min(), df.index.max()))



# 绘制散点图
fig, ax = plt.subplots(figsize = (5,5))
r_df.plot.scatter(x="1 yr", y="2 yr",
                  ax = ax)

ax.set_xlim(-0.1, 0.25)
ax.set_ylim(-0.1, 0.25)



# 绘制成对特征散点图
from pandas.plotting import scatter_matrix
scatter_matrix(r_df, alpha=0.2, figsize=(6, 6), diagonal="kde")
plt.show()


# 六边形图
r_df.plot.hexbin(x="1 yr", y="2 yr",
                 gridsize = 15,
                 cmap="RdYlBu_r")
ax.set_xlim(-0.1, 0.25)
ax.set_ylim(-0.1, 0.25)


## 柱状图，竖直
r_df.mean().plot.bar()
# plt.savefig("柱状图.svg")


## 柱状图，水平
r_df.mean().plot.barh()
# plt.savefig("水平柱状图.svg")

# 直方图，子图布置
r_df.plot.hist(subplots=True, layout=(2,4),
             sharex = True, sharey = True,
             bins = 20,
             legend = False)

# plt.savefig("利率日收益率直方图，子图.svg")



# KDE，子图布置
r_df.plot.kde(subplots=True, layout=(2,4),
             sharex = True, sharey = True,
              ylim = (0,20),
             legend = False)

# plt.savefig("利率日收益率KDE，子图.svg")


from pandas.plotting import table
fig, ax = plt.subplots(1, 1)

table(ax, np.round(df.describe(), 2), loc="upper right")



# 绘制箱型图
r_df.plot.box()

# plt.savefig("利率日收益率箱型图.svg")


# 绘制箱型图，水平
r_df.plot.box(vert=False)

# plt.savefig("利率日收益率箱型图，水平.svg")

























































































































































































































































































































































































































































































































