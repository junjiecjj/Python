

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 时间序列中的缺失值


from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader
import scipy.stats as stats
import pylab

# 下载数据
df = pandas_datareader.data.DataReader(['sp500'], data_source='fred', start='01-01-2021', end='08-01-2021')
df.to_csv('sp500.csv')
df.to_pickle('sp500.pkl')



# 随机插入缺失值
df_NaN = df.copy()
mask = np.random.uniform(0,1,size = df_NaN.shape)
mask = (mask <= 0.3)
df_NaN[mask] = np.NaN
df_NaN.tail



# 向前
df_NaN_forward  = df_NaN.ffill()
# ffill() is equivalent to fillna(method='ffill')

fig, axs = plt.subplots()

df_NaN_forward['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()


# 向后
df_NaN_backward = df_NaN.bfill()
# bfill() is equivalent to fillna(method='bfill')
fig, axs = plt.subplots()

df_NaN_backward['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()

# 线性插值
df_NaN_interpolate = df_NaN.interpolate()
fig, axs = plt.subplots()

df_NaN_interpolate['sp500'].plot(color = 'r')
df_NaN['sp500'].plot(marker = 'x')
plt.xlabel('Date')
plt.ylabel('Price level with NaN')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 从时间数据中发现趋势

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
import scipy.stats as stats
import pylab

df = pandas_datareader.data.DataReader(['UNRATENSA'], data_source='fred', start='08-01-1950', end='08-01-2021')
df = df.dropna()
df.to_csv('UNRATENSA.csv')
df.to_pickle('UNRATENSA.pkl')




#%% long term trend

average_rate = df['UNRATENSA'].mean()
fig, axs = plt.subplots()
df['UNRATENSA'].plot()

plt.axhline(y=average_rate, color= 'r', zorder=0)
plt.axhline(y=df['UNRATENSA'].max(), color= 'r', zorder=0)
plt.axhline(y=df['UNRATENSA'].min(), color= 'r', zorder=0)
axs.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

plt.xlabel('Date')
plt.ylabel('Unemployment rate')
plt.show()



#%% Zoom in
fig, axs = plt.subplots()

df['UNRATENSA']['1989-01-01':'1999-01-01'].plot()
plt.xlabel('Date')
plt.ylabel('Unemployment rate')
plt.show()

axs.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])



#%%
import seaborn as sns

df['year'] = pd.DatetimeIndex(df.index).year
df['month'] = pd.DatetimeIndex(df.index).month
import calendar
df['month'] = df['month'].apply(lambda x: calendar.month_abbr[x])


#%%
fig, axs = plt.subplots()

sns.lineplot(data=df['1989-01-01':'1999-01-01'], x="year", y="UNRATENSA", hue="month")
plt.show()


fig, axs = plt.subplots()

sns.lineplot(data=df['1989-01-01':'1999-01-01'], x="month", y="UNRATENSA", hue="year")
plt.show()



#%%

fig, axs = plt.subplots()

sns.boxplot(x='year', y='UNRATENSA', data=df)
plt.xticks(rotation = 90)
plt.show()

fig, axs = plt.subplots()

sns.boxplot(x='month', y='UNRATENSA', data=df)
plt.xticks(rotation = 45)
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 季节调整

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
import scipy.stats as stats





df = pandas_datareader.data.DataReader(['UNRATENSA'], data_source='fred', start='08-01-2000', end='08-01-2021')
df = df.dropna()

# deal with missing values
df['UNRATENSA'].interpolate(inplace=True)

res = sm.tsa.seasonal_decompose(df['UNRATENSA'])

# generate subplots
resplot = res.plot()

res.resid
res.seasonal
res.trend



#%% Original data

fig, axs = plt.subplots()

df['UNRATENSA'].plot()
plt.xlabel('Date')
plt.ylabel('Original')
plt.show()


#%% plot trend on top of original curve

df['UNRATENSA'].plot()
res.trend.plot(color = 'r')
plt.xlabel('Date')
plt.ylabel('Trend')
plt.show()



#%% plot seasonal component

fig, axs = plt.subplots()

res.seasonal.plot()
plt.axhline(y = 0, color = 'r')
plt.xlabel('Date')
plt.ylabel('Seasonal')
plt.show()


#%% plot irregular

fig, axs = plt.subplots()

res.resid.plot()
plt.axhline(y = 0, color = 'r')
plt.xlabel('Date')
plt.ylabel('irregular')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 时间数据讲故事

import plotly.express as px
import numpy as np
import pandas as pd



# 导入数据
df = px.data.gapminder()
df.rename(columns={"country": "country_or_territory"},
          inplace = True)
# 列名称不够准确
df.to_pickle('gapminder.pkl')
df.to_csv('gapminder.csv')


df.columns

# 人口
df_pop_continent_over_t = df.groupby(['year','continent'],  as_index=False).agg({'pop': 'sum'})


df_pop_continent_over_t


fig = px.bar(df_pop_continent_over_t,
             x = 'year', y = 'pop',
             width=600, height=380,
             color = 'continent',
             labels={"year": "Year",
                     "pop": "Population"})
fig.write_image("各大洲人口随时间变化，柱状图.svg")
fig.show()


fig = px.line(df_pop_continent_over_t,
             x = 'year', y = 'pop',
             width=600, height=380,
             color = 'continent',
             labels={"year": "Year",
                     "pop": "Population"})
fig.write_image("各大洲人口随时间变化，线图.svg")
fig.show()


# 步骤1: 计算每年全球的总人口
global_pop = df.groupby('year')['pop'].sum().reset_index()

# 步骤2: 计算每个大洲每年的总人口
continent_pop = df.groupby(['continent', 'year'])['pop'].sum().reset_index()

# 步骤3: 合并数据，计算比例
pop_with_ratio = pd.merge(continent_pop,
                          global_pop,
                          on='year',
                          suffixes=('_continent', '_global'))
pop_with_ratio['ratio'] = pop_with_ratio['pop_continent'] / pop_with_ratio['pop_global']

# 使用Plotly绘制叠加填充线图
fig = px.bar(pop_with_ratio,
              x="year", y="ratio",
              color="continent",
              width=600, height=380,
              labels={"ratio": "Population ratio"})
fig.write_image("各大洲人口占比随时间变化，堆积柱状图.svg")
fig.show()


# 使用Plotly绘制叠加填充线图
fig = px.area(pop_with_ratio,
              x="year", y="ratio",
              color="continent",
              width=600, height=380,
              labels={"ratio": "Population ratio"})
fig.write_image("各大洲人口占比随时间变化，面积图.svg")
fig.show()


continent_population_2007 = df[df['year'] == 2007].groupby('continent')['pop'].sum().reset_index()

fig = px.pie(continent_population_2007,
             values='pop',
             hole = 0.68,
             names='continent')
# fig.write_image("各大洲人口比例，2007年.svg")
fig.show()


fig = px.sunburst(df.query("year == 2007"),
                  path=['continent', 'country_or_territory'],
                  values='pop',
                  color='lifeExp',
                  width=600, height=600,
                  hover_data=['iso_alpha'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['lifeExp'],
                                                       weights=df['pop']))
# fig.write_image("人口比例，太阳爆炸图，2007年.svg")
fig.show()


fig = px.line(pop_with_ratio,
             x = 'year', y = 'ratio',
             width=600, height=380,
             color = 'continent',
             labels={"year": "Year",
                     "ratio": "Population Ratio"})
# fig.write_image("各大洲人口占比随时间变化，线图.svg")
fig.show()


# create bar charts
fig = px.bar(df.query("country_or_territory == 'Canada'"),
             x='year', y='pop',
             width=600, height=500)
# fig.write_image("特定国家 (加拿大) 人口随年份变化.svg")
fig.show()


fig = px.bar(df.query("year == 2007 and pop > 1.e8"),
             y='pop', x='country_or_territory',
             width=600, height=380,
             labels={"year": "Year","pop": "Population"},
             text_auto='.2s')
# fig.write_image("人口超过一亿的国家，2007年.svg")
fig.show()



fig = px.bar(df.query("year == 2007 and pop > 1.e8"),
             y='pop', x='country_or_territory',
             width=600, height=380,
             labels={"year": "Year","pop": "Population"},
             text_auto='.2s')
# fig.write_image("人口超过一亿的国家，2007年.svg")
fig.show()



# fig = px.line(df_wide.diff(),
#              width=600, height=500,
#              color = 'continent',
#              labels={"year": "Year",
#                      "pop": "Population"})
# # fig.write_image("各大洲人口年度变化，线图.svg")
# fig.show()



# # 百分比变化
# fig = px.line(df_wide.pct_change() * 100,
#              width=600, height=500,
#              color = 'continent',
#              labels={"year": "Year",
#                      "pop": "Population"})
# # fig.write_image("各大洲人口年度百分比变化，线图.svg")
# fig.show()






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






























































































































































































































































































