#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:39:41 2023

@author: jack
https://blog.csdn.net/weixin_45826022/article/details/122912279
https://blog.csdn.net/wzk4869/article/details/129864811
https://blog.csdn.net/weixin_45826022/article/details/122912484
https://cloud.tencent.com/developer/article/1439757
https://blog.csdn.net/2301_80239908/article/details/134853480


"""
import plotly.express as px
# import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.io as pio


from plotly.validators.scatter.marker import SymbolValidator

raw_symbols = SymbolValidator().values

# 设置plotly默认主题
pio.templates.default = 'plotly_white'
# Templates configuration
# -----------------------
#     Default template: 'plotly'
#     Available templates:
#         ['ggplot2', 'seaborn', 'simple_white', 'plotly',
#          'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
#          'ygridoff', 'gridon', 'none']


# 生成100个点
x = np.linspace(0, 2, 100)
y0 = np.random.randn(100) + 7
y1 = np.random.randn(100)
y2 = np.random.randn(100) - 7


x = ['2021-{:0>2d}'.format(s) for s in range(1,13)]
y1 = [70,72,80,65,76,80,60,67,80,90,94,82]
y2 = [65,42,35,25,67,54,34,45,38,46,64,34]

dfdata = {"月份":x, "召回率": y1, "准确率": y2}

fig = px.line(data_frame = dfdata,  x = "月份", y = ["召回率","准确率"], title ="线上模型表现变化趋势")
# fig.layout.yaxis.title = "指标"

# 设置图表标题和轴标签
fig.update_layout(
    ## 标题
    title = {'text':"线上模型表现变化趋势", 'y':0.9,  # 位置，坐标轴的长度看做1
        'x':0.5,
        'xanchor': 'center',   # 相对位置
        'yanchor': 'top',
        'font':{ 'size':22,  'family':'Time New Roman', 'color':' black',} },
    # titlefont={ 'size':22,  'family':'Time New Roman', 'color':' yellowgreen',  }, # titlefont：字典型，用于独立控制标题字体的部分，

    ## X label
    xaxis = {'visible':True, # 是否显示坐标轴，默认为 True，当设置为 False 时隐藏坐标轴，但是依旧可以进行拖拽。
             'type':'-', # str型，用于控制横坐标轴类型，'-'表示根据输入数据自适应调整，'linear'表示线性坐标轴，'log'表示对数坐标轴，'date'表示日期型坐标轴，'category'表示分类型坐标轴，默认为'-'

             ## 设置x轴（y轴）位于绘画区域的 底部（左侧）还是 顶部（右侧）,str型可取 'top'、 'bottom'、 'left'、 'right'。注：x轴只可取 'top' 或 'bottom'，同样，y轴只可取 'left' 或 'right'
             'side':'bottom',

             ## 坐标轴标题
             'title':{'text':'月份', 'font':{ 'size':22, 'color':'#28a428', 'family':'Time New Roman',}, 'standoff':1, }, # str型，设置横坐标轴上的标题
             # 'color':' blue', # str型，传入十六进制色彩，控制横坐标上所有元素的基础颜色（在未单独指定颜色之前，这些元素均采用此处color指定的颜色）

             ## 刻度(那个短线)
             'ticks':'inside', # 如果为空字符串 ""，则不绘制刻度线，默认值；如果为 'outside'，则绘制刻度线，且方向向图表外侧；如果为 'inside'，则绘制刻度线，且方向向图表内侧。
             'tickwidth':3, # int型，设置刻度线的宽度，大于等于0的整数，默认值为1
             'ticklen':5,  # 设置刻度线的长度，大于等于0的整数，默认值为5
             'tickcolor':'black',  #设置刻度线的颜色。
             # 'tick0':  # 设置此轴上第一个显示的刻度位置，搭配 dtick 共同使用。如果坐标轴的类型为 ‘log’，那么你传入的应该是取 ‘log’ 后的值；如果坐标轴类型为 ‘date’，那么传入的值也应该为日期字符串；如果坐标轴类型为 ‘category’，那么应该传入一个整数，从0开始，按出现顺序排列。
             # 'dtick':,  # ：设置显示刻度的步长，搭配 tick0 共同使用。必须为一个正整数，或者一个字符串（当坐标轴类型为 ‘date’ 或 ‘log’ 时）。
             # tickvals=list(range(len(line_dash))), # 显示列表中出现的刻度。只有当 tickmode 为 'array'时有效，与 ticktext 搭配使用。 列表、numpy数组、pandas Series。值为数字、字符串或日期
            # ticktext=line_dash, # 用于替换 tickvals 的刻度标签，与 tickvals 一一对应。例如 tickvals=[1, 2] 且 ticktext=['a', 'b'] 那么坐标轴上 1 刻度位置的标签将显示 ‘a’，而不是1，类似的 2刻度显示的标签为 ‘b’。  列表、numpy数组、pandas Series。值为数字、字符串或日期

             ## 刻度标签(坐标轴的数字)
             'showticklabels':True, #是否显示刻度标签，默认为 True
             'tickfont':{'size':22, 'color':' red', 'family':'Time New Roman', }, #字典型，设置刻度标签的字体。字典类型，

             'tickangle':20,   # 刻度标签的水平倾斜角度。

             ##坐标轴
             'showline':True, # bool型，控制是否绘制出该坐标轴上的直线部分
             'linewidth':2, # int型，设置坐标轴直线部分的像素宽度
             'linecolor':'black',   # 坐标轴线的颜色。
             # 'standoff':2, # 设置坐标轴标题与刻度标签之间的距离
             'mirror':True,  ## 决定是否将坐标轴 和（或）刻度镜像到绘图区域的另一侧。如果为 True，则坐标轴线被镜像；如果为 'ticks' ，则坐标轴线和刻度都被镜像；如果为 False，不使用镜像功能；如果为 'all'，坐标轴线会在所有共享该坐标轴的子图上镜像；如果为 'allticks'，坐标轴线和刻度会在所有共享该坐标轴的子图上镜像。

             ## 网格线
             'showgrid':True,   #  bool型，控制是否绘制网格线
             'gridcolor':'gray',  # str型，十六进制色彩，控制网格线的颜色
             'gridwidth':1, # int型，控制网格线的像素宽度

             ## 是否让图边距适应标签长度（因为标签总是限制在边框内，此参数可以让边框自动适应标签的长度）。
             'automargin':True,
             },

    ## Y label
    yaxis = {'visible':True,
             'type':'-',
             'side':'bottom',

             ## 坐标轴标题
             'title':{'text':'指标', 'font':{ 'size':22, 'color':'#28a428', 'family':'Time New Roman',}, 'standoff':1, }, # str型，设置横坐标轴上的标题

             ## 刻度(那个短线)
             'ticks':'inside',
             'tickwidth':3,
             'ticklen':5,
             'tickcolor':'black',
             # 'tick0':
             # 'dtick':,
             # tickvals=list(range(len(line_dash))),
             # ticktext=line_dash,

             ## 刻度标签(坐标轴的数字)
             'showticklabels':True, #是否显示刻度标签，默认为 True
             'tickfont':{'size':22, 'color':' red', 'family':'Time New Roman', }, #字典型，设置刻度标签的字体。字典类型，

             'tickangle':20,   # 刻度标签的水平倾斜角度。

             ##坐标轴
             'showline':True, # bool型，控制是否绘制出该坐标轴上的直线部分
             'linewidth':2, # int型，设置坐标轴直线部分的像素宽度
             'linecolor':'black',   # 坐标轴线的颜色。
             # 'standoff':2, # 设置坐标轴标题与刻度标签之间的距离
             'mirror':True,  ## 决定是否将坐标轴 和（或）刻度镜像到绘图区域的另一侧。

             ## 网格线
             'showgrid':True,   #  bool型，控制是否绘制网格线
             'gridcolor':'gray',  # str型，十六进制色彩，控制网格线的颜色
             'gridwidth':1, # int型，控制网格线的像素宽度

             ## 是否让图边距适应标签长度（因为标签总是限制在边框内，此参数可以让边框自动适应标签的长度）。
             'automargin':True,
             },

    ## legend , https://blog.csdn.net/2301_80239908/article/details/134853480
    showlegend=True,
    # legend_title_text = 'Trend',
    legend = {
        'tracegroupgap':1,  # 设置图例组之间的间隔
        'title_font_family': "Times New Roman",
        'font':{'size':22, 'color':'orangered', 'family':'simsun',}, # 字典型，设置图例文字部分的字体，同前面所有font设置规则
        'bgcolor':'LightSteelBlue',   # str型，十六进制设置图例背景颜色
        'bordercolor':'gray',   # 设置图例边框的颜色
        'borderwidth':2,  # int型，设置图例边框的cuxi
        'orientation':'v', # str型，设置图例各元素的堆叠方向，'v'表示竖直，'h'表示水平堆叠
        'x':0,
        'y':1,
        'xanchor':"left", # str型，用于直接设置图例水平位置的固定位置，有'left'、'center'、'right'和'auto'几个可选项
        'yanchor':"top",
        },
    # paper_bgcolor = 'white',   # str型，传入十六进制色彩，控制图床的颜色
    # plot_bgcolor = 'white',    # str型，传入十六进制色彩，控制绘图区域的颜色
    # margin=dict(l=20, r=20, t=20, b=20),  # 上下左右的边距大小
    )

# pio.write_image(fig, 'plot_scatter.pdf', )
# 在 notebook 中，直接通过 fig 即可显示图表
fig.show()







# import plotly.express as px
# df = px.data.iris()
# fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", symbol="species")
# fig.show()


import plotly.express as px
df = px.data.gapminder().query("continent == 'Oceania'")
fig = px.line(df, x='year', y='lifeExp', color='country', symbol="country")
fig.show()


















