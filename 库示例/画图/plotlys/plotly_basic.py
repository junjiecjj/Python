#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:39:41 2023

@author: jack
"""
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd


# 生成100个点
x = np.linspace(0, 2, 100)  
y0 = np.random.randn(100) + 5
y1 = np.random.randn(100)
y2 = np.random.randn(100) - 5
# 里面的参数一会儿解释
trace0 = go.Scatter(
    x=x,  # x 轴的坐标
    y=y0,  # y 轴的坐标
    mode="markers",  # 纯散点绘图
    name="散点"  # 曲线名称
)

trace1 = go.Scatter(
    x=x,  
    y=y1, 
    # 散点 + 线段绘图
    mode="markers + lines",  
    name="散点 + 线段" 
)

trace2 = go.Scatter(
    x=x,  
    y=y2, 
    mode="lines",  # 线段绘图
    name="线段"  
)  
# 我们看到比较神奇的地方是，Scatter 居然也可以绘制线段
# 是的，如果不指定 mode 为 "markers"，默认绘制的就是线段

# 以上就创建了三条轨迹，下面该干什么了？对，创建画布
# 将轨迹组合成列表传进去，因为一张画布是可以显示多条轨迹的
fig = go.Figure(data=[trace0, trace1, trace2])
# 在 notebook 中，直接通过 fig 即可显示图表
fig.show()


 
from IPython.display import HTML # 导入HTML
 

HTML(fig.to_html()) # to_html()




 
