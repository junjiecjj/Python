
"""
以下是plotly绘制箱型图常用参数的说明：

x：指定箱型图的横坐标数据，可以是一个包含数值的列表、一维numpy数组、pandas的Series或DataFrame对象；
y：指定箱型图的纵坐标数据，可以是一个包含数值的列表、一维numpy数组、pandas的Series或DataFrame对象；
name：指定该箱型图的名称，用于在图例中展示；
boxpoints：指定是否在箱型图中展示数据点。可选值包括’all’（所有数据点均展示）、‘outliers’（仅展示异常值）、False（不展示数据点）；
boxmean：指定是否在箱型图中展示均值线。可选值为True或False；
orientation：指定箱型图的方向。可选值为’h’（水平方向）或’v’（竖直方向）；
notched：指定箱型图是否展示缺口（notch），用于展示置信区间。可选值为True或False；
notchwidth：指定缺口的宽度，取值范围为[0,1]；
notchspan：指定缺口的跨度，取值范围为[0,1]；
whiskerwidth：指定箱线的线宽；
line_width：指定箱线、均值线等的线宽；
line_color：指定箱线、均值线等的颜色；
fillcolor：指定箱型图的填充颜色；
opacity：指定箱型图的透明度；
marker：指定箱型图中数据点的样式，包括color（颜色）、size（大小）、symbol（形状）等参数。


"""
# https://blog.csdn.net/wzk4869/article/details/129864811



import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "day": ["Mon", "Tues", "Wed", "Thurs", "Fri", "Sat", "Sun"],
    "value1": [1, 3, 2, 5, 7, 8, 4],
    "value2": [4, 6, 5, 3, 2, 3, 6],
    "value3": [7, 4, 5, 6, 5, 2, 4]
})

fig = px.box(df, x="day", y=["value1", "value2", "value3"])
fig.show()
