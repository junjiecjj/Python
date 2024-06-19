


# https://blog.csdn.net/2301_80240808/article/details/135284863

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 创建多子图布局，指定行数和列数
fig = make_subplots(rows=2,
                    cols=2,
                    subplot_titles=("子图1", "子图2", "子图3", "子图4"),  # 子图标题
                    specs=[[{}, {}], [{}, {"type": "pie"}]]  # 每个子图的类型
                   )

# 添加子图1：散点图
trace1 = go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="markers", name="散点图")
fig.add_trace(trace1, row=1, col=1)

# 添加子图2：柱状图
trace2 = go.Bar(x=[1, 2, 3], y=[2, 3, 1], name="柱状图")
fig.add_trace(trace2, row=1, col=2)

# 添加子图3：折线图
trace3 = go.Scatter(x=[1, 2, 3], y=[10, 8, 9], mode="lines", name="折线图")
fig.add_trace(trace3, row=2, col=1)

# 添加子图4：饼图
trace4 = go.Pie(labels=["A", "B", "C"], values=[40, 30, 30], name="饼图")
fig.add_trace(trace4, row=2, col=2)

# 更新子图的布局属性
fig.update_layout(
    title_text = "多子图示例",
    showlegend=False,  # 隐藏图例
)

# 显示图表
fig.show()














