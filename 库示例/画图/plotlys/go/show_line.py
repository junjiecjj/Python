#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:18:32 2024

@author: jack
"""
import plotly.graph_objects as go
# from plotly.validators.scatter.marker import SymbolValidator

fig = go.Figure()

line_dash = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
for i, d in enumerate(line_dash):
    fig.add_trace(
        go.Scatter(
            x=[0, 10], y=[i, i], mode='lines',
            line=dict(dash=d, width=3,),
            showlegend=False
    ))


fig.update_layout(
    width=600, height=500,
    yaxis=dict(
        type='category',
        tickvals=list(range(len(line_dash))),
        ticktext=line_dash,
        showgrid=False
    ),
    xaxis_showticklabels=False,
    xaxis_showgrid=False,
)

# fig.write_image('../pic/lines_1.png', scale=10)
fig.show()





#%%
fig = go.Figure()

line_dash = ['5px,10px,3px,2px',
             '10px,10px,5px,5px,3px',
             '30px',
             '30px,10px']
for i, d in enumerate(line_dash):
    fig.add_trace(
        go.Scatter(
            x=[0, 10], y=[i, i], mode='lines',
            line_dash=d,
            showlegend=False
    ))


fig.update_layout(
    width=600, height=500,
    yaxis=dict(
        type='category',
        tickvals=list(range(len(line_dash))),
        ticktext=line_dash,
        showgrid=False
    ),
    xaxis_showticklabels=False,
    xaxis_showgrid=False,
)

# fig.write_image('../pic/lines_2.png', scale=10)
fig.show()
