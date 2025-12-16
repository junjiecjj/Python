#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 13:54:51 2025

@author: jack
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（确保系统中有所用字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimHei"
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

# 创建画布和子图
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

# 数据
years = [str(year) for year in range(2020, 2041, 2)]

# 左侧图表数据：物联网设备连接数量（单位：亿台）
iot_devices = np.linspace(100, 300, num = len(years)).astype(int)

# 右侧图表数据：数据流量（单位：EB/月）
k = np.log10(60000)
data_traffic = ( ( np.linspace(5, 10, num=len(years)) )**k ).astype(int)  # 示例数据
data_traffic[:-2] = data_traffic[:-2] + 200
# 左侧图表：全球物联网设备连接数量预测
bars1 = ax1.bar(years, iot_devices, color='skyblue', edgecolor='black', capsize = 11)
ax1.set_title('全球物联网设备连接数量预测(来源：Statista)', fontsize=24, pad=15)
ax1.set_ylabel('单位：亿台', fontsize=24)
# ax1.set_xlabel('', fontsize=18, style='italic')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

ax1.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 18, width=3,)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置y轴刻度
ax1.set_ylim(0, 350)
ax1.set_yticks(range(0, 351, 50))

out_fig = plt.gcf()
out_fig.savefig('/home/jack/文档/中山大学/19 毕业大论文/ChenJJ/figures/Chap_01/num_device.pdf' )
plt.show()
plt.close()


######
fig, ax2 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

# 右侧图表：全球数据流量预测
bars2 = ax2.bar(years, data_traffic, color='lightcoral', edgecolor='black')
ax2.set_title('全球数据流量预测(来源：ITU)', fontsize=24, pad=22)
ax2.set_ylabel('单位：EB/月', fontsize=24)
# ax2.set_xlabel('（）', fontsize=18, style='italic')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

ax2.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 18, width=3,)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(16) for label in labels]  # 刻度值字号

# 在柱子上添加数值标签（可选）
# for bar in bars2:
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height}', ha='center', va='bottom', fontsize=12)

# 设置y轴刻度
ax2.set_ylim(0, 66500)
ax2.set_yticks(range(0, 66501, 10000))

# 调整布局
plt.tight_layout()

out_fig = plt.gcf()
out_fig.savefig('/home/jack/文档/中山大学/19 毕业大论文/ChenJJ/figures/Chap_01/num_flow.pdf' )
plt.show()
plt.close()








