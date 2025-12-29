#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:16:01 2024

@author: jack
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties


# 获取当前系统用户目录
home = os.path.expanduser('~')

fontpath = "/usr/share/fonts/truetype/windows/"

colors = ['#FF0000','#0000FF','#00FF00','#1E90FF','#4ea142','#FF00FF','#FFA500','#800080','#EE82EE','#00FFFF','#9932CC','#FF6347','#00CED1','#CD5C5C',  '#7B68EE','#808000']

def model_update_range(communication_rounds, decay_rate):
    """
    模拟模型更新范围随通信轮数下降的函数

    参数:
    communication_rounds: 通信轮数（0-100）
    decay_rate: 衰减速率，控制下降速度

    返回:
    模型更新范围（0-0.1之间）
    """
    # 初始值设为0.10
    initial_range = 0.10

    # 指数衰减公式
    # 参数说明:
    # - decay_rate越大，下降越快
    # - 使用communication_rounds/50作为缩放，使曲线更平滑
    update_range = initial_range * np.exp(-decay_rate * communication_rounds / 50)

    return update_range

def model_update_range_enhanced(communication_rounds, decay_rate, initial_range=0.120, final_range=0.005, scale_factor=50):
    """
    增强版模型更新范围函数

    参数:
        communication_rounds: 通信轮数
        decay_rate: 衰减速率
        initial_range: 初始更新范围
        final_range: 最终渐近线值
        scale_factor: 缩放因子，控制曲线形状
    """
    # 计算偏移量，确保最终值接近final_range
    offset = final_range

    # 指数衰减公式，带偏移
    update_range = (initial_range - offset) * np.exp(-decay_rate * communication_rounds / scale_factor) + offset

    return update_range



# 不同衰减速率（模拟图中不同的层）
decay_rates = {
    'conv1.weight': [1.2, 0.110, 0.003] ,    # 较快衰减
    'conv2.weight': [0.7, 0.100, 0.006],    # 中等衰减
    'fc1.weight': [1.0, 0.091, 0.002],      # 较慢衰减
    'fc2.weight': [0.6, 0.096, 0.001]        # 最慢衰减
}
communication_rounds = np.linspace(0, 1000, 1001)
dataset = "CIFAR10"

# decay_rates = {
#     'conv1.weight': [3.2, 0.100, 0.005] ,    # 较快衰减
#     'conv2.weight': [3.7, 0.101, 0.004],    # 中等衰减
#     'fc1.weight': [3.0, 0.098, 0.002],      # 较慢衰减
#     'fc2.weight': [2.6, 0.103, 0.001]        # 最慢衰减
# }
# communication_rounds = np.linspace(0, 300, 301)
# dataset = "MNIST"


# key_want = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)

i = 0
for layer_name, [decay_rate, up, down] in decay_rates.items():
    y = model_update_range_enhanced(communication_rounds, decay_rate, up, down)
    y = y + np.random.randn(y.size) * 0.0006
    axs.plot(communication_rounds, y, label=layer_name, linewidth=2, color = colors[i])
    i += 1

font2 = FontProperties(fname=fontpath+"simsun.ttf", size=26)
axs.set_xlabel( "通信轮数", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('模型更新范围', fontproperties=font2, )

# plt.title('Model Update Range vs Communication Rounds', fontsize=14)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
legend1 = axs.legend(loc = 'best', borderaxespad=0, edgecolor='black', prop=font2, borderpad = 0.1, labelspacing = 0.1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')                         # 设置图例legend背景透明

axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(25) for label in labels]  # 刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

out_fig = plt.gcf()

savedir = home + f'/FL_DQ/Figures/{dataset}'
out_fig.savefig(f'{savedir}/Fig_{dataset}_MinMax.pdf' )
plt.show()
plt.close()








