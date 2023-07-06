#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:58:49 2023

@author: jack
"""


import matplotlib.pyplot as plt


#===============================================================================
#             动态画图
#===============================================================================

# plt.cla()清除轴 ，即当前图中的当前活动轴。 它使其他轴保持不变。
# plt.clf()使用其所有轴清除整个当前图形 ，但使窗口保持打开状态，以便可以将其重新用于其他绘图。
# plt.close()关闭一个 window，如果没有另外指定，它将是当前窗口。



x = list(range(1, 10))  # epoch array
loss = [2 / (i**2) for i in x]  # loss values array
plt.ion()
for i in range(1, len(x)):
    ix = x[:i]
    iy = loss[:i]
    plt.clf()
    plt.title("loss")
    plt.plot(ix, iy)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.pause(0.5)
plt.ioff()
plt.show()


#===============================================================================
# 动态地展示多张图片
#===============================================================================

# import imageio.v2 as imageio

# imglist = ["/home/jack/公共的/Python/OpenCV/Figures/bird.png",
#             "/home/jack/公共的/Python/OpenCV/Figures/baby.png",
#             "/home/jack/公共的/Python/OpenCV/Figures/flower.jpg",
#             ]

# f, a = plt.subplots(1, 1, figsize=(3, 3))
# plt.ion()
# for imgPath in imglist:
#     img = imageio.imread(imgPath)
#     a.imshow(img )
#     a.set_xticks(())
#     a.set_yticks(())
#     plt.pause(1)

# plt.ioff()
# plt.show()






#===============================================================================
#  动态地展示 scatter
#===============================================================================


# 我们学习以pyplot方法绘制动态图方法，哪我们来实操一下吧

# 调用numpy.arange()准备x,y轴数据
# 调用pyplot.scatter()绘制散点图
# 使用for循环包含以上步骤，在for循环开始调用pyplot.ion()打开交互模式
# for循环结束后，调用pyplot.ioff()关闭交互模式
# 最后调用pyplot.show()展示图像画面


import numpy as np
import matplotlib.pyplot as plt

def scatter_plot():
    # 打开交互模式
    plt.ion()
    for index in range(50):
        # plt.cla()

        plt.title("动态散点图")
        plt.grid(True)

        point_count = 5
        x_index = np.random.random(point_count)
        y_index = np.random.random(point_count)
  
        color_list = np.random.random(point_count)
        scale_list = np.random.random(point_count) * 100

        plt.scatter(x_index, y_index, s=scale_list, c=color_list, marker="^")

        plt.pause(0.2)

    plt.ioff()

    plt.show()

# scatter_plot()















