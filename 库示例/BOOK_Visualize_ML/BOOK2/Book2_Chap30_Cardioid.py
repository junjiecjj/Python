



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  从心形线到模数乘法表

import numpy as np
import matplotlib.pyplot as plt
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


N = 360 # 请大家尝试720

theta_array = np.linspace(0,2*np.pi,N+1)
x_array = np.cos(theta_array)
y_array = np.sin(theta_array)
points = np.column_stack((x_array,y_array))


def visualize(k = 2):
    # 可视化
    fig, ax = plt.subplots(figsize=(8,8))
    # 用hsv颜色映射一次渲染每一条弦
    colors = plt.cm.hsv(np.linspace(0, 1, N+1))

    # i 为弦第一个点的序号
    for i in range(N+1):

        # j 为弦第二个点的序号
        j = (i*k) % N

        # 绘制弦线段，两个点分别为
        # point[i], points[j]
        plt.plot([points[i,0], points[j,0]],
                 [points[i,1], points[j,1]],
                 lw = 0.1,c = colors[i])

    ax.axis('off')
    # fig.savefig('Figures\\' + str(k) + '.svg') # png

visualize(2)

visualize(321)

visualize(90)


visualize(206)

visualize(358)


visualize(310)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正方形

import numpy as np
import matplotlib.pyplot as plt

def linear_interpolation(point1, point2, num_intervals):
    x1, y1 = point1
    x2, y2 = point2

    x_step = (x2 - x1) / num_intervals
    y_step = (y2 - y1) / num_intervals

    interpolated_coordinates = [(x1 + i * x_step, y1 + i * y_step) for i in range(num_intervals + 1)]

    return interpolated_coordinates

N = 360
A = (0,0)
B = (1,0)
C = (1,1)
D = (0,1)

AB = linear_interpolation(A, B, 90)
BC = linear_interpolation(B, C, 90)
CD = linear_interpolation(C, D, 90)
DA = linear_interpolation(D, A, 90)

points = AB[:-1] + BC[:-1] + CD[:-1] + DA

points = np.array(points)

def visualize(k = 2):
    fig, ax = plt.subplots(figsize=(8,8))
    colors = plt.cm.hsv(np.linspace(0, 1, N+1))
    for i in range(N+1):
        j = (i*k) % N
        plt.plot([points[i,0], points[j,0]], [points[i,1], points[j,1]], lw = 0.1,c = colors[i])

    ax.axis('off')
    # fig.savefig('Figures\\' + str(k) + '.svg') # png

visualize(2)

visualize(241)

visualize(270)


visualize(188)

visualize(358)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 等边三角形
import numpy as np
import matplotlib.pyplot as plt
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


def linear_interpolation(point1, point2, num_intervals):
    # 自定义插值函数
    x1, y1 = point1
    x2, y2 = point2

    x_step = (x2 - x1) / num_intervals
    y_step = (y2 - y1) / num_intervals

    interpolated_coordinates = [(x1 + i * x_step, y1 + i * y_step) for i in range(num_intervals + 1)]

    return interpolated_coordinates

N = 360
# 等边三角形的三个顶点坐标
A = (0,0)
B = (1,0)
C = (1/2,np.sqrt(3)/2)

AB = linear_interpolation(A, B, 120)
BC = linear_interpolation(B, C, 120)
CA = linear_interpolation(C, A, 120)



points = AB[:-1] + BC[:-1] + CA

points = np.array(points)


def visualize(k = 2):
    fig, ax = plt.subplots(figsize=(8,8))
    colors = plt.cm.hsv(np.linspace(0, 1, N+1))
    for i in range(N+1):
        j = (i*k) % N
        plt.plot([points[i,0], points[j,0]],
                 [points[i,1], points[j,1]],
                 lw = 0.1,c = colors[i])

    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig('Figures\\' + str(k) + '.svg') # png
visualize(7)

visualize(241)


visualize(270)

visualize(188)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


