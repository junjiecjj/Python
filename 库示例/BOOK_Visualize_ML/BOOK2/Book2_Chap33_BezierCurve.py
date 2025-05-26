
import numpy as np
__all__ = ["Bezier"]
class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            # print("i1 =", i1)
            # print("points[i1] =", points[i1])
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            # print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve


#%% 一阶贝塞尔曲线原理
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def lerp(P_a, P_b, t_array):

    P_out = [P_a * t_idx + P_b * (1 - t_idx) for t_idx in t_array]
    P_out = np.array(P_out)

    return P_out

P_0 = np.array([-4, 4])
P_1 = np.array([4, -4])

delta_t = 1/16
t_array = np.linspace(delta_t, 1, int(1/delta_t - 1), endpoint = False)
t_array_fine = np.linspace(0, 1, 101, endpoint = True)

B_array = lerp(P_0, P_1, t_array)
B_array_fine = lerp(P_0, P_1, t_array_fine)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(t_array)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

t = mpl.markers.MarkerStyle(marker='x')
t._transform = t.get_transform().rotate_deg(30)

plt.plot([P_0[0], P_1[0]], [P_0[1], P_1[1]], color = 'k', marker = t, ms = 20, lw = 0.5)

for i in range(num):
    plt.plot(B_array[i,0],B_array[i,1], marker = t, c = colors[i], ms = 10, zorder = 1e5)

plt.plot(B_array_fine[:,0],B_array_fine[:,1],c = 'k', lw = 2)
# plt.plot(([i for (i,j) in P_0_1], [i for (i,j) in P_1_2]),
#          ([j for (i,j) in P_0_1], [j for (i,j) in P_1_2]),
#          c=[0.6,0.6,0.6], alpha = 0.5)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 5), ax.set_ylim(-5, 5)
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.grid(c = '0.8', lw = 0.25)
plt.show()


#%% # 二阶贝塞尔曲线原理
import numpy as np
import matplotlib.pyplot as plt
import os

def lerp(P_a, P_b, t_array):
    P_out = [P_a * (1 - t_idx) + P_b * t_idx for t_idx in t_array]
    P_out = np.array(P_out)
    return P_out

def lerp_2nd(P_a_array, P_b_array, t_array):
    t_array = t_array.reshape(-1,1)
    P_out = P_a_array * (1 - t_array) + P_b_array * t_array
    return P_out

def Bezier_2nd(P_0, P_1, P_2, t_array):
    B_array = lerp_2nd(lerp(P_0, P_1, t_array), lerp(P_1, P_2, t_array), t_array)
    # B_array = [(1 - t_idx)**2 * P_2 +
    #            2 * (1 - t_idx) * t_idx * P_1 +
    #            t_idx**2 * P_0 for t_idx in t_array]
    # B_array = np.array(B_array_fine)
    return B_array

P_0 = np.array([-4, 4])
P_1 = np.array([4, 4])
P_2 = np.array([4, -4])

delta_t = 1/16
t_array = np.linspace(delta_t,1,int(1/delta_t - 1), endpoint = False)
t_array_fine = np.linspace(0, 1, 101, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)

B_array = lerp_2nd(P_0_1, P_1_2, t_array)
B_array_fine = Bezier_2nd(P_0, P_1, P_2, t_array_fine)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

plt.plot([P_0[0],P_1[0]], [P_0[1],P_1[1]], color = 'k', marker = 'x', ms = 20, lw = 0.5)
plt.plot([P_1[0],P_2[0]], [P_1[1],P_2[1]], color = 'k', marker = 'x', ms = 20, lw = 0.5)

for i in range(num):
    plt.plot([P_0_1[i, 0], P_1_2[i, 0]], [P_0_1[i, 1], P_1_2[i, 1]], color=colors[i], marker = '.', ms = 10, lw = 0.25)
    plt.plot(B_array[i,0],B_array[i,1], marker = 'x', c = colors[i], ms = 10, zorder = 1e5)

plt.plot(B_array_fine[:,0],B_array_fine[:,1],c = 'k', lw = 2)
# plt.plot(([i for (i,j) in P_0_1], [i for (i,j) in P_1_2]),
#          ([j for (i,j) in P_0_1], [j for (i,j) in P_1_2]),
#          c=[0.6,0.6,0.6], alpha = 0.5)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 5), ax.set_ylim(-5, 5)
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.grid(c = '0.8', lw = 0.25)
# fig.savefig('Figures/二阶贝塞尔曲线原理.svg', format='svg')
plt.show()


#%% # 使用Bezier

# from Bezier import Bezier
import numpy as np
# from numpy import array as a
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#~~~#
fig = plt.figure(dpi=128)

#~~~# Simple arch.
t_points = np.arange(0, 1, 0.01)
test = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]])
test_set_1 = Bezier.Curve(t_points, test)

plt.subplot(2, 3, 3)
plt.xticks([i1 for i1 in range(-20, 20)]),
plt.yticks([i1 for i1 in range(-20, 20)])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(  which='major', axis='both')

plt.plot(test_set_1[:, 0], test_set_1[:, 1])
plt.plot(test[:, 0], test[:, 1], 'ro:')

#~~~# Simple wave.
t_points = np.arange(0, 1, 0.01)
test = np.array([[0, 5], [4, 10], [6, 10], [4, 0], [6, 0], [10, 5]])
test_set_1 = Bezier.Curve(t_points, test)

plt.subplot(2, 3, 6)
plt.xticks([i1 for i1 in range(-20, 20)]),
plt.yticks([i1 for i1 in range(-20, 20)])
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(  which='major', axis='both')

plt.plot(test_set_1[:, 0], test_set_1[:, 1])
plt.plot(test[:, 0], test[:, 1], 'ro:')

#~~~# Plushy heart.
points2 = np.array([[5, 0], [0, 2], [0, 10], [6, 10], [14, -2], [-4, -2], [4, 10], [10, 10], [10, 2], [5, 0]])
t_points = np.arange(0, 1, 0.01)
curve2 = Bezier.Curve(t_points, points2)

plt.subplot(2, 3, 2)
plt.xticks([i1 for i1 in range(-20, 20)]), plt.yticks([i1 for i1 in range(-20, 20)])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid( which='major', axis='both')

plt.plot(curve2[:, 0], curve2[:, 1], 'r')
plt.plot(points2[:, 0], points2[:, 1], 'yx:')

#~~~# 10-point Bezier curve.
points2 = np.array([[0, 2], [2, 8], [6, 6], [4, 4], [2, 2], [6, 0], [8, 4], [10, 8], [8, 10], [6, 9]])
t_points = np.arange(0, 1, 0.01)
curve2 = Bezier.Curve(t_points, points2)

plt.subplot(2, 3, 1)
plt.xticks([i1 for i1 in range(-20, 20)]), plt.yticks([i1 for i1 in range(-20, 20)])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(  which='major', axis='both')

plt.plot(curve2[:, 0], curve2[:, 1])
plt.plot(points2[:, 0], points2[:, 1], 'ro:')

#~~~# 3 of 3-point Bezier curves
points_set_1 = np.array([[0, 2], [2, 8], [6, 6], [4, 4]])
points_set_2 = np.array([[4, 4], [2, 2], [6, 0], [8, 4]])
points_set_3 = np.array([[8, 4], [10, 8], [8, 10], [6, 9]])
t_points = np.arange(0, 1, 0.01)

curve_set_1 = Bezier.Curve(t_points, points_set_1)
curve_set_2 = Bezier.Curve(t_points, points_set_2)
curve_set_3 = Bezier.Curve(t_points, points_set_3)

plt.subplot(2, 3, 4)
plt.xticks([i1 for i1 in range(-20, 20)]), plt.yticks([i1 for i1 in range(-20, 20)])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid( which='major', axis='both')

plt.plot(curve_set_1[:, 0], curve_set_1[:, 1])
plt.plot(points_set_1[:, 0], points_set_1[:, 1], 'bs:')

plt.plot(curve_set_2[:, 0], curve_set_2[:, 1])
plt.plot(points_set_2[:, 0], points_set_2[:, 1], 'ro:')

plt.plot(curve_set_3[:, 0], curve_set_3[:, 1])
plt.plot(points_set_3[:, 0], points_set_3[:, 1], 'gx:')

#~~~# 3D "Clockwise helix and drop"
points_set_1 = np.array([[0, 0, 0], [0, 4, 0], [2, 5, 0], [4, 5, 0], [5, 4, 0], [5, 1, 0], [4, 0, 0], [1, 0, 3], [0, 0, 4], [0, 2, 5], [0, 4, 5], [4, 5, 5], [5, 5, 4], [5, 5, 0]])
t_points = np.arange(0, 1, 0.01)
curve_set_1 = Bezier.Curve(t_points, points_set_1)

ax = fig.add_subplot(235, projection='3d')
ax.plot(curve_set_1[:, 0], curve_set_1[:, 1], curve_set_1[:, 2])
ax.plot(points_set_1[:, 0], points_set_1[:, 1], points_set_1[:, 2], 'o:')
#~~~#
plt.show()
help(Bezier)

#%% # 二阶、三阶、四阶贝塞尔曲线
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import random as rnd

def comb(n, k):
    # 产生系数
    return factorial(n) // (factorial(k) * factorial(n-k))

def get_bezier_curve(points):
    n = len(points) - 1
    # 贝塞尔曲线阶数
    # 匿名函数，产生特定比例 t 对应的贝塞尔曲线坐标
    # 下式相当于加权平均
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))

def evaluate_bezier(points, total = 200):
    # 产生连续贝塞尔曲线坐标点
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]

### 二阶
A = [0, 0]
B = [1, 1]
# 曲线的两个端点
fig = plt.figure(figsize = (10,16), tight_layout=True)
for idx in range(18):
    P1 = [rnd.random(), rnd.random()]
    # 用随机数生成控制点P1
    points = np.array([A, P1, B])
    ax = fig.add_subplot(6, 3, idx + 1)
    x, y = points[:, 0], points[:, 1]
    bx, by = evaluate_bezier(points, 200)
    colors = plt.cm.RdYlBu(np.linspace(0,1,200, endpoint = True))
    plt.scatter(bx, by, c = colors)
    plt.plot(x, y, 'k:', marker = 'x')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal', adjustable='box')


### 三阶
fig = plt.figure(figsize = (10,16), tight_layout=True)
for idx in range(18):
    P1 = [rnd.random(), rnd.random()]
    P2 = [rnd.random(), rnd.random()]
    # 随机数生成两个控制点：P1、P2
    points = np.array([A, P1, P2, B])
    ax = fig.add_subplot(6, 3, idx + 1)
    x, y = points[:, 0], points[:, 1]
    bx, by = evaluate_bezier(points, 200)
    colors = plt.cm.RdYlBu(np.linspace(0,1,200, endpoint = True))
    plt.scatter(bx, by, c = colors)
    plt.plot(x, y, 'k:', marker = 'x')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal', adjustable='box')

### 四阶
fig = plt.figure(figsize = (10,16), tight_layout=True)
for idx in range(18):
    P1 = [rnd.random(), rnd.random()]
    P2 = [rnd.random(), rnd.random()]
    P3 = [rnd.random(), rnd.random()]
    points = np.array([A, P1, P2, P3, B])
    ax = fig.add_subplot(6, 3, idx + 1)
    x, y = points[:, 0], points[:, 1]
    bx, by = evaluate_bezier(points, 200)
    colors = plt.cm.RdYlBu(np.linspace(0,1,200, endpoint = True))
    plt.scatter(bx, by, c = colors)
    plt.plot(x, y, 'k:', marker = 'x')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal', adjustable='box')


#%% # RGB色彩空间中的贝塞尔曲线

# from Bezier import Bezier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

t_points = np.arange(0, 1, 0.005)
def Bezier_in_RGB_space(ax, start_point, end_point, B_order, filename):
    if start_point is None:
        P_array = np.random.random((B_order + 1,3))
    else:
        mid_points = np.random.random((B_order - 1,3))
        P_array = np.concatenate((start_point, mid_points, end_point), axis=0)
    curve_set_1 = Bezier.Curve(t_points, P_array)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(curve_set_1[:, 0], curve_set_1[:, 1], curve_set_1[:, 2], c = curve_set_1)
    ax.plot(P_array[:, 0], P_array[:, 1], P_array[:, 2], 'x:', c = 'k')
    ax.set_proj_type('ortho')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])

start_point = np.array([[0, 0, 0]])
end_point = np.array([[1, 1, 1]])
B_order = 1
Bezier_in_RGB_space(ax, start_point, end_point, B_order, '1阶贝塞尔')

start_point = np.array([[0, 0, 0]])
end_point = np.array([[1, 1, 1]])
B_order = 2
Bezier_in_RGB_space(ax, start_point, end_point, B_order, '2阶贝塞尔')


start_point = np.array([[0, 0, 0]])
end_point = np.array([[1, 1, 1]])
B_order = 3
Bezier_in_RGB_space(ax, start_point, end_point, B_order, '3阶贝塞尔')

start_point = np.array([[0, 0, 0]])
end_point = np.array([[1, 1, 1]])
B_order = 4
Bezier_in_RGB_space(ax, start_point, end_point, B_order, '4阶贝塞尔')

start_point = np.array([[0, 0, 0]])
end_point = np.array([[1, 1, 1]])
B_order = 5
Bezier_in_RGB_space(ax, start_point, end_point, B_order, '5阶贝塞尔')



#%% # 鸢尾花曲线
import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import factorial

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))

def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]

# 定义散点数量
randPntNum = 20
tPntNum = 500
tList = np.arange(1, tPntNum + 1)

# 产生连续变化的角度
randList1 = (np.random.rand(randPntNum) - 0.5) * 2 * np.pi
randList2 = (np.random.rand(randPntNum) - 0.5) * 2 * np.pi
randList3 = (np.random.rand(randPntNum) - 0.5) * 2 * np.pi
thetaList1 = np.interp(tList, np.linspace(1, randPntNum, randPntNum), randList1, period=2*np.pi)
thetaList2 = np.interp(tList, np.linspace(1, randPntNum, randPntNum), randList2, period=2*np.pi)
thetaList3 = np.interp(tList, np.linspace(1, randPntNum, randPntNum), randList3, period=2*np.pi)

# 设定色谱颜色
CList = np.array([[4, 99, 128],
                  [22, 25, 59],
                  [53, 71, 140],
                  [78, 122, 199],
                  [127, 178, 240],
                  [173, 213, 247]]) / 255
ti = np.linspace(1, tPntNum, CList.shape[0])
CList = np.column_stack([np.interp(tList, ti, CList[:, i], period=2*np.pi) for i in range(3)])

fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')

# 随机生成两个椭圆参数
A1 = np.random.rand()
B1 = np.random.rand()
A2 = np.random.rand()
B2 = np.random.rand()

# 产生贝塞尔曲线
for i in range(tPntNum):
    X = np.array([0, A1 * np.cos(thetaList1[i]), A2 * np.cos(thetaList2[i]), np.cos(thetaList3[i])])
    Y = np.array([0, B1 * np.sin(thetaList1[i]), B2 * np.sin(thetaList2[i]), np.sin(thetaList3[i])])
    bx, by = evaluate_bezier(np.column_stack([X, Y]), 200)
    plt.plot(bx, by, color=np.append(CList[i, :], 0.2), linewidth=0.5)

# fig.savefig('Figures/鸢尾花曲线，贝塞尔.svg', format='svg')






































































































































































































































































































































































































































































































































































































































































































