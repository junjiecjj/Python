



#%% 内旋轮线Hypotrochoid
import matplotlib.pyplot as plt
import numpy as np
from math import lcm
## 自定义函数
def draw_spirograph(R, r, d):
    num = lcm(R,r)/R
    print(num)
    theta = np.linspace(0,num*2*np.pi, 10000)
    theta_cir = np.linspace(0,2*np.pi,180)
    x = (R-r) * np.cos(theta) + d * np.cos((R-r)/r * theta)
    y = (R-r) * np.sin(theta) - d * np.sin((R-r)/r * theta)
    fig, ax = plt.subplots(figsize = (3,3))
    plt.plot(0, 0, marker = 'x', markersize = 6)
    plt.plot(x, y, c='k')
    # 大圆圆周
    plt.plot(np.cos(theta_cir) * R,
             np.sin(theta_cir) * R,
             lw = 0.2, c = 'b', ls = '--')
    # 小圆圆心运动轨迹
    plt.plot(np.cos(theta_cir) * (R - r),
             np.sin(theta_cir) * (R - r),
             lw = 0.2, c = 'k', ls = 'dashed')
    plt.plot(np.cos(theta_cir) * r + (R - r),
             np.sin(theta_cir) * r,
             lw = 1, c = 'r')
    plt.plot(R - r, 0, marker = 'x', c = 'r', markersize = 6)
    # P点位置
    plt.plot(R - r + d, 0, marker = '.', c = 'k', markersize = 6)

    plt.xlim(-R-3,R+3)
    plt.ylim(-R-3,R+3)
    # plt.axis('equal')
    plt.axis('off')
    plt.savefig('R = ' + str(R) + ', r = ' + str(r) + ', d = ' + str(d) + '.svg')
    plt.show()

draw_spirograph(2, 1, 0)
draw_spirograph(2, 1, 2)

draw_spirograph(3, 1, 2)

draw_spirograph(4, 1, 0.5)

draw_spirograph(6, 1, 0)


#%% # 内旋轮线Hypotrochoid
import matplotlib.pyplot as plt
import numpy as np
from math import lcm
from matplotlib.collections import LineCollection

def draw_spirograph(R, r, d, Ratio = 1):
    num = lcm(R,r)/R
    print(num)
    theta = np.linspace(0,num*2*np.pi*Ratio, 10000)
    colors = np.arange(0, 1, 1/len(theta))
    x = (R-r) * np.cos(theta) + d * np.cos((R-r)/r * theta)
    y = (R-r) * np.sin(theta) - d * np.sin((R-r)/r * theta)
    dist = np.sqrt(x**2 + y**2)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(theta.min(), theta.max())
    lc = LineCollection(segments, cmap='hsv', norm=norm)
    # Set the values used for colormapping
    lc.set_array(theta)
    lc.set_linewidth(0.25)

    fig, ax = plt.subplots(figsize = (3,3))
    ax.add_collection(lc)
    # plt.plot(x, y, c='k')
    # colors = np.arange(0, 1, 1/len(theta))
    # plt.scatter(x, y, c=colors, cmap=color_map, s=5)
    # plt.colorbar()
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig('R = ' + str(R) + ', r = ' + str(r) + ', d = ' + str(d) + '.svg')
    plt.show()

draw_spirograph(23, 9, 13)

draw_spirograph(57, 29, 9)

draw_spirograph(57, 29, 19)

draw_spirograph(59, 29, 29)

draw_spirograph(59, 19, 19)




#%% # 外旋轮线Epitrochoid
import matplotlib.pyplot as plt
import numpy as np
from math import lcm
## 自定义函数

def draw_spirograph(R, r, d):
    num = lcm(R,r)/R
    print(num)
    theta = np.linspace(0,num*2*np.pi, 10000)
    theta_cir = np.linspace(0,2*np.pi,180)

    x = (R+r) * np.cos(theta) - d * np.cos((R+r)/r * theta)
    y = (R+r) * np.sin(theta) - d * np.sin((R+r)/r * theta)
    fig, ax = plt.subplots(figsize = (3,3))
    plt.plot(0, 0, marker = 'x', markersize = 6)
    plt.plot(x, y, c='k')
    # 大圆圆周
    plt.plot(np.cos(theta_cir) * R,
             np.sin(theta_cir) * R,
             lw = 0.2, c = 'b', ls = '--')
    # 小圆圆心运动轨迹
    plt.plot(np.cos(theta_cir) * (R + r),
             np.sin(theta_cir) * (R + r),
             lw = 0.2, c = 'k', ls = 'dashed')
    plt.plot(np.cos(theta_cir) * r + (R + r),
             np.sin(theta_cir) * r,
             lw = 1, c = 'r')
    plt.plot(R + r, 0, marker = 'x', c = 'r', markersize = 6)
    # P点位置
    plt.plot(R + r - d, 0, marker = '.', c = 'k', markersize = 6)
    plt.xlim(-R-3,R+3)
    plt.ylim(-R-3,R+3)
    # plt.axis('equal')
    plt.axis('off')
    # plt.savefig('R = ' + str(R) + ', r = ' + str(r) + ', d = ' + str(d) + '.svg')
    plt.show()

draw_spirograph(1, 1, 0)

draw_spirograph(1, 1, 2)

draw_spirograph(2, 1, 1)

draw_spirograph(4, 1, 0)

draw_spirograph(6, 1, 2)


#%% # 外旋轮线Epitrochoid
import matplotlib.pyplot as plt
import numpy as np
from math import lcm
from matplotlib.collections import LineCollection


def draw_spirograph(R, r, d, Ratio = 1):
    num = lcm(R,r)/R
    print(num)
    theta = np.linspace(0,num*2*np.pi*Ratio, 10000)
    colors = np.arange(0, 1, 1/len(theta))
    x = (R+r) * np.cos(theta) - d * np.cos((R+r)/r * theta)
    y = (R+r) * np.sin(theta) - d * np.sin((R+r)/r * theta)
    dist = np.sqrt(x**2 + y**2)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(theta.min(), theta.max())

    lc = LineCollection(segments, cmap='hsv', norm=norm)
    # Set the values used for colormapping
    lc.set_array(theta)
    lc.set_linewidth(0.25)

    fig, ax = plt.subplots(figsize = (3,3))
    ax.add_collection(lc)
    # plt.plot(x, y, c='k')
    # colors = np.arange(0, 1, 1/len(theta))
    # plt.scatter(x, y, c=colors, cmap=color_map, s=5)
    # plt.colorbar()
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig('R = ' + str(R) + ', r = ' + str(r) + ', d = ' + str(d) + '.svg')
    plt.show()

draw_spirograph(23, 9, 13)

draw_spirograph(57, 29, 39)

draw_spirograph(59, 29, 29)



















































#%%
































#%%
































#%%















































































































































































































































































































































































































































































































































































































































































































