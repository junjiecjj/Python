




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用patches绘制平面几何形状
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# Prepare the data for the PathPatch below.
Path = mpath.Path
codes, verts = zip(*[
    (Path.MOVETO, [0.018, -0.11]),
    (Path.CURVE4, [-0.031, -0.051]),
    (Path.CURVE4, [-0.115, 0.073]),
    (Path.CURVE4, [-0.03, 0.073]),
    (Path.LINETO, [-0.011, 0.039]),
    (Path.CURVE4, [0.043, 0.121]),
    (Path.CURVE4, [0.075, -0.005]),
    (Path.CURVE4, [0.035, -0.027]),
    (Path.CLOSEPOLY, [0.018, -0.11])])

artists = [
    mpatches.Circle((0, 0), 0.1, ec="none"),
    mpatches.Rectangle((-0.025, -0.05), 0.05, 0.1, ec="none"),
    mpatches.Wedge((0, 0), 0.1, 30, 270, ec="none"),
    mpatches.RegularPolygon((0, 0), 5, radius=0.1),
    mpatches.Ellipse((0, 0), 0.2, 0.1),
    mpatches.Arrow(-0.05, -0.05, 0.1, 0.1, width=0.1),
    mpatches.PathPatch(mpath.Path(verts, codes), ec="none"),
    mpatches.FancyBboxPatch((-0.025, -0.05), 0.05, 0.1, ec="none", boxstyle=mpatches.BoxStyle("Round", pad=0.02)),
    mlines.Line2D([-0.06, 0.0, 0.1], [0.05, -0.05, 0.05], lw=5),
]

axs = plt.figure(figsize = (6, 6)).subplots(3, 3)
for i, (ax, artist) in enumerate(zip(axs.flat, artists)):
    artist.set(color = mpl.cm.get_cmap('hsv')(i / len(artists)))
    ax.add_artist(artist)
    ax.set(title = type(artist).__name__, aspect = 1, xlim = (-.2, .2), ylim = (-.2, .2))
    ax.set_axis_off()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 利用patches绘制正圆，以及外切、内接正多边形
# 导入包
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import numpy as np

# 可视化
fig, axs = plt.subplots(nrows = 1, ncols = 4)
for num_vertices, ax in zip([4,5,6,8], axs.ravel()):
    hexagon_inner = RegularPolygon((0,0), numVertices=num_vertices, radius=1, alpha=0.2, edgecolor='k')
    ax.add_patch(hexagon_inner)
    # 绘制正圆内接多边形

    hexagon_outer = RegularPolygon((0,0), numVertices=num_vertices, radius=1/np.cos(np.pi/num_vertices), alpha=0.2, edgecolor='k')
    ax.add_patch(hexagon_outer)
    # 绘制正圆外切多边形

    circle = Circle((0,0), radius=1, facecolor = 'none', edgecolor='k')
    ax.add_patch(circle)
    # 绘制正圆

    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正圆的生成艺术

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

delta_angle = 5 # degrees

fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
range_array = np.arange(10)
ax.plot(0, 0,  color = 'r', marker = 'o', markersize = 10)
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))
for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = plt.Circle(point_of_rotation, radius = width, fill = False, edgecolor = colors[i], transform = Affine2D().rotate_deg_around(0, 0, deg) + ax.transData)
    ax.add_patch(rec)
plt.axis('off')
plt.show()


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(100)
delta_angle = 3.6 # degrees

colors = plt.cm.hsv(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 2
    point_of_rotation = np.array([0, -width])
    rec = plt.Circle(point_of_rotation, radius=width, fill = False, edgecolor = colors[i], transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
plt.show()


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='b', marker='o', markersize=10)

range_array = np.arange(36)
delta_angle = 10 # degrees
colors = plt.cm.hsv(np.linspace(0, 1, len(range_array)))
for i in range_array:
    deg = delta_angle * i
    width = 2
    point_of_rotation = np.array([0, -width])
    rec = plt.Circle(point_of_rotation, radius=width, fill = True, edgecolor = 'b', alpha = 0.1, transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 椭圆的生成艺术

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection

num = 30
x = np.arange(num)
y = np.arange(num)
X, Y = np.meshgrid(x, y)
XY = np.column_stack((X.ravel(), Y.ravel()))
btm = 0.3
ww = (1 - btm) * X / num + btm
hh = (1 - btm) * Y / num + btm
aa = X * 2*np.pi


fig, ax = plt.subplots(figsize = (8,8))
ec = EllipseCollection(ww, hh, aa, units='x', offsets=XY, transOffset=ax.transData,cmap = 'RdYlBu')
ec.set_array((X**2 + Y**2).ravel())
ax.add_collection(ec)
ax.autoscale_view()
ax.set_xlabel('X')
ax.set_ylabel('y')
# cbar = plt.colorbar(ec)
# cbar.set_label('X+Y')
plt.axis('off')
# fig.savefig('Figures/一组椭圆.svg', format='svg')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 椭圆的生成艺术，两组

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 10 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = Ellipse(point_of_rotation, width=width, height = width * 1.5, fill = False, edgecolor = colors[i], transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转椭圆_A.svg', format='svg')
plt.show()



fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(200)
delta_angle = 30 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))
for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = Ellipse(point_of_rotation, width=width, height = width * 1.5, fill = False, edgecolor = colors[i], transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)

plt.axis('off')
# fig.savefig('Figures/旋转椭圆_B.svg', format='svg')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 正方形的生成艺术，两组

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color = 'r', marker = 'o', markersize = 10)

range_array = np.arange(10)
delta_angle = 2.5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    point_of_rotation = np.array([-width/2, -width/2])
    rec = plt.Rectangle(point_of_rotation, width=width, height=width, fill = False, edgecolor = colors[i], transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)
# plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(10)
delta_angle = 5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))
for i in range_array:
    deg = delta_angle * i
    width = 0.05 + i * 0.05
    rec = plt.Rectangle((0,0), width=width, height=width, fill = False, edgecolor = colors[i], transform=Affine2D().rotate_deg_around(0,0, deg)+ax.transData)
    ax.add_patch(rec)
# plt.axis('off')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 沿横轴填充

# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 产生两个函数数据
x_array = np.linspace(0, 4*np.pi, 1001)
y_array = np.sin(x_array)
y2_array = np.sin(2*x_array)

# 同一个颜色
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
plt.fill_between(x_array, 0, y_array, color='#0088FF', alpha=.25)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')


# 调整直线高度，增加阴影
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
plt.fill_between(x_array, -1.2, y_array, color='#0088FF', alpha=.25, hatch = '///')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array, color='#0088FF')
plt.fill_between(x_array, 1.2, y_array, color='r', alpha=.25, hatch = '///')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 双色
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
plt.fill_between(x_array, 0, y_array, y_array > 0, color='#0088FF', alpha=.25, hatch = '///')
plt.fill_between(x_array, 0, y_array, y_array < 0, color='r', alpha=.25, hatch = '///')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')


# 两条曲线
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
plt.plot(x_array, y2_array, color='#0088FF')
plt.fill_between(x_array, y2_array, y_array, color='#0088FF', alpha=.25)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')


fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array, color='#0088FF')
plt.plot(x_array, y2_array, color='#0088FF')
plt.fill_between(x_array, y2_array, y_array, y_array > y2_array, color='#0088FF', alpha=.25, hatch = '///')
plt.fill_between(x_array, y2_array, y_array, y_array < y2_array, color='r', alpha=.25, hatch = '///')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 沿纵轴填充

#导入包

import matplotlib.pyplot as plt
import numpy as np

# 产生两个函数数据
x_array = np.linspace(0, 4*np.pi, 1001)
y_array = np.sin(x_array)
y2_array = np.sin(2*x_array)

fig, [ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(1, 6, sharey=True, figsize=(8, 6))

ax1.plot(y_array, x_array, color = '#0088FF')
ax1.fill_betweenx(x_array, y_array, 0, color='#0088FF', alpha=.25)
ax1.set_ylim((0,4*np.pi))
ax1.set_xlim((-1.2, 1.2))

ax2.plot(y_array, x_array, color = '#0088FF')
ax2.fill_betweenx(x_array, y_array, 1.2, color='#0088FF', alpha=.25)
ax2.set_ylim((0,4*np.pi))
ax2.set_xlim((-1.2, 1.2))

ax3.plot(y_array, x_array, color = '#0088FF')
ax3.fill_betweenx(x_array, y_array, -1.2, color='#0088FF', alpha=.25)
ax3.set_ylim((0,4*np.pi))
ax3.set_xlim((-1.2, 1.2))

ax4.plot(y_array, x_array, color = '#0088FF')
ax4.fill_betweenx(x_array, y_array, 0, y_array > 0, color='#0088FF', alpha=.25)
ax4.fill_betweenx(x_array, y_array, 0, y_array < 0, color='r', alpha=.25)
# , hatch = '///'
ax4.set_ylim((0,4*np.pi))
ax4.set_xlim((-1.2, 1.2))

ax5.plot(y_array, x_array, color = '#0088FF')
ax5.plot(y2_array, x_array, color = '#0088FF')
ax5.fill_betweenx(x_array, y_array, y2_array, color='#0088FF', alpha=.25)
ax5.set_ylim((0,4*np.pi))
ax5.set_xlim((-1.2, 1.2))

ax6.plot(y_array, x_array, color = '#0088FF')
ax6.plot(y2_array, x_array, color = '#0088FF')
ax6.fill_betweenx(x_array, y_array, y2_array, y_array > y2_array, color='#0088FF', alpha=.25)
ax6.fill_betweenx(x_array, y_array, y2_array, y_array < y2_array, color='r', alpha=.25)
ax6.set_ylim((0,4*np.pi))
ax6.set_xlim((-1.2, 1.2))
# fig.savefig('Figures/沿纵轴填充.svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 阴影
# 导入包
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 定义函数
def hatches_plot(ax, hatch_style):
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=hatch_style))
    # 增加长方形，始于 (0, 0)，长宽均为2
    ax.text(1, 2.5, f"' {hatch_style} '", size=15, ha="center")
    ax.axis('equal')
    ax.axis('off')

hatches = ['/', '\\', '|', '-', '+',
           'x', 'o', 'O', '.', '*',
           '//', '\\\\', '||', '--', '++',
           'xx', 'oo', 'OO', '..', '**',
           '/o', '\\|', '|*', '-\\', '+o',
           'x*', 'o-', 'O|', 'O.', '*-']

fig, axs = plt.subplots(6, 5, constrained_layout=True, figsize=(6, 12))

for ax, hatch_style in zip(axs.flat, hatches):
    hatches_plot(ax, hatch_style)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 参考填充色块
# 导入包
import matplotlib.pyplot as plt
import numpy as np

x_array = np.linspace(0, 4*np.pi, 101)
# 等差数列的公差为 4*pi/100；数列有101个值
y_array = np.sin(x_array)
# 水平填充
fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array, color='#0088FF')
ax.axhspan(-0.5, 0.5, facecolor='0.8')
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 竖直填充
fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array, color='#0088FF')
ax.axvspan(0, np.pi, facecolor='0.8')
ax.axvspan(np.pi*2, np.pi*3, facecolor='0.8')
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 展示旋转
import matplotlib.pyplot as plt
import numpy as np

# 定义函数
def plot_shape(X, copy = False):
    if copy:
        fill_color = np.array([255, 236, 255])/255
        edge_color = np.array([255, 0, 0])/255
    else:
        fill_color = np.array([219, 238, 243])/255
        edge_color = np.array([0, 153, 255])/255
    plt.fill(X[:,0], X[:,1], color = fill_color, edgecolor = edge_color, alpha = 0.5)
    plt.plot(X[:,0], X[:,1],marker = 'x', markeredgecolor = edge_color*0.5, linestyle = 'None')
X = np.array([[1,1],
              [0,-1],
              [-1,-1],
              [-1,1]])
X = X + [2, 2]

# 可视化
num = 24
thetas = np.linspace(360/num, 360, num, endpoint = False)
fig, ax = plt.subplots(figsize = (10, 10))
plot_shape(X)      # plot original
for theta in thetas:
    theta = theta/180*np.pi;
    # rotation
    R = np.array([[np.cos(theta),  np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    Z = X@R
    # 旋转
    plot_shape(Z,True) # plot copy

# 装饰
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.axhline(y=0, color='k', linewidth = 0.25)
plt.axvline(x=0, color='k', linewidth = 0.25)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])

ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 展示最小二乘
import numpy as np
import matplotlib.pyplot as plt

num_data = 60;

rng = np.random.default_rng(2)
e = rng.normal(0,0.5, num_data)

x_points = rng.normal(1,1, num_data);
xx = np.linspace(-2,5,10);

b1 = 0.75;
b0 = 1;
y_points = b1*x_points + b0 + e;
y_hat_points = b1*x_points + b0;
yy = b1*xx + b0;

def plot_square(x,y1,y2):
    if y2 > y1:
        temp = y2;
        y2 = y1;
        y1 = temp;
    d = y1 - y2;
    plt.fill(np.vstack((x, x + d, x + d, x)), np.vstack((y2, y2, y1, y1)), facecolor='#0088FF', edgecolor='none', alpha = 0.2)

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(x_points,y_points,'x');
plt.plot(x_points,y_hat_points,'xr');
plt.plot(xx,yy,'r');

plt.plot(np.vstack((x_points,x_points)),
      np.vstack((y_points,y_hat_points)),
      color = '#0088FF');

for i in range(0,num_data):
    plot_square(x_points[i],y_points[i],y_hat_points[i]);

plt.axis('scaled')
plt.xlim(-2,5)
plt.ylim(-1,5)
ax.set_xticks([])
ax.set_yticks([])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  平面仿射变换

# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 产生网格数据
x1 = np.arange(-20, 20 + 1, step = 1)
x2 = np.arange(-20, 20 + 1, step = 1)

XX1, XX2 = np.meshgrid(x1,x2)
X = np.column_stack((XX1.ravel(), XX2.ravel()))

# 自定义可视化函数
def visualize_transform(XX1, XX2, ZZ1, ZZ2, cube, arrows, fig_name):
    colors = np.arange(len(XX1.ravel()))
    fig, ax = plt.subplots(figsize = (5,5))
    # 绘制原始网格
    plt.plot(XX1 ,XX2, color = [0.8,0.8,0.8], lw = 0.25)
    plt.plot(XX1.T, XX2.T, color = [0.8,0.8,0.8], lw = 0.25)
    # plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'plasma', zorder=1e3)

    #绘制几何变换后的网格
    plt.plot(ZZ1, ZZ2, color = '#0070C0', lw = 0.25)
    plt.plot(ZZ1.T, ZZ2.T, color = '#0070C0', lw = 0.25)

    ax.fill(cube[:,0], cube[:,1], color = '#92D050', alpha = 0.5)
    ax.quiver(0, 0, arrows[0,0], arrows[0,1], color = 'r', angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, arrows[1,0], arrows[1,1], color = 'g', angles='xy', scale_units='xy', scale=1)

    plt.axis('scaled')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.axhline(y = 0, color = 'k')
    ax.axvline(x = 0, color = 'k')
    plt.xticks([])
    plt.yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # fig.savefig('Figures/' + fig_name + '.svg', format='svg')

#>>>>>>>>>>> 原始网格
colors = np.arange(len(XX1.ravel()))
fig, ax = plt.subplots(figsize = (5,5))
cube = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
arrows = np.array([[1, 0], [0, 1]])

# 绘制原始网格
plt.plot(XX1, XX2, color = '#0070C0', lw = 0.25)
plt.plot(XX1.T, XX2.T, color = '#0070C0', lw = 0.25)
ax.fill(cube[:,0], cube[:,1], color = '#92D050', alpha = 0.5)
ax.quiver(0,0,arrows[0,0], arrows[0,1], color = 'r', angles='xy', scale_units='xy', scale=1)
ax.quiver(0,0,arrows[1,0], arrows[1,1], color = 'g', angles='xy', scale_units='xy', scale=1)

plt.axis('scaled')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#>>>>>>>>>>> 旋转
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
Z = X@R.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '逆时针旋转30度'

cube_ = cube @ R.T;
arrows_ = arrows @ R.T;
visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)
#>>>>>>>>>>> 等比例放大
S = np.array([[2, 0], [0, 2]])
Z = X@S
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例放大'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 等比例缩小
S = np.array([[0.4, 0], [0,   0.4]])
Z = X@S;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例缩小'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 非等比例缩放
S = np.array([[2, 0], [0, 0.5]])
Z = X@S;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '非等比例缩放'
cube_ = cube @ S.T;
arrows_ = arrows @ S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 先缩放，再旋转
Z = X@S.T@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先缩放，再旋转'
cube_ = cube @S.T@R.T;
arrows_ = arrows @S.T@R.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 先旋转，再放大
Z = X@R.T@S.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再缩放'
cube_ = cube @R.T@S.T;
arrows_ = arrows @R.T@S.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)

#>>>>>>>>>>> 沿横轴剪切
T = np.array([[1, 1.5], [0, 1]])
Z = X@T.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿横轴剪切'
cube_ = cube @T.T;
arrows_ = arrows @T.T;

visualize_transform(XX1, XX2, ZZ1, ZZ2, cube_, arrows_, fig_name)
















































































































































































