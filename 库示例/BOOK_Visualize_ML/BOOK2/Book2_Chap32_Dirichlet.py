


#%% 平面上可视化Dirichlet分布
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt


theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1,tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None

# 定义可视化函数

def plot_Dirichlet_PDF_contour(alpha_array):

    PDF = dirichlet.pdf([tt1.ravel(),tt2.ravel(),tt3.ravel()], alpha_array)

    fig, ax = plt.subplots(figsize = (5,5))

    plt.contour(tt1, tt2, PDF.reshape(tt1.shape),
                 levels = 20,
                 cmap='RdYlBu_r')
    plt.plot((0,1),(1,0), c = 'k',lw = 1.25)
    plt.axis('equal')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')

    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig('Figures/平面等高线_' + title + '.svg', format='svg')


### 不同参数
alpha_array = [1, 2, 4]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [4,1,2]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [2,4,1]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [2,4,8]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [4,8,2]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [8,2,4]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [2,2,2]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [8,8,8]
plot_Dirichlet_PDF_contour(alpha_array)



#%% # Dirichlet分布PDF投影到斜面
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1,tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None

### 自定义可视化函数


def visualize_Dirichlet(alpha_array, num = 50):

    PDF = dirichlet.pdf([tt1.ravel(),tt2.ravel(),tt3.ravel()], alpha_array)

    fig = plt.figure(figsize=(5, 5))

    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape),
                 levels = 20,
                 cmap='RdYlBu_r')

    plt.clf()

    norm = Normalize(vmin=all_contours.levels.min(),
                                       vmax=all_contours.levels.max(),
                                       clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

    ax = plt.axes(projection="3d")

    for level_idx, ctr_idx in zip(all_contours.levels,
                                  all_contours.allsegs):

        for i in range(0,len(ctr_idx)):

            t1_i,t2_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i

            # 绘制映射结果
            ax.plot(t1_i, t2_i, t3_i,
                    color = mapper.to_rgba(level_idx),
                    linewidth = 1)

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot((1,0),(0,1),(0,0),lw = 1,c = 'k')
    ax.plot((0,0),(1,0),(0,1),lw = 1,c = 'k')
    ax.plot((1,0),(0,0),(0,1),lw = 1,c = 'k')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig('Figures/斜面等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)

alpha_array = [4,1,2]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,1]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,8]
visualize_Dirichlet(alpha_array)


alpha_array = [4,8,2]
visualize_Dirichlet(alpha_array)

alpha_array = [8,2,4]
visualize_Dirichlet(alpha_array)


alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)

alpha_array = [8,8,8]
visualize_Dirichlet(alpha_array)



#%% # 斜面等高线 + x1x2等高线
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0.25,
                           0.25,
                           0.25]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs
if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new
###patch end###

theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1,tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None


### 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):

    PDF = dirichlet.pdf([tt1.ravel(),tt2.ravel(),tt3.ravel()], alpha_array)

    fig = plt.figure(figsize=(5, 5))

    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape), levels = 20, cmap='RdYlBu_r')

    plt.clf()

    norm = Normalize(vmin=all_contours.levels.min(), vmax=all_contours.levels.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

    ax = plt.axes(projection="3d")

    ax.contour(tt1, tt2, PDF.reshape(tt1.shape), zdir = 'z', offset = 0, levels = 20, zorder = 1, cmap='RdYlBu_r')

    for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):
        for i in range(0,len(ctr_idx)):
            t1_i,t2_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i
            # 绘制映射结果
            ax.plot(t1_i, t2_i, t3_i, color = mapper.to_rgba(level_idx), linewidth = 1, zorder = 1000)

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot((1,0),(0,1),(0,0),lw = 1,c = 'k')
    ax.plot((0,0),(1,0),(0,1),lw = 1,c = 'k')
    ax.plot((1,0),(0,0),(0,1),lw = 1,c = 'k')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig('Figures/斜面等高线 + 12水平等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)

alpha_array = [4,1,2]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,1]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,8]
visualize_Dirichlet(alpha_array)

alpha_array = [4,8,2]
visualize_Dirichlet(alpha_array)

alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)


alpha_array = [8,8,8]
visualize_Dirichlet(alpha_array)

#%% # 斜面等高线 + x1x3等高线
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0.25,
                           0.25,
                           0.25]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs
if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new
###patch end###


theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1,tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None


### 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):

    PDF = dirichlet.pdf([tt1.ravel(),tt2.ravel(),tt3.ravel()], alpha_array)

    fig = plt.figure(figsize=(5, 5))

    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape),
                 levels = 20,
                 cmap='RdYlBu_r')

    plt.clf()

    norm = Normalize(vmin=all_contours.levels.min(),
                                       vmax=all_contours.levels.max(),
                                       clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

    ax = plt.axes(projection="3d")

    ax.contour(tt1, PDF.reshape(tt1.shape), tt3, zdir = 'y', offset = 0,
                 levels = 20, zorder = 1,
                 cmap='RdYlBu_r')

    for level_idx, ctr_idx in zip(all_contours.levels,
                                  all_contours.allsegs):

        for i in range(0,len(ctr_idx)):

            t1_i,t2_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i

            # 绘制映射结果
            ax.plot(t1_i, t2_i, t3_i,
                    color = mapper.to_rgba(level_idx),
                    linewidth = 1, zorder = 1000)

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot((1,0),(0,1),(0,0),lw = 1,c = 'k')
    ax.plot((0,0),(1,0),(0,1),lw = 1,c = 'k')
    ax.plot((1,0),(0,0),(0,1),lw = 1,c = 'k')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig('Figures/斜面等高线 + 13水平等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,8]
visualize_Dirichlet(alpha_array)

alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)

alpha_array = [8,8,8]
visualize_Dirichlet(alpha_array)



#%% # 斜面等高线 + x2x3等高线
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0.25,
                           0.25,
                           0.25]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs
if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new
###patch end###

theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1,tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None


### 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):

    PDF = dirichlet.pdf([tt1.ravel(),tt2.ravel(),tt3.ravel()], alpha_array)

    fig = plt.figure(figsize=(5, 5))

    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape),
                 levels = 20,
                 cmap='RdYlBu_r')

    plt.clf()

    norm = Normalize(vmin=all_contours.levels.min(),
                                       vmax=all_contours.levels.max(),
                                       clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

    ax = plt.axes(projection="3d")

    ax.contour(PDF.reshape(tt1.shape), tt2, tt3, zdir = 'x', offset = 0,
                 levels = 20, zorder = 1,
                 cmap='RdYlBu_r')

    for level_idx, ctr_idx in zip(all_contours.levels,
                                  all_contours.allsegs):

        for i in range(0,len(ctr_idx)):

            t1_i,t2_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i

            # 绘制映射结果
            ax.plot(t1_i, t2_i, t3_i,
                    color = mapper.to_rgba(level_idx),
                    linewidth = 1, zorder = 1000)

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot((1,0),(0,1),(0,0),lw = 1,c = 'k')
    ax.plot((0,0),(1,0),(0,1),lw = 1,c = 'k')
    ax.plot((1,0),(0,0),(0,1),lw = 1,c = 'k')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig('Figures/斜面等高线 + 23水平等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)


alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)



#%%# 重心坐标系
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


## 定义三角形

# 等边三角形
corners = np.array([[0, 0], [1, 0], [0.5,0.75**0.5]]).T
# 等腰直角三角形
# corners = np.array([[0, 0], [1, 0], [0, 1]]).T
# 任意三角形
# corners = np.array([[0, 0.2], [0.9, 0], [0.3, 0.8]]).T
# corners

## 三角网格
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)
trimesh_2 = refiner.refine_triangulation(subdiv=2)
# 每个列向量代表一个三角网格坐标点
r_array = np.row_stack((trimesh_2.x,trimesh_2.y))

### 直角坐标系坐标

fig, ax = plt.subplots(figsize = (5,5))
plt.triplot(trimesh_2)
plt.plot(r_array[0,:],
         r_array[1,:],
         '.r',
         markersize = 10)

for x_idx, y_idx in zip(trimesh_2.x, trimesh_2.y):

    text_idx = ('(' + format(x_idx, '.2f') +
                ', ' + format(y_idx, '.2f') + ')')
    plt.text(x_idx, y_idx+0.03,
             text_idx,
             fontsize = 8,
             horizontalalignment = 'center',
             bbox=dict(facecolor='w', alpha=0.5, edgecolor = 'None'))
ax.set_aspect('equal')
ax.set_xlim(0,1); ax.set_ylim(0,1)
# plt.savefig('1.svg')
plt.show()


### 坐标转换
# 提取大三角形的三个顶点列向量 (坐标点)
r1 = corners[:,[0]]
r2 = corners[:,[1]]
r3 = corners[:,[2]]
# 构造矩阵T
T = np.column_stack((r1 - r3,r2 - r3))
# 计算 theta_1和theta_2
theta_1_2 = np.linalg.inv(T) @ (r_array - r3)
# 计算theta_3
theta_3 = 1 - theta_1_2[0,:] - theta_1_2[1,:]

# 创建theta坐标，每一列代表一个重心坐标系坐标
# r_array和theta_1_2_3每一列一一对应
theta_1_2_3 = np.row_stack((theta_1_2,theta_3))

# 对重心坐标进行截断，避免超出 [0,1] 区间
theta_1_2_3 = np.clip(theta_1_2_3, 1e-6, 1.0 - 1e-6)
# numpy.clip(a, a_min, a_max, out=None)
# 其中a是一个数组，后面两个参数分别表示最小和最大值，
# 将数组中的元素限制在a_min, a_max之间，
# 大于a_max的就使得它等于 a_max，
# 小于a_min,的就使得它等于a_min

### 重心坐标系坐标
fig, ax = plt.subplots(figsize = (5,5))

plt.triplot(trimesh_2)
plt.plot(r_array[0,:],
         r_array[1,:],
         '.r',
         markersize = 10)

for theta_idx,x_idx,y_idx in zip(theta_1_2_3.T,
                                 trimesh_2.x,
                                 trimesh_2.y):

    theta_1 = theta_idx[0]
    theta_2 = theta_idx[1]
    theta_3 = theta_idx[2]

    text_idx = ('(' + format(theta_1, '.2f') +
    ', ' + format(theta_2, '.2f') +
    ', ' + format(theta_3, '.2f') + ')')

    plt.text(x_idx, y_idx+0.03, text_idx,
             fontsize = 7,
             horizontalalignment = 'center',
             bbox=dict(facecolor='w',
                       alpha=0.5,
                       edgecolor = 'None'))
plt.axis('equal')
plt.xlim(0, 1)
plt.text(0.9, 0.45,  r'$\theta_1$')
plt.text(0.1, 0.45, r'$\theta_2$')
plt.text(0.5, -0.1,  r'$\theta_3$')

ax.set_aspect('equal')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
# plt.savefig('2.svg')
plt.show()

#%%# # 重心坐标系中混合红绿蓝
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# 等边三角形
corners = np.array([[0, 0], [1, 0], [0.5,0.75**0.5]]).T
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)

fig, axs = plt.subplots(3,2,figsize=(6,9))

for subdiv,ax in zip(range(1,7), axs.ravel()):

    trimesh_2 = refiner.refine_triangulation(subdiv=subdiv)
    r_array = np.row_stack((trimesh_2.x,trimesh_2.y))

    r1 = corners[:,[0]]
    r2 = corners[:,[1]]
    r3 = corners[:,[2]]

    T = np.column_stack((r1 - r3,r2 - r3))

    theta_1_2 = np.linalg.inv(T) @ (r_array - r3)
    theta_3 = 1 - theta_1_2[0,:] - theta_1_2[1,:]
    theta_1_2_3 = np.row_stack((theta_1_2,theta_3))
    theta_1_2_3 = np.clip(theta_1_2_3, 1e-6, 1.0 - 1e-6)

    ax.scatter(r_array[0,:],r_array[1,:], c=theta_1_2_3.T, s=3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.01,1.05); ax.set_ylim(-0.01,1.05)
    ax.axis('off')
# fig.savefig('RGB.svg')




#%%# 可视化Dirichlet分布
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


### 构造重心坐标系
# 定义等边三角形
corners = np.array([[0, 0], [1, 0], [0.5,0.75**0.5]]).T
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)
trimesh_8 = refiner.refine_triangulation(subdiv=8)


# 自定义函数
def xy2bc(trimesh_8):

    # 每个列向量代表一个三角网格坐标点
    r_array = np.row_stack((trimesh_8.x,trimesh_8.y))

    r1 = corners[:,[0]]
    r2 = corners[:,[1]]
    r3 = corners[:,[2]]
    # 坐标转换
    T = np.column_stack((r1 - r3,r2 - r3))
    theta_1_2 = np.linalg.inv(T) @ (r_array - r3)
    theta_3 = 1 - theta_1_2[0,:] - theta_1_2[1,:]

    theta_1_2_3 = np.row_stack((theta_1_2,theta_3))
    theta_1_2_3 = np.clip(theta_1_2_3, 1e-6, 1.0 - 1e-6)
    theta_1_2_3 = theta_1_2_3/theta_1_2_3.sum(axis = 0)
    # 归一化

    return theta_1_2_3

# 定义可视化函数

def plot_Dirichlet_PDF_contour(alpha_array):

    PDF = dirichlet.pdf(xy2bc(trimesh_8), alpha_array)

    fig, ax = plt.subplots(figsize = (5,5))

    plt.tricontourf(trimesh_8, PDF,
                    levels = 20,
                    cmap='RdYlBu_r')
    plt.axis('equal')
    plt.xlim(0, 1); plt.ylim(0, 0.75**0.5)
    plt.text(0.8, 0.45,  r'$\theta_1$')
    plt.text(0.15, 0.45, r'$\theta_2$')
    plt.text(0.5, -0.1,  r'$\theta_3$')
    plt.axis('off'); plt.title(alpha_array)


alpha_array = [1, 1, 2]
plot_Dirichlet_PDF_contour(alpha_array)


alpha_array = [2, 2, 2]
plot_Dirichlet_PDF_contour(alpha_array)































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































