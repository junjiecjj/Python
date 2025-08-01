


#%% 平面上可视化Dirichlet分布 Bk_2_Ch32_01
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
# 确定随机数种子，保证结果可复刻
np.random.seed(42)

theta_1_array = np.linspace(0,1,1001)
theta_2_array = np.linspace(0,1,1001)
tt1, tt2 = np.meshgrid(theta_1_array,theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None

# 定义可视化函数
def plot_Dirichlet_PDF_contour(alpha_array):
    PDF = dirichlet.pdf([tt1.ravel(), tt2.ravel(), tt3.ravel()], alpha_array)
    fig, ax = plt.subplots(figsize = (5,5))
    plt.contour(tt1, tt2, PDF.reshape(tt1.shape), levels = 20, cmap='RdYlBu_r')
    plt.plot((0, 1),(1, 0), c = 'k',lw = 1.25)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')

    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    plt.title(title)
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


plt.close('all')

#%% # Dirichlet分布PDF投影到斜面 Bk_2_Ch32_02
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

theta_1_array = np.linspace(0, 1, 1001)
theta_2_array = np.linspace(0, 1, 1001)
tt1,tt2 = np.meshgrid(theta_1_array, theta_2_array)
tt3 = 1 - tt1 - tt2
mask = ((tt3 > 1) | (tt3 < 0))
tt1[mask] = None
tt2[mask] = None
tt3[mask] = None

### 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):
    PDF = dirichlet.pdf([tt1.ravel(), tt2.ravel(), tt3.ravel()], alpha_array)
    fig = plt.figure(figsize=(5, 5), )
    # 四维到三维才需要这样的技巧画等高线，见Book2_chap29
    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape), levels = 20, cmap='RdYlBu_r')
    plt.clf()
    norm = Normalize(vmin = all_contours.levels.min(), vmax = all_contours.levels.max(), clip = True)
    mapper = cm.ScalarMappable(norm = norm, cmap = cm.RdYlBu_r)
    ax = plt.axes(projection="3d")
    # or used follow colors
    colors = cm.RdYlBu_r(np.linspace(0, 1, len(all_contours.levels)))
    # colors = cm.hsv(np.linspace(0, 1, len(all_contours.levels)))
    for k, (level_idx, ctr_idx) in enumerate(zip(all_contours.levels, all_contours.allsegs)):
        for i in range(0, len(ctr_idx)):
            t1_i, t2_i = ctr_idx[i][:,0], ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i
            # 绘制映射结果
            # ax.plot(t1_i, t2_i, t3_i, color = mapper.to_rgba(level_idx), linewidth = 1)
            ax.plot(t1_i, t2_i, t3_i, color = colors[k,:], linewidth = 1)
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
    plt.title(title)
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


plt.close('all')

#%% # 斜面等高线 + x1x2等高线  Bk_2_Ch32_03
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
            t1_i, t2_i = ctr_idx[i][:,0], ctr_idx[i][:,1]
            t3_i = 1 - t1_i - t2_i
            # 绘制映射结果
            ax.plot(t1_i, t2_i, t3_i, color = mapper.to_rgba(level_idx), linewidth = 1, zorder = 1000)

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot((1,0), (0,1), (0,0), lw = 1, c = 'k')
    ax.plot((0,0), (1,0), (0,1), lw = 1, c = 'k')
    ax.plot((1,0), (0,0), (0,1), lw = 1, c = 'k')

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(r'$\alpha_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    plt.title(title)
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

#%% # 斜面等高线 + x1x3等高线 Bk_2_Ch32_04
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
    all_contours = plt.contour(tt1, tt2, PDF.reshape(tt1.shape), cmap='RdYlBu_r')
    plt.clf()
    norm = Normalize(vmin=all_contours.levels.min(), vmax=all_contours.levels.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)
    ax = plt.axes(projection="3d")
    ax.contour(tt1, PDF.reshape(tt1.shape), tt3, zdir = 'y', offset = 0, levels = 20, zorder = 1, cmap='RdYlBu_r')
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
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(r'$\alpha_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    plt.title(title)
    # fig.savefig('Figures/斜面等高线 + 13水平等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)

alpha_array = [2,4,8]
visualize_Dirichlet(alpha_array)

alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)

alpha_array = [8,8,8]
visualize_Dirichlet(alpha_array)

plt.close('all')

#%% # 斜面等高线 + x2x3等高线 Bk_2_Ch32_05
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
    ax.contour(PDF.reshape(tt1.shape), tt2, tt3, zdir = 'x', offset = 0, levels = 20, zorder = 1, cmap='RdYlBu_r')
    # ax.contour(tt1, tt2, tt3, zdir = 'x', offset = 0, levels = 20, zorder = 1, cmap='RdYlBu_r')
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
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel(r'$\alpha_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    # fig.savefig('Figures/斜面等高线 + 23水平等高线_' + title + '.svg', format='svg')

alpha_array = [1, 2, 4]
visualize_Dirichlet(alpha_array)

alpha_array = [2,2,2]
visualize_Dirichlet(alpha_array)

plt.close('all')

#%%# 重心坐标系 Bk_2_Ch32_06, 参考 Bk_2_Ch32_06, BK_2_Ch10_07, Bk_2_Ch18_03
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

## 定义三角形
# 等边三角形
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]]).T
# 等腰直角三角形
# corners = np.array([[0, 0], [1, 0], [0, 1]]).T
# 任意三角形
# corners = np.array([[0, 0.2], [0.9, 0], [0.3, 0.8]]).T

## 三角网格
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)
trimesh_2 = refiner.refine_triangulation(subdiv = 2)
# 每个列向量代表一个三角网格坐标点
r_array = np.row_stack((trimesh_2.x, trimesh_2.y)) # (2, 15)

### 直角坐标系坐标
fig, ax = plt.subplots(figsize = (5,5))
plt.triplot(trimesh_2) # 画网格
plt.plot(r_array[0,:], r_array[1,:], '.r', markersize = 10) # 画坐标点

for x_idx, y_idx in zip(trimesh_2.x, trimesh_2.y):
    text_idx = ('(' + format(x_idx, '.2f') + ', ' + format(y_idx, '.2f') + ')')
    plt.text(x_idx, y_idx+0.03, text_idx, fontsize = 8, horizontalalignment = 'center', bbox=dict(facecolor='w', alpha=0.5, edgecolor = 'None'))
ax.set_aspect('equal')
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.show()
plt.close('all')


### 坐标转换
# 提取大三角形的三个顶点列向量 (坐标点)
r1 = corners[:,[0]]
r2 = corners[:,[1]]
r3 = corners[:,[2]]
# 构造矩阵T
T = np.column_stack((r1 - r3, r2 - r3))
# 计算 theta_1和theta_2
theta_1_2 = np.linalg.inv(T) @ (r_array - r3) # (2,15)
# 计算theta_3
theta_3 = 1 - theta_1_2[0,:] - theta_1_2[1,:]

# 创建theta坐标，每一列代表一个重心坐标系坐标
# r_array和theta_1_2_3每一列一一对应
theta_1_2_3 = np.row_stack((theta_1_2, theta_3)) # (3, 15)

# 对重心坐标进行截断，避免超出 [0,1] 区间
theta_1_2_3 = np.clip(theta_1_2_3, 1e-6, 1.0 - 1e-6)
# numpy.clip(a, a_min, a_max, out=None) 其中a是一个数组，后面两个参数分别表示最小和最大值， 将数组中的元素限制在a_min, a_max之间， 大于a_max的就使得它等于 a_max， 小于a_min, 的就使得它等于a_min

### 重心坐标系坐标
fig, ax = plt.subplots(figsize = (10,10))
plt.triplot(trimesh_2)
plt.plot(r_array[0,:], r_array[1,:], '.r', markersize = 10)
for theta_idx, x_idx, y_idx in zip(theta_1_2_3.T, trimesh_2.x, trimesh_2.y):
    theta_1 = theta_idx[0]
    theta_2 = theta_idx[1]
    theta_3 = theta_idx[2]
    text_idx = ('(' + format(theta_1, '.2f') + ', ' + format(theta_2, '.2f') + ', ' + format(theta_3, '.2f') + ')')
    plt.text(x_idx, y_idx+0.03, text_idx, fontsize = 10, horizontalalignment = 'center', bbox=dict(facecolor='w', alpha=0.5, edgecolor = 'None'))
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

plt.show()
plt.close('all')
#%%# # 重心坐标系中混合红绿蓝 Bk_2_Ch32_07
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# 等边三角形
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]]).T
triangle = tri.Triangulation(corners[0,:], corners[1,:])
refiner = tri.UniformTriRefiner(triangle)
fig, axs = plt.subplots(3,2,figsize=(6,9))
for subdiv,ax in zip(range(1, 7), axs.ravel()):
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
plt.show()
plt.close('all')



#%%# 可视化Dirichlet分布 Bk_2_Ch32_08
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
    r_array = np.row_stack((trimesh_8.x, trimesh_8.y))

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
    plt.tricontourf(trimesh_8, PDF, levels = 20, cmap='RdYlBu_r')
    plt.axis('equal')
    plt.xlim(0, 1); plt.ylim(0, 0.75**0.5)
    plt.text(0.8, 0.45,  r'$\theta_1$')
    plt.text(0.15, 0.45, r'$\theta_2$')
    plt.text(0.5, -0.1,  r'$\theta_3$')
    plt.axis('off'); plt.title(alpha_array)
    plt.show()

alpha_array = [1, 1, 2]
plot_Dirichlet_PDF_contour(alpha_array)

alpha_array = [2, 2, 2]
plot_Dirichlet_PDF_contour(alpha_array)

plt.close('all')


#%%>>>>>>>>>>>>>>>>>>>>> Dirichlet distribution
import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs

fig = plt.figure(figsize=(10.8, 8.8))
fig.subplots_adjust(
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.5,
    hspace=0.5,
)

alphas = ((1.5, 1.5, 1.5), (5.0, 5.0, 5.0), (1.0, 2.0, 2.0), (2.0, 4.0, 8.0))
for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(2, 2, i + 1, projection="ternary")
    t, l, r, v = get_dirichlet_pdfs(n = 61, alpha = alpha)
    cmap = "Blues"
    shading = "gouraud"
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
    ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)

    ax.set_tlabel("$x_1$")
    ax.set_llabel("$x_2$")
    ax.set_rlabel("$x_3$")

    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")

    ax.set_title("${\\mathbf{\\alpha}}$ = " + str(alpha))

    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("PDF", rotation=270, va="baseline")

plt.show()
plt.close('all')


"""
https://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247486666&idx=1&sn=bbf36c621d333e2263ab19a238f5431d&chksm=c22f25808492ba82d302373b9acf2e79c2e322b32f2c4a3527d90380111b47172c34d20ac300&mpshare=1&scene=1&srcid=0916C538zqDddemYDVhOPbTq&sharer_shareinfo=b568983f12459b351182d0f92b4609bf&sharer_shareinfo_first=b568983f12459b351182d0f92b4609bf&exportkey=n_ChQIAhIQos7fEUw5lco2fWQBk5xCfxKfAgIE97dBBAEAAAAAAMU5FD8B5ooAAAAOpnltbLcz9gKNyK89dVj0Qaiu0T5AJTSKAPd10lePgIrTWh0uF6a%2B7%2Fd9qBMdUWCFj8flJ4PpBU2rqDON1RHXOD66BfqZbgD29PXXZa1mrBK1Uw2CVkR3i7uvBrHkT%2F%2B38SIgxrprQ1kaB8voHsjje6l%2B0RMEdQNA0NKRV%2FCUivMVNkodfXTVuQHpuuy7urxRZKHh1VMvRrmUVcmy2d0JcZwZdRQ1tgfJlgU4qOwP2724vzE85s%2FO3x705TBoi2qZ0oChJLbCYbr3sEjiCVnQbLKZXoRkue%2BPtS%2F%2F8wkxCUKFZ%2Fm2eKR%2FSsr%2FvdUlznegKRA%2BZay53cS9AOQs6uXNN54DHL3scszO&acctmode=0&pass_ticket=BWewIwspFdxzXTY4iL9mQ7lp8S6Uttm7xTk6ZcH2UAV9FENBDldeKgnK4AvPDki5&wx_header=0#rd

https://mpltern.readthedocs.io/en/latest/gallery/index.html

"""



import matplotlib.pyplot as plt
from mpltern.datasets import get_dirichlet_pdfs
import pandas as pd
# 定义 alpha 参数
alpha = (2.0, 4.0, 8.0)
# 获取 Dirichlet 分布的概率密度函数 (PDF) 数据
t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)
df = pd.DataFrame({
    'x1': t,
    'x2': l,
    'x3': r,
    'pdf': v
})
# df


# 绘制三元图
fig = plt.figure(figsize=(8, 6), )

ax = fig.add_subplot(1, 1, 1, projection="ternary")
# 设置颜色映射和阴影
cmap = "Blues"
shading = "gouraud"
# 绘制三元图的颜色图
cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)
# 添加等高线
ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)
ax.set_tlabel("$x_1$")
ax.set_llabel("$x_2$")
ax.set_rlabel("$x_3$")
ax.set_title("${\\mathbf{\\alpha}}$=" + str(alpha))
cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)
colorbar.set_label("PDF", rotation=270, va="baseline")
# plt.savefig("1.pdf", format='pdf', bbox_inches='tight')
plt.show()


# 不同条件下的三元图对比
fig = plt.figure(figsize=(10.8, 8.8), )
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
# 定义 alpha 参数
alphas = ((1.5, 1.5, 1.5), (5.0, 5.0, 5.0), (1.0, 2.0, 2.0), (2.0, 4.0, 8.0))

# 循环绘制四个不同的三元图
for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(2, 2, i + 1, projection="ternary")

    # 获取 Dirichlet 分布的概率密度函数 (PDF) 数据
    t, l, r, v = get_dirichlet_pdfs(n=61, alpha=alpha)

    # 设置颜色映射和阴影
    cmap = "Blues"
    shading = "gouraud"

    # 绘制三元图的颜色图
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, shading=shading, rasterized=True)

    # 添加等高线
    ax.tricontour(t, l, r, v, colors="k", linewidths=0.5)

    # 设置标签
    ax.set_tlabel("$x_1$")
    ax.set_llabel("$x_2$")
    ax.set_rlabel("$x_3$")

    # 设置标题
    ax.set_title("${\\mathbf{\\alpha}}$=" + str(alpha))

    # 插入颜色条
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("PDF", rotation=270, va="baseline")
# plt.savefig("2.pdf", format='pdf', bbox_inches='tight')
plt.show()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk5_Ch07_11.py Dirichlet 分布：多元 Beta 分布
import numpy as np
import scipy.stats as st
import scipy.interpolate as si
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# alpha = np.array([1, 1, 1])
# alpha = np.array([2, 2, 2])
# alpha = np.array([4, 4, 4])

# alpha = np.array([1, 4, 4])
# alpha = np.array([4, 1, 4])
# alpha = np.array([4, 4, 1])

# alpha = np.array([4, 2, 2])
# alpha = np.array([2, 4, 2])
# alpha = np.array([2, 2, 4])

# alpha = np.array([1, 2, 4])
# alpha = np.array([2, 1, 4])
alpha = np.array([4, 2, 1])
rv = st.dirichlet(alpha)
x1 = np.linspace(0, 1, 201)
x2 = np.linspace(0, 1, 201)
xx1, xx2 = np.meshgrid(x1, x2)

xx3 = 1.0 - xx1 - xx2
xx3 = np.where(xx3 > 0.0, xx3, np.nan)

PDF_ff = rv.pdf(np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])))
PDF_ff = np.reshape(PDF_ff, xx1.shape)
# PDF_ff = np.nan_to_num(PDF_ff)

#>>>>>>>>>> 2D contour
fig, ax = plt.subplots(figsize=(6, 6))
ax.contourf(xx1, xx2, PDF_ff, 20, cmap='RdYlBu_r')
plt.show()
plt.close()

# 3D contour
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx1, xx2, PDF_ff, color = [0.7,0.7,0.7], linewidth = 0.25, rstride=10, cstride=10)
ax.contour(xx1, xx2, PDF_ff, levels = 20,  cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,20])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,20])
ax.view_init(azim=-120, elev=30)
plt.tight_layout()
ax.grid(True)
plt.show()
plt.close()

#>>>>>>>>>>  3D visualization
x1_ = np.linspace(0, 1, 51)
x2_ = np.linspace(0, 1, 51)
xx1_, xx2_ = np.meshgrid(x1_, x2_)
xx3_ = 1.0 - xx1_ - xx2_
xx3_ = np.where(xx3_ > 0.0, xx3_, np.nan)
PDF_ff_ = rv.pdf(np.array(([xx1_.ravel(), xx2_.ravel(), xx3_.ravel()])))
PDF_ff_ = np.reshape(PDF_ff_, xx1_.shape)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

# Creating plot
PDF_ff_ = np.nan_to_num(PDF_ff_)
ax.scatter3D(xx1_.ravel(), xx2_.ravel(), xx3_.ravel(), c = PDF_ff_.ravel(), marker='.', cmap = 'RdYlBu_r')
ax.contour(xx1_, xx2_, PDF_ff_, 15, zdir='z', offset=0, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x, y, z, u, v, w, arrow_length_ratio = 0.1, color = "black")
# ax.set_axis_off()

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])

ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=-30, elev=20)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()
plt.close('all')
#>>>>>>>>>>  Marginal distributions
from scipy.stats import beta

alpha = np.array([1, 2, 4])
x_array = np.linspace(0,1,200)
alpha_array = alpha
beta_array = alpha.sum() - alpha

# PDF of Beta Distributions
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
for alpha_idx, beta_idx, ax in zip(alpha_array.ravel(), beta_array.ravel(), axs.ravel()):
    title_idx = '\u03B1 = ' + str(alpha_idx) + '; \u03B2 = ' + str(beta_idx)
    ax.plot(x_array, beta.pdf(x_array, alpha_idx, beta_idx), lw=1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,4)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,2,4])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')
    ax.set_box_aspect(1)
    ax.set_title(title_idx)

#>>>>>>>>>> Scatter plot of random data
random_data = np.random.dirichlet(alpha, 500).T
pdf = st.dirichlet(alpha).pdf(random_data)
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.scatter3D(random_data[0,:], random_data[1,:], random_data[2,:], c = pdf, marker='.')
ax.set_proj_type('ortho')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xticks(np.linspace(0,1,6))
ax.set_yticks(np.linspace(0,1,6))
ax.set_zticks(np.linspace(0,1,6))

x, y, z = np.array([[0,0,0],[0,0,0],[0,0,0]])
u, v, w = np.array([[1.2,0,0],[0,1.2,0],[0,0,1.2]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_zticks([0,1])
ax.plot([0,1],[1,0],[0,0],c='k',ls = '--')
ax.plot([0,1],[0,0],[1,0],c='k',ls = '--')
ax.plot([0,0],[0,1],[1,0],c='k',ls = '--')
ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_zlim3d([0,1])
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=30, elev=20)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))
ax.grid()
plt.show()
plt.close('all')


#%% Book2_Chap13  Dirichlet分布概率密度
import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):
    x1_ = np.linspace(0,1,num + 1)
    x2_ = np.linspace(0,1,num + 1)

    xx1_, xx2_ = np.meshgrid(x1_, x2_)

    xx3_ = 1.0 - xx1_ - xx2_
    xx3_ = np.where(xx3_ > 0.0005, xx3_, np.nan)

    rv = st.dirichlet(alpha_array)

    PDF_ff_ = rv.pdf(np.array(([xx1_.ravel(), xx2_.ravel(), xx3_.ravel()])))
    PDF_ff_ = np.reshape(PDF_ff_, xx1_.shape)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")

    # Creating plot
    PDF_ff_ = np.nan_to_num(PDF_ff_)
    ax.scatter3D(xx1_.ravel(), xx2_.ravel(), xx3_.ravel(), c=PDF_ff_.ravel(), alpha = 1, marker='.', cmap = 'RdYlBu_r')
    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.set_xticks(np.linspace(0,1,6))
    # ax.set_yticks(np.linspace(0,1,6))
    # ax.set_zticks(np.linspace(0,1,6))
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(x1_.min(), x1_.max())
    ax.set_ylim(x2_.min(), x2_.max())
    ax.set_zlim3d([0,1])
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))
    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    # fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()

alpha_array = [1, 2, 2]
visualize_Dirichlet(alpha_array)

alpha_array = [2, 1, 2]
visualize_Dirichlet(alpha_array)

alpha_array = [2, 2, 1]
visualize_Dirichlet(alpha_array)


alpha_array = [4, 2, 1]
visualize_Dirichlet(alpha_array)

alpha_array = [1, 1, 1]
visualize_Dirichlet(alpha_array)

alpha_array = [2, 2, 2]
visualize_Dirichlet(alpha_array)

alpha_array = [4, 4, 4]
visualize_Dirichlet(alpha_array)

# plt.show()
plt.close('all')
#%% Dirichlet分布随机数
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet

def visualize_sample(alpha_array):
    samples = np.random.dirichlet(alpha_array, size=500)
    # 计算Dirichlet概率密度值
    pdf_values = dirichlet.pdf(samples.T, alpha_array)
    # 创建三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制散点图，并用颜色映射表示概率密度值
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s = 3.8, c=pdf_values, cmap='RdYlBu_r')
    ax.plot([0,1],[1,0],[0,0],c='k',ls = '--')
    ax.plot([0,1],[0,0],[1,0],c='k',ls = '--')
    ax.plot([0,0],[0,1],[1,0],c='k',ls = '--')

    ax.set_proj_type('ortho')
    ax.view_init(azim = 30, elev = 30)
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticklabels([])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.grid(c = '0.88')

    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    ax.set_title(title)
    plt.show()

alpha_array = [1, 2, 2]
visualize_sample(alpha_array)

alpha_array = [2, 1, 2]
visualize_sample(alpha_array)

alpha_array = [2, 2, 1]
visualize_sample(alpha_array)


alpha_array = [4, 4, 4]
visualize_sample(alpha_array)


alpha_array = [8, 8, 8]
visualize_sample(alpha_array)

# plt.show()
plt.close('all')

#%% Book2_Chap13 用三维散点可视化多项分布
# 导入包
from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_multinomial(p_array, num = 20):
    x1_array = np.arange(num + 1)
    x2_array = np.arange(num + 1)

    xx1, xx2 = np.meshgrid(x1_array, x2_array)

    xx3 = num - xx1 - xx2
    xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

    PMF_ff = multinomial.pmf(x = np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, n = num, p = p_array)
    PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)
    PMF_ff = np.reshape(PMF_ff, xx1.shape)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection = "3d")
    scatter_plot = ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(), s = 50, marker = '.', alpha = 1, c = PMF_ff.ravel(), cmap = 'RdYlBu_r')

    ax.set_proj_type('ortho')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xticks([0,num])
    ax.set_yticks([0,num])
    ax.set_zticks([0,num])

    ax.set_xlim(0, num)
    ax.set_ylim(0, num)
    ax.set_zlim3d(0, num)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')
    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    # fig.colorbar(scatter_plot, ax = ax, shrink = 0.5, aspect = 10)
    title = '_'.join(str(round(p_i,2)) for p_i in p_array)
    title = 'p_array_' + title
    ax.set_title(title)
    plt.show()

p_array = [1/3, 1/3, 1/3]
visualize_multinomial(p_array)

p_array = [0.2, 0.2, 0.6]
visualize_multinomial(p_array)

p_array = [0.2, 0.6, 0.2]
visualize_multinomial(p_array)
plt.close('all')

#%% Book5_Ch05_04.py
from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

num = 8
x1_array = np.arange(num + 1)
x2_array = np.arange(num + 1)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

xx3 = num - xx1 - xx2
xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

def heatmap_sum(data, i_array, j_array, title, vmin, vmax, cmap, annot = False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(data,cmap= cmap, #'YlGnBu', # YlGnBu
                     cbar_kws={"orientation": "horizontal"},
                     yticklabels=i_array, xticklabels=j_array,
                     ax = ax, annot = annot,
                     linewidths=0.25, linecolor='grey',
                     vmin = vmin, vmax = vmax,
                     fmt = '.3f')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.title(title)
    plt.yticks(rotation=0)

### calculate multinomial probability
p_array = [0.6, 0.1, 0.3]
p_array = [0.3, 0.4, 0.3]
p_array = [0.1, 0.6, 0.3]

PMF_ff = multinomial.pmf(x=np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, n=num, p=p_array)
PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)
PMF_ff = np.reshape(PMF_ff, xx1.shape)

### save to excel file
df = pd.DataFrame(np.flipud(PMF_ff))
filepath = 'PMF_ff.xlsx'
df.to_excel(filepath, index=False)

### 3D/2D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(), s = 400, marker='.', c=PMF_ff.ravel(), cmap = 'RdYlBu_r')
# ax.contour(xx1, xx2, PMF_ff, 15, zdir='z', offset=0, cmap='RdYlBu_r')
ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xticks([0,num])
ax.set_yticks([0,num])
ax.set_zticks([0,num])

ax.set_xlim(0, num)
ax.set_ylim(0, num)
ax.set_zlim3d(0, num)
# ax.view_init(azim=20, elev=20)
ax.view_init(azim=-30, elev=20)
ax.view_init(azim=-90, elev=90)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
# ax.set_aspect('equal')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
plt.show()

### heatmap
title = 'PMF of binomial distribution'
heatmap_sum(PMF_ff,x1_array,x2_array,title,0,0.12,'plasma_r',True)

### 3D stem chart

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')

ax.stem(xx1.ravel(), xx2.ravel(), PMF_ff.ravel(), basefmt=" ")

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('PMF')

ax.set_xlim((0,8))
ax.set_ylim((0,8))
ax.set_zlim((0,0.12))
# ax.set_zticks([])
# ax.grid(False)
ax.view_init(azim=-100, elev=20)
ax.set_proj_type('ortho')
plt.show()

# test only

# print(multinomial.pmf(x=(5,2,1), n=num, p=p_array))


