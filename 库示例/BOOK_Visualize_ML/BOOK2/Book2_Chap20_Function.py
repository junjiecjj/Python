

#%% 一元一次函数
import numpy as np
import matplotlib.pyplot as plt


## 创建数据
x1_array = np.linspace(-2,2,21)
f1_array = x1_array # f(x) = x
ticks_array = np.linspace(x1_array.min(),x1_array.max(), num = int(x1_array.max() - x1_array.min()) + 1,  endpoint = True)


## 线图
fig, ax = plt.subplots()

ax.plot(x1_array, f1_array)

ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_xticks(ticks_array); ax.set_yticks(ticks_array);
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(x1_array.min(),x1_array.max());
ax.set_ylim(x1_array.min(),x1_array.max());
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()

#%% 二元一次函数

import numpy as np
import matplotlib.pyplot as plt

## 创建数据
x1_array = np.linspace(-2,2,21)
x2_array = np.linspace(-2,2,21)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

f2_array = xx1 + xx2
ticks_array = np.linspace(x1_array.min(),x1_array.max(),  num = int(x1_array.max() - x1_array.min()) + 1,  endpoint = True)


### 网格面
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array,  rstride=1, cstride=1, linewidth = 1)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array); ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30); ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array,  rstride=1, cstride=0)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array,  rstride=0, cstride=1)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


### 等高线
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array,  rstride=1, cstride=1, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array,  levels = 20, cmap = 'RdYlBu_r')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


### 平面等高线
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
# 平面等高线
ax.contour(xx1, xx2, f2_array, levels = 20, cmap = 'RdYlBu_r')

ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks(ticks_array); ax.set_yticks(ticks_array)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot()
# 平面填充等高线
ax.contourf(xx1, xx2, f2_array, levels = 20, cmap = 'RdYlBu_r')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks(ticks_array); ax.set_yticks(ticks_array)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()



x2_levels = np.linspace(xx2.min(),xx2.max(),21, endpoint = True)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array, rstride=0, cstride=1, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array, levels = x2_levels, zdir='y', cmap = 'RdYlBu_r')

ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array, rstride=0, cstride=1, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array, levels = x2_levels, zdir='y', offset = xx2.max(), cmap = 'RdYlBu_r')
ax.contour(xx1, xx2, f2_array, levels = x2_levels, zdir='y', cmap = 'RdYlBu_r')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()



fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1,1,1])
ax.contour(xx1, xx2, f2_array,  levels = x2_levels, zdir='y', offset = xx2.max(), cmap = 'RdYlBu_r')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-90, elev=0)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()



x1_levels = np.linspace(xx1.min(),xx1.max(),21, endpoint = True)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array, rstride=1, cstride=0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array, levels = x1_levels, zdir='x', cmap = 'RdYlBu_r')
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array, rstride=1, cstride=0, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array, levels = x1_levels, zdir='x', offset = xx1.max(), cmap = 'RdYlBu_r')
ax.contour(xx1, xx2, f2_array, levels = x1_levels, zdir='x', cmap = 'RdYlBu_r')

ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1,1,1])
ax.contour(xx1, xx2, f2_array, levels = x1_levels, zdir='x', offset = xx1.max(), cmap = 'RdYlBu_r')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks(ticks_array);
ax.set_yticks(ticks_array)
ax.view_init(azim=0, elev=0)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()













































































































































































































































