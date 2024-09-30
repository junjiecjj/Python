#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:59:51 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247487103&idx=1&sn=31d796544ecf1ad5434baa23384d6a5f&chksm=c1859a86aa29bfeb0cbf4c72190b4f689fe118345efd72a0c6ae6b5312902d5980c1d7e1ce9a&mpshare=1&scene=1&srcid=09309WPosSR99rvCrCVmPwHJ&sharer_shareinfo=d1459db293ea600cbfabe123d9d59c7c&sharer_shareinfo_first=d1459db293ea600cbfabe123d9d59c7c&exportkey=n_ChQIAhIQYzb5NFyko1eb%2Ba7WA0QjrRKfAgIE97dBBAEAAAAAAIExLi%2FNBbAAAAAOpnltbLcz9gKNyK89dVj06Y8Z3ZG1AvFFuC1CiomvbxSya%2FOzRc53xK4K9w6DGGMNzJeIj8EOueO9%2B3oFTmCCDUXCBCZ4fkMc3%2FKD77Jz1L6UJLsP7s7URUvxwepvYN69LrpLxgqYCLsZvmwOb76qPIXMhv1VQaHBnSJbG7QpKYFe88Xqrimns1aGO%2ButVQFIDJ0KeZmuJvqcJfI9FcH7nldephI6nGc8HEcl1b%2FPN4VIR7aNrtTyj6Oh5Fso0tRaraeR8skmBYsvun6g%2FlG9OKnSGvwG%2BZkZQ88u6DtJvhtS8A1x3OFDBi13puRjbXkOx8AcXlVDa7CGcMxAo6kqwrtZn3hH5Jmq&acctmode=0&pass_ticket=c%2FRN53yL3GOcBVu%2BJkDgV4cYRKkCYwvMj%2BWk%2FC3%2FoJuo%2BbwCNmaE1wJCQOjt7fCP&wx_header=0#rd


"""



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. 子图布局（Subplots Layout）

import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x / 3)

# 创建 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 第一个子图
axs[0, 0].plot(x, y1, 'r', label='sin(x)')
axs[0, 0].set_title('Sine Wave')
axs[0, 0].legend()

# 第二个子图
axs[0, 1].plot(x, y2, 'g', label='cos(x)')
axs[0, 1].set_title('Cosine Wave')
axs[0, 1].legend()

# 第三个子图
axs[1, 0].plot(x, y3, 'b', label='tan(x)')
axs[1, 0].set_title('Tangent Wave')
axs[1, 0].legend()

# 第四个子图
axs[1, 1].plot(x, y4, 'm', label='exp(x/3)')
axs[1, 1].set_title('Exponential Function')
axs[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2. 共享坐标轴 (Shared Axes)
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建共享X轴的子图
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# 上图
axs[0].plot(x, y1, 'r', label='sin(x)')
axs[0].set_title('Sine Wave')
axs[0].legend()

# 下图
axs[1].plot(x, y2, 'b', label='cos(x)')
axs[1].set_title('Cosine Wave')
axs[1].legend()

# 设置共享的x轴标签
plt.xlabel('X Axis')
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3. 自定义刻度和网格 (Custom Ticks and Grids)
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制基本图形
ax.plot(x, y)

# 自定义 X 轴刻度和网格
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten'], fontsize=12)

# 设置网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.7)

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4. 使用对数坐标 (Logarithmic Scales)
# 生成数据
x = np.linspace(1, 1000, 100)
y = x ** 2

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制对数坐标图
ax.plot(x, y)
ax.set_xscale('log')
ax.set_yscale('log')

# 设置标签
ax.set_xlabel('Log X')
ax.set_ylabel('Log Y')

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5. 添加注释 (Annotations)
# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制基本图形
ax.plot(x, y)

# 添加注释
ax.annotate('Local Max', xy=(1.57, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6. 彩色映射 (Colormaps)


# 生成数据
x = np.random.randn(1000)
y = np.random.randn(1000)
colors = np.sqrt(x**2 + y**2)

# 使用彩色映射绘制散点图
plt.scatter(x, y, c=colors, cmap='viridis', s=50)
plt.colorbar()  # 显示色条
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7. 三维绘图 (3D Plotting)

from mpl_toolkits.mplot3d import Axes3D

# 生成数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)


X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D图形
ax.plot_surface(X, Y, Z, cmap='inferno')
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8. 双坐标轴 (Twin Axes)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.log(x + 1)

fig, ax1 = plt.subplots()

# 第一个坐标轴
ax1.plot(x, y1, 'g-')
ax1.set_ylabel('sin(x)', color='g')

# 第二个坐标轴
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-')
ax2.set_ylabel('log(x+1)', color='b')

plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 9. 热力图 (Heatmaps)

data = np.random.rand(10, 10)

plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 10. 自定义图形风格 (Custom Styles)
print(plt.style.available) # 打印样式列表
for style in plt.style.available:
    plt.style.use(style)
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    axs.plot(x, y)
    axs.set_title(f'{style}')
    plt.show()
plt.close("all")
#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 案例描述：


import matplotlib.pyplot as plt
import numpy as np

# 生成虚拟数据集
np.random.seed(42)
n = 1000
A = np.random.normal(loc=50, scale=10, size=n)  # 特征 A，正态分布
B = A + np.random.normal(loc=0, scale=5, size=n)  # 特征 B，与 A 相关联
C = np.random.randint(0, 3, size=n)  # 特征 C，离散类别数据 (0, 1, 2)

# 创建一个图形，包含 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Data Analysis: Feature A, B, and C', fontsize=16)

# 子图 1: A 和 B 的折线图
axs[0, 0].plot(A, label='Feature A', color='red', linewidth=2)
axs[0, 0].plot(B, label='Feature B', color='blue', linestyle='--', linewidth=2)
axs[0, 0].set_title('Line Plot of Feature A and B')
axs[0, 0].set_xlabel('Index')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 子图 2: A 和 B 的散点图，C 作为颜色映射
scatter = axs[0, 1].scatter(A, B, c=C, cmap='plasma', s=100, edgecolor='k', alpha=0.8)
axs[0, 1].set_title('Scatter Plot of Feature A vs Feature B (Colored by Feature C)')
axs[0, 1].set_xlabel('Feature A')
axs[0, 1].set_ylabel('Feature B')
# 添加颜色条
cbar = fig.colorbar(scatter, ax=axs[0, 1])
cbar.set_label('Feature C Categories')

# 子图 3: A 和 B 的直方图
axs[1, 0].hist(A, bins=15, color='green', alpha=0.7, label='Feature A')
axs[1, 0].hist(B, bins=15, color='orange', alpha=0.7, label='Feature B')
axs[1, 0].set_title('Histogram of Feature A and B')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 子图 4: 箱线图，展示不同类别 C 的数据分布 (针对 A 和 B)
axs[1, 1].boxplot([A[C == 0], A[C == 1], A[C == 2]], positions=[1, 2, 3], widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='cyan', color='black'),
                  medianprops=dict(color='black'))
axs[1, 1].boxplot([B[C == 0], B[C == 1], B[C == 2]], positions=[4, 5, 6], widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='pink', color='black'),
                  medianprops=dict(color='black'))
axs[1, 1].set_title('Boxplot of Feature A and B Grouped by Feature C')
axs[1, 1].set_xticks([1.5, 4.5])
axs[1, 1].set_xticklabels(['Feature A', 'Feature B'])
axs[1, 1].set_ylabel('Value')

# 调整子图之间的布局
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 显示图形
plt.show()




