#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:53:02 2024

@author: jack
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486458&idx=1&sn=1da0e7d98fa1b4280e4822b44d9d5ed4&chksm=c14f7dd35ea75a0f8ce7f3ece7ae5c6a4cdfc6523eb344423efc57b16bfae282f57e2a0abfb5&mpshare=1&scene=1&srcid=0909XVg1J0N7YWafLLdxil3O&sharer_shareinfo=7a5ad6305011a0c9a5a3c30bc5624696&sharer_shareinfo_first=7a5ad6305011a0c9a5a3c30bc5624696&exportkey=n_ChQIAhIQgxYWk1CzRfgSfYREFwbOYRKfAgIE97dBBAEAAAAAAMZ7MCsLoJsAAAAOpnltbLcz9gKNyK89dVj0clrovH3uvscfvlwaYaVf9d6gtmphSRM35kvmRMSN6ThRV3PJlzyQvzpCWF0C2G%2F5YWsjy4SqHn7wCq70NpzqnLM%2B%2BsCp2F5L5MlxElbkPgUZXwGKHxj7ELfhnAAdE1oLiRTaFR6feUAT5v8vUadiKMkUUrz4wq21eZ8oEUyadOjopqpJRMCBFNBNthRqXE6yQSHp3jxDcItI94I1RHIQGW1u7JAnOlv1hlLK8Bs8MK0bDBwAXGrkTJjc16kmQ9iuw9%2BtSR%2F19O4N%2BCRwvsBziZ%2BWc687ceaOc86uLujsMBqLNEuspfsisFFJzbvHJ%2F5o62HhoCT%2F%2FTjh&acctmode=0&pass_ticket=otO0lnxAYPdRPAR9Via1yynH6MtwIw9xfuGImUeSs%2BHVJLBsJCuHKRBmHcbXLf5b&wx_header=0#rd
"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. plot()：线形图（Line Plot）
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制两条线形图，带有不同的样式和标记
plt.plot(x, y1, color='blue', linestyle='-', linewidth=2, marker='o', label='sin(x)')
plt.plot(x, y2, color='red', linestyle='--', linewidth=2, marker='x', label='cos(x)')

# 添加标题、标签和图例
plt.title("Detailed Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()

# 显示图形
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. scatter()：散点图（Scatter Plot）
import matplotlib.pyplot as plt

# 创建数据
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]
colors = [5, 20, 15, 25, 10, 30, 5, 15, 20, 25]
sizes = [100, 200, 150, 300, 100, 250, 100, 200, 300, 150]

# 绘制散点图
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')

# 添加标题和标签
plt.title("Detailed Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 添加颜色条
plt.colorbar()

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 3. bar()：条形图（Bar Plot）

import matplotlib.pyplot as plt

# 创建数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [5, 7, 8, 4, 6]
colors = ['blue', 'green', 'red', 'purple', 'orange']

# 绘制条形图
plt.bar(categories, values, color=colors, edgecolor='black')

# 添加标题和标签
plt.title("Detailed Bar Plot Example")
plt.xlabel("Categories")
plt.ylabel("Values")

# 添加数值标签
for i, value in enumerate(values):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 4. hist()：直方图（Histogram）
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = np.random.randn(1000)

# 绘制直方图
plt.hist(data, bins=30, color='purple', edgecolor='black', alpha=0.7, density=True)

# 添加标题和标签
plt.title("Detailed Histogram Example")
plt.xlabel("Value")
plt.ylabel("Density")

# 添加一个拟合曲线（正态分布）
mean = np.mean(data)
std_dev = np.std(data)
x = np.linspace(min(data), max(data), 100)
pdf = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
plt.plot(x, pdf, color='red', linewidth=2)

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 5. boxplot()：箱线图（Box Plot）
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
np.random.seed(10)
data1 = np.random.normal(100, 10, 200)
data2 = np.random.normal(90, 20, 200)
data3 = np.random.normal(80, 30, 200)

# 绘制箱线图
plt.boxplot([data1, data2, data3], patch_artist=True, notch=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(markerfacecolor='green', marker='o', markersize=12, linestyle='none'))

# 添加标题和标签
plt.title("Detailed Box Plot Example")
plt.xlabel("Groups")
plt.ylabel("Values")

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 6. pie()：饼图（Pie Chart）
import matplotlib.pyplot as plt

# 创建数据
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.1, 0, 0)  # 突出第二个部分

# 绘制饼图
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# 确保饼图是圆形
plt.axis('equal')

# 添加标题
plt.title("Detailed Pie Chart Example")

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 7. heatmap()：热图（Heatmap）
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 创建数据
data = np.random.rand(10, 12)

# 绘制热图
sns.heatmap(data, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5)

# 添加标题
plt.title("Detailed Heatmap Example")

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 8. contour()：等高线图（Contour Plot）
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制等高线图
contour = plt.contour(X, Y, Z, levels=15, cmap='coolwarm')

# 添加等高线标签
plt.clabel(contour, inline=True, fontsize=8)

# 添加标题和标签
plt.title("Detailed Contour Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 9. stackplot()：堆积图（Stack Plot）
import matplotlib.pyplot as plt

# 创建数据
days = [1, 2, 3, 4, 5]
apples = [3, 4, 3, 2, 4]
bananas = [1, 2, 1, 3, 4]
oranges = [2, 3, 2, 4, 5]

# 绘制堆积图
plt.stackplot(days, apples, bananas, oranges, labels=['Apples', 'Bananas', 'Oranges'],
              colors=['red', 'yellow', 'orange'])

# 添加标题、标签和图例
plt.title("Detailed Stack Plot Example")
plt.xlabel("Days")
plt.ylabel("Fruit Counts")
plt.legend(loc='upper left')

# 显示图形
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 10. fill_between()：填充图（Filled Plot）
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.2

# 绘制填充图
plt.fill_between(x, y1, y2, color='green', alpha=0.3)

# 添加标题和标签
plt.title("Detailed Filled Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图形
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>

import matplotlib.pyplot as plt
import numpy as np

# 创建虚拟数据
np.random.seed(42)
x = np.linspace(0, 10, 200)
y1 = np.sin(x) + np.random.normal(0, 0.1, 200)
y2 = np.cos(x) + np.random.normal(0, 0.1, 200)
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 50, size=5)

# 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：折线图
axs[0, 0].plot(x, y1, label='Sin(x)', color='blue', linewidth=2)
axs[0, 0].plot(x, y2, label='Cos(x)', color='red', linestyle='--', linewidth=2)
axs[0, 0].set_title('Line Plot')
axs[0, 0].set_xlabel('X-axis')
axs[0, 0].set_ylabel('Y-axis')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 子图2：散点图
scatter = axs[0, 1].scatter(x, y1, c=y2, cmap='coolwarm', s=50, edgecolor='black')
axs[0, 1].set_title('Scatter Plot')
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[0, 1], label='Cos(x) Value')

# 子图3：条形图
axs[1, 0].bar(categories, values, color='purple')
axs[1, 0].set_title('Bar Chart')
axs[1, 0].set_xlabel('Categories')
axs[1, 0].set_ylabel('Values')

# 子图4：折线图与条形图的组合
axs[1, 1].plot(x, y1, label='Sin(x)', color='green', linewidth=2)
axs[1, 1].bar(categories, values, color='orange', alpha=0.6)
axs[1, 1].set_title('Combined Line and Bar Plot')
axs[1, 1].set_xlabel('Mixed X-axis')
axs[1, 1].set_ylabel('Mixed Y-axis')
axs[1, 1].legend()

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>






#%%>>>>>>>>>>>>>>>>>>>>>>>






#%%>>>>>>>>>>>>>>>>>>>>>>>







