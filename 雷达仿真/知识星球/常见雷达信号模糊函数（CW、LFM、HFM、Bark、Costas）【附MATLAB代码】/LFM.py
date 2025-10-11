#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:06:30 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzU0MTAzMjkyMg==&mid=2247498922&idx=1&sn=befd04f173b8f1dba873b3830f5b62fe&chksm=fa76da753ea943b83e968746889635e7798e3c7f9d54513c215fcd2d9620dadc683b64d44aa9&mpshare=1&scene=1&srcid=1008Rv8zeh9aX7vAtOxa2ZSm&sharer_shareinfo=1dba6367293ca5fa42defd77972d8fd8&sharer_shareinfo_first=1dba6367293ca5fa42defd77972d8fd8&exportkey=n_ChQIAhIQpPxCqwhQBFi%2BxXnq0mHelBKfAgIE97dBBAEAAAAAAGZOMNXew0IAAAAOpnltbLcz9gKNyK89dVj0jvHbNpSNV4CGuFzXTCm7iJHuovGFIs%2Fjz6bxiDbRI4j5sjcedF6TqUfaQlqMGVqPZsYlQwHPLR7FJlmMCTD%2BSzxhc8zB12YXTYIKfbajhg6S3wXK8ums5zdyf3V4yhsCEoQs2el5wIhkZax4hFZHMa9vHAxnebeQKr2XrWbf4JW77jYJeBNhtlsOXhVWUATwrN%2FJKYDSeWWXs1Mf9MBU0E3Ukya6FdvU1beOAF%2Fk%2FTOlwW%2FFknyal3exzCVSSGixT4hEoDTXLNhyFwYrxyDJWICGWhluXgQaR%2FAcR7mXyu6%2BuYxIyGEege3GLaPBitKNUxwbCardtZZk&acctmode=0&pass_ticket=R%2F%2BSG0Djwa9mZyum6k8cs0WfQP8hh0Y28gDwHfCKGvHrUdWguVAxG84H9J7k%2B6d5&wx_header=0#rd


"""

#%% 绘制三维图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建 x 和 y 的值
tdivtau = np.linspace(-1, 1, 1000)
Ftau = np.linspace(-10, 10, 1000)
t, F  = np.meshgrid(tdivtau, Ftau)

# 计算 z 的值
Z = np.abs(np.sinc(F*(1-np.abs(t)))*(1-np.abs(t)))

# 创建三维图形
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面
surf=ax.plot_surface(t, F, Z, cmap='viridis',edgecolor='none',antialiased=True)

# 翻转 x 轴
ax.invert_xaxis()

# 设置标签
ax.set_xlabel('t/tau')
ax.set_ylabel('FD*tau')
ax.set_zlabel('A(t,FD)')
# 添加颜色条以显示 Z 值的映射关系
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# 调整视角
ax.view_init(elev=20, azim=140)  # 改变 elev 和 azim 参数以调整视角

# 显示图形
plt.show()
plt.close()

#%% 绘制等高线
import numpy as np
import matplotlib.pyplot as plt

# 创建 x 和 y 的值
tdivtau = np.linspace(-1, 1, 1000)
Ftau = np.linspace(-10, 10, 1000)
t, F = np.meshgrid(tdivtau, Ftau)

# 计算原始函数的值
A_original = np.abs(np.sinc(F*(1-np.abs(t)))*(1-np.abs(t)))

# 将函数转换为 dB 单位（20*log10()）
A_dB = 20 * np.log10(A_original)

# 绘制等高线图
plt.figure(figsize=(10, 8))

# 绘制指定 dB 值的等高线
contour_plot = plt.contour(F, t, A_dB, levels=[-20, -10, -3, -0.5], cmap='viridis')

# 添加颜色条
plt.colorbar(contour_plot, label='A(t,FD) (dB)')

# 设置标签和标题
plt.xlabel('FD*tau')
plt.ylabel('t/tau')
plt.xticks(np.arange(-10, 11, step=2))
plt.yticks(np.arange(-1, 1.1, step=0.2))
plt.title('Contour Plot of A(t,FD) in dB')

# 显示图形
plt.show()
plt.close()

#%% 绘制Zero_delay截面图
import numpy as np
import matplotlib.pyplot as plt

# 创建 F 的值
Ftau = np.linspace(-10, 10, 1000)

# 设置 t 为常数
t = 0

# 计算 Z 的值
Z = np.abs(np.sinc(Ftau*(1-np.abs(t)))*(1-np.abs(t)))

# 绘制二维函数图
plt.figure(figsize=(8, 6))
plt.plot(Ftau, Z, label=f't/tau = {t}')
plt.xlabel('FD*tau')
plt.ylabel('A(t,FD)')
plt.title('Plot of A(t,FD) for t/tau = 0')

plt.xticks(np.arange(-10,11,step=2))
plt.legend()
plt.grid(True)
plt.show()
plt.close()

#%% 绘制Zero_Doppler截面图
import numpy as np
import matplotlib.pyplot as plt

# 设置 F 为常数
Ftau = 0

# 设置t
t = np.linspace(-1, 1, 1000)

# 计算 Z 的值
Z = np.abs(np.sinc(Ftau*(1-np.abs(t)))*(1-np.abs(t)))

# 绘制二维函数图
plt.figure(figsize=(8, 6))
plt.plot(t, Z, label=f'FD*tau = {Ftau}')
plt.xlabel('t/tau')
plt.ylabel('A(t,FD)')
plt.title('Plot of A(t,FD) for FD*tau = 0')


plt.xticks(np.arange(-1,1.1,step=0.2))
plt.legend()
plt.grid(True)
plt.show()
plt.close()

#%%


#%%


#%%


#%%


#%%


















































































