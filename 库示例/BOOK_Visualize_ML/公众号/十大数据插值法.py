#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:10:11 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488088&idx=1&sn=95c453fb33771526bf80adb49115b74a&chksm=c0e5c89ef792418893e3f14de1b2abc7dde28f8c8ea3c5d4e14ab4d177eab7efc0e3434ebc31&cur_album_id=3445855686331105280&scene=190#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. 线性插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 生成带有缺失值的虚拟数据集
np.random.seed(0)
x = np.linspace(0, 10, 20)
y = np.sin(x) + 0.5 * np.random.normal(size=len(x))

# 随机选择一部分数据置为空以模拟缺失
mask = np.random.choice([True, False], size=y.shape, p=[0.3, 0.7])
y_missing = y.copy()
y_missing[mask] = np.nan  # 将随机部分置为NaN模拟缺失值

# 使用线性插值填补缺失值
x_valid = x[~mask]  # 有效的x值（非NaN）
y_valid = y[~mask]  # 有效的y值（非NaN）

# 生成插值函数
linear_interp_func = interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
y_interpolated = linear_interp_func(x)  # 生成插值数据

# 计算插值误差（仅对原始非缺失位置）
error = np.abs(y - y_interpolated)

# 绘制图形
plt.figure(figsize=(12, 8))

# 图1：原始数据和插值数据对比
plt.subplot(3, 1, 1)
plt.plot(x, y, 'o-', color="royalblue", label='Original Data', alpha=0.6)
plt.plot(x, y_missing, 'o', color="tomato", label='Data with Missing Values', alpha=0.5)
plt.plot(x, y_interpolated, 'o--', color="forestgreen", label='Interpolated Data', alpha=0.7)
plt.title('Original Data, Missing Data, and Interpolated Data')
plt.legend()

# 图2：插值误差图
plt.subplot(3, 1, 2)
plt.plot(x, error, 's-', color="purple", markerfacecolor="yellow", label='Interpolation Error')
plt.title('Interpolation Error')
plt.xlabel('X')
plt.ylabel('Error')
plt.legend()

# 图3：插值数据的趋势
plt.subplot(3, 1, 3)
plt.plot(x, y_interpolated, 'o-', color="darkorange", label='Interpolated Data')
plt.title('Trend of Interpolated Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2. 多项式插值
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# 生成虚拟数据集
np.random.seed(42)
x = np.linspace(-5, 5, 10)  # 原始数据的x坐标
y = np.sin(x) + np.random.normal(0, 0.1, len(x))  # 带噪声的y数据

# 生成高精度的插值点，用于绘制插值曲线
x_interp = np.linspace(-5, 5, 500)  # 插值后高分辨率的x坐标

# 多项式插值
degree = 9  # 多项式的阶数
coefficients = np.polyfit(x, y, degree)  # 拟合多项式的系数
poly = np.poly1d(coefficients)  # 创建多项式函数
y_interp = poly(x_interp)  # 插值后的y数据

# 计算误差：真实函数和插值函数之间的差异
true_function = np.sin(x_interp)
error = true_function - y_interp

# 创建图像
plt.figure(figsize=(12, 6))

# 图1：数据点和插值曲线
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='red', label='Data points')  # 原始数据点
plt.plot(x_interp, y_interp, color='blue', label=f'Polynomial (degree {degree})')  # 插值曲线
plt.plot(x_interp, true_function, color='green', linestyle='--', label='True function')  # 真实函数
plt.title('Polynomial Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 图2：插值误差曲线
plt.subplot(1, 2, 2)
plt.plot(x_interp, error, color='purple', label='Error')
plt.axhline(0, color='black', linestyle='--')  # y=0的参考线
plt.title('Interpolation Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

# 显示图像
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3. 样条插值
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 生成不规则的温度数据
np.random.seed(42)  # 固定随机种子
time_points = np.sort(np.random.choice(np.linspace(0, 24, 50), 12, replace=False))  # 随机12个时间点，范围0到24小时
temperature_points = np.sin(time_points * np.pi / 12) * 10 + np.random.normal(0, 1, len(time_points))  # 温度数据模拟

# 使用样条插值进行数据补全
cs = CubicSpline(time_points, temperature_points)  # 创建样条插值对象

# 生成插值点
time_fine = np.linspace(0, 24, 200)
temperature_fine = cs(time_fine)
temperature_derivative2 = cs(time_fine, 2)  # 计算二阶导数

# 设置画布和子图布局
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# 图形1：原始数据点与样条插值曲线
ax1.plot(time_fine, temperature_fine, label="Cubic Spline Interpolation", color="blue", linewidth=2)
ax1.scatter(time_points, temperature_points, color="red", label="Original Data Points", s=50)
ax1.set_title("Temperature over Time with Cubic Spline Interpolation")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
ax1.grid(True)

# 图形2：样条插值后温度的二阶导数（加速度）
ax2.plot(time_fine, temperature_derivative2, label="Second Derivative (Acceleration)", color="green", linewidth=2)
ax2.set_title("Second Derivative of Temperature (Acceleration)")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Acceleration of Temperature Change")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4. 拉格朗日插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# 生成虚拟数据集
# 原始数据点
x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 0, 5, 3, 4, 2])

# 创建拉格朗日插值多项式
poly = lagrange(x, y)

# 生成更密集的x，用于绘制平滑的插值曲线
x_dense = np.linspace(min(x), max(x), 500)
y_dense = poly(x_dense)

# 生成测试数据，模拟"真实值" (在这个案例中, 假设我们知道真实值)
y_true = np.sin(x_dense) + 2  # 真实值可以人为定义为一个复杂的函数

# 计算插值误差
error = y_true - y_dense

# 绘制插值曲线和误差图

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=100)

# 1. 原始数据点与插值曲线图
ax1.plot(x_dense, y_dense, label='Lagrange Interpolation', color='blue', linewidth=2)
ax1.scatter(x, y, color='red', label='Original Data Points', zorder=5)
ax1.plot(x_dense, y_true, label='True Values (sin(x) + 2)', color='green', linestyle='--', linewidth=2)

ax1.set_title('Lagrange Interpolation and Original Data')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# 2. 插值误差图
ax2.plot(x_dense, error, label='Interpolation Error', color='purple', linewidth=2)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Interpolation Error (True Value - Interpolated Value)')
ax2.set_xlabel('x')
ax2.set_ylabel('Error')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5. 牛顿插值

import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据
np.random.seed(0)
x = np.linspace(0, 10, 10)  # 原始数据x点
y = np.sin(x) + np.random.normal(0, 0.1, len(x))  # 添加一些噪声的原始y值

# 牛顿插值算法实现
def newton_interpolation(x, y, x_new):
    n = len(x)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x[i + j] - x[i])

    # 计算插值值
    y_new = np.zeros_like(x_new)
    for i in range(len(x_new)):
        term = 1
        y_new[i] = divided_diff[0, 0]
        for j in range(1, n):
            term *= (x_new[i] - x[j - 1])
            y_new[i] += divided_diff[0, j] * term

    return y_new

# 插值计算
x_new = np.linspace(0, 10, 100)  # 新的x点
y_interp = newton_interpolation(x, y, x_new)  # 牛顿插值获得的y值

# 真实的sin(x)曲线
y_true = np.sin(x_new)

# 误差计算
error = y_true - y_interp

# 绘图
plt.figure(figsize=(12, 8))

# 图1：原始数据点与插值曲线
plt.subplot(2, 1, 1)
plt.scatter(x, y, color='red', label='Original Data Points')
plt.plot(x_new, y_interp, color='blue', label='Newton Interpolation Curve')
plt.plot(x_new, y_true, color='green', linestyle='--', label='True sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton Interpolation vs. True Function')
plt.legend()
plt.grid()

# 图2：误差分析图
plt.subplot(2, 1, 2)
plt.plot(x_new, error, color='purple', label='Interpolation Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Interpolation Error Analysis')
plt.legend()
plt.grid()

# 展示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6. 最近邻插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 生成二维数据集
np.random.seed(0)
x = np.random.rand(100) * 10
y = np.random.rand(100) * 10
z = np.sin(x/2) * np.cos(y/2)

# 添加一些NaN值模拟缺失数据
mask = np.random.choice([True, False], size=z.shape, p=[0.1, 0.9])
z[mask] = np.nan

# 定义插值网格
xi = np.linspace(0, 10, 100)
yi = np.linspace(0, 10, 100)
xi, yi = np.meshgrid(xi, yi)

# 使用最近邻插值填补缺失值
zi_nn = griddata((x[~mask], y[~mask]), z[~mask], (xi, yi), method='nearest')

# 创建图形窗口
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 1. 原始数据分布图（包含NaN）
sc1 = axs[0].scatter(x, y, c=z, cmap='jet', s=50)
axs[0].set_title('Original Data with NaN')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
fig.colorbar(sc1, ax=axs[0])

# 2. 最近邻插值后数据分布图
sc2 = axs[1].contourf(xi, yi, zi_nn, cmap='rainbow', levels=100)
axs[1].set_title('Data after Nearest Neighbor Interpolation')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')
fig.colorbar(sc2, ax=axs[1])

# 3. 插值前后差异图
zi_original = griddata((x, y), z, (xi, yi), method='nearest', fill_value=np.nan)
diff = np.abs(zi_nn - zi_original)
sc3 = axs[2].contourf(xi, yi, diff, cmap='plasma', levels=100)
axs[2].set_title('Difference between Original and Interpolated Data')
axs[2].set_xlabel('X-axis')
axs[2].set_ylabel('Y-axis')
fig.colorbar(sc3, ax=axs[2])

# 调整图像布局
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7. 分段常数插值

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 生成虚拟数据
np.random.seed(42)
time_stamps = np.sort(np.random.choice(range(0, 100), size=15, replace=False))  # 时间点
temperature_group_1 = np.random.uniform(20, 30, size=len(time_stamps))  # 区域1温度
temperature_group_2 = np.random.uniform(25, 35, size=len(time_stamps))  # 区域2温度

# 创建DataFrame
df = pd.DataFrame({
    'Time': time_stamps,
    'Temperature_Group_1': temperature_group_1,
    'Temperature_Group_2': temperature_group_2
})

# 分段常数插值
time_full = np.arange(0, 100)  # 定义插值后的完整时间范围
f_const_1 = interp1d(df['Time'], df['Temperature_Group_1'], kind='previous', fill_value="extrapolate")
f_const_2 = interp1d(df['Time'], df['Temperature_Group_2'], kind='previous', fill_value="extrapolate")

temperature_interpolated_1 = f_const_1(time_full)
temperature_interpolated_2 = f_const_2(time_full)

# 计算区域平均温度
mean_temp_1 = np.mean(temperature_interpolated_1)
mean_temp_2 = np.mean(temperature_interpolated_2)

# 绘图
plt.figure(figsize=(12, 8))

# 图1：原始数据点
plt.subplot(3, 1, 1)
plt.plot(df['Time'], df['Temperature_Group_1'], 'o-', label="Group 1 - Original", color='blue')
plt.plot(df['Time'], df['Temperature_Group_2'], 'o-', label="Group 2 - Original", color='green')
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Original Temperature Data Points")
plt.legend()
plt.grid(True)

# 图2：分段常数插值
plt.subplot(3, 1, 2)
plt.step(time_full, temperature_interpolated_1, where='post', label="Group 1 - Step Interpolated", color='blue')
plt.step(time_full, temperature_interpolated_2, where='post', label="Group 2 - Step Interpolated", color='green')
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Data with Stepwise Constant Interpolation")
plt.legend()
plt.grid(True)

# 图3：区域平均温度对比
plt.subplot(3, 1, 3)
plt.bar(['Group 1', 'Group 2'], [mean_temp_1, mean_temp_2], color=['blue', 'green'])
plt.xlabel("Group")
plt.ylabel("Average Temperature (°C)")
plt.title("Average Temperature Comparison After Interpolation")
plt.grid(True)

# 调整布局和显示
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8. 高斯过程回归插值

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. 生成虚拟数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(30, 1), axis=0)  # 输入数据
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # 真实值 + 噪声

# 测试点
X_test = np.linspace(0, 5, 1000).reshape(-1, 1)

# 2. 配置高斯过程回归模型
# 使用常数核和RBF核的乘积
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# 3. 拟合数据
gp.fit(X, y)

# 4. 预测
y_pred, sigma = gp.predict(X_test, return_std=True)

# 5. 计算残差
y_train_pred, _ = gp.predict(X, return_std=True)
residuals = y - y_train_pred

# 6. 绘图
plt.figure(figsize=(12, 8))

# 第一张图：原始数据与预测插值曲线
plt.subplot(2, 1, 1)
plt.plot(X_test, y_pred, 'b-', label='Prediction (mean)', lw=2)  # 预测均值曲线
plt.fill_between(X_test.ravel(), y_pred - 2*sigma, y_pred + 2*sigma, color='lightgray', alpha=0.5, label='Confidence interval (±2σ)')  # 置信区间
plt.scatter(X, y, c='r', s=50, zorder=10, label='Observed data')  # 真实数据点
plt.title('Gaussian Process Regression with Confidence Interval')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend(loc='upper left')

# 第二张图：残差分析
plt.subplot(2, 1, 2)
plt.scatter(X, residuals, c='purple', s=50, zorder=10)
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.title('Residuals Analysis')
plt.xlabel('Input')
plt.ylabel('Residuals (True - Predicted)')

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 9. 克里金插值
# pip install pykrige
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# 生成虚拟数据集
np.random.seed(42)
num_samples = 30  # 原始观测点的数量
x_observed = np.random.uniform(0, 100, num_samples)
y_observed = np.random.uniform(0, 100, num_samples)
z_observed = np.sin(x_observed / 10) * np.cos(y_observed / 10) + np.random.normal(0, 0.1, num_samples)

# 定义插值网格
grid_x = np.linspace(0, 100, 200)
grid_y = np.linspace(0, 100, 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# 使用普通克里金插值 (Ordinary Kriging)
OK = OrdinaryKriging(
    x_observed, y_observed, z_observed,
    variogram_model="spherical",
    verbose=False, enable_plotting=False
)
z_kriging, z_variance = OK.execute("grid", grid_x[0], grid_y[:, 0])

# 创建图形
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 原始数据分布图
scatter = ax[0].scatter(x_observed, y_observed, c=z_observed, cmap="coolwarm", s=50, edgecolor='k')
ax[0].set_title("Observed Data Distribution")
ax[0].set_xlabel("X coordinate")
ax[0].set_ylabel("Y coordinate")
fig.colorbar(scatter, ax=ax[0], label="Observed Value")

# 克里金插值结果图
im = ax[1].imshow(z_kriging, extent=(0, 100, 0, 100), origin="lower", cmap="coolwarm")
ax[1].set_title("Kriging Interpolation Result")
ax[1].set_xlabel("X coordinate")
ax[1].set_ylabel("Y coordinate")
fig.colorbar(im, ax=ax[1], label="Interpolated Value")

# 插值误差（方差）分布图
im_var = ax[2].imshow(z_variance, extent=(0, 100, 0, 100), origin="lower", cmap="YlOrBr")
ax[2].set_title("Interpolation Variance")
ax[2].set_xlabel("X coordinate")
ax[2].set_ylabel("Y coordinate")
fig.colorbar(im_var, ax=ax[2], label="Variance")

# 显示所有图形
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 10. 双线性/三线性插值

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 创建虚拟数据
def generate_2d_data():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 4, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    return X, Y, Z

def generate_3d_data():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 4, 5)
    z = np.linspace(0, 4, 5)
    X, Y, Z = np.meshgrid(x, y, z)
    V = np.sin(X) * np.cos(Y) * np.sin(Z)
    return X, Y, Z, V

# 进行双线性插值
def bilinear_interpolation(X, Y, Z):
    xi = np.linspace(0, 4, 100)
    yi = np.linspace(0, 4, 100)
    XI, YI = np.meshgrid(xi, yi)
    Zi = griddata((X.flatten(), Y.flatten()), Z.flatten(), (XI, YI), method='linear')
    return XI, YI, Zi

# 进行三线性插值
def trilinear_interpolation(X, Y, Z, V):
    xi = np.linspace(0, 4, 20)
    yi = np.linspace(0, 4, 20)
    zi = np.linspace(0, 4, 20)
    XI, YI, ZI = np.meshgrid(xi, yi, zi)
    Vi = griddata((X.flatten(), Y.flatten(), Z.flatten()), V.flatten(), (XI, YI, ZI), method='linear')
    return XI, YI, ZI, Vi

# 绘制双线性插值结果
def plot_bilinear(X, Y, Z, XI, YI, Zi):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='jet')
    plt.title("Original Data (2D)")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.contourf(XI, YI, Zi, levels=15, cmap='jet')
    plt.title("Bilinear Interpolated Data (2D)")
    plt.colorbar()

# 绘制三线性插值结果
def plot_trilinear(X, Y, Z, V, XI, YI, ZI, Vi):
    fig = plt.figure(figsize=(10, 8))

    # 绘制原始数据切片
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    slice_z = V[:, :, 2]
    ax1.plot_surface(X[:, :, 2], Y[:, :, 2], slice_z, cmap='jet')
    ax1.set_title('Original Data (Slice z=2)')

    # 绘制插值后的切片
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    slice_zi = Vi[:, :, 10]
    ax2.plot_surface(XI[:, :, 10], YI[:, :, 10], slice_zi, cmap='jet')
    ax2.set_title('Trilinear Interpolation (Slice z=10)')

    # 绘制体积可视化
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=V.flatten(), cmap='jet')
    ax3.set_title('Original 3D Data Points')

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(XI.flatten(), YI.flatten(), ZI.flatten(), c=Vi.flatten(), cmap='jet')
    ax4.set_title('Trilinear Interpolated 3D Data')

# 主程序
X2, Y2, Z2 = generate_2d_data()
XI2, YI2, Zi2 = bilinear_interpolation(X2, Y2, Z2)
plot_bilinear(X2, Y2, Z2, XI2, YI2, Zi2)

X3, Y3, Z3, V3 = generate_3d_data()
XI3, YI3, ZI3, Vi3 = trilinear_interpolation(X3, Y3, Z3, V3)
plot_trilinear(X3, Y3, Z3, V3, XI3, YI3, ZI3, Vi3)

plt.tight_layout()
plt.show()










