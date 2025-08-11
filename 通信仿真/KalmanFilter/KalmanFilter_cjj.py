#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:12:15 2024

@author: jack
https://blog.csdn.net/linkcian/article/details/103824169

https://blog.csdn.net/Canvaskan/article/details/115503937

https://blog.csdn.net/WANGWUSHAN/article/details/134256098?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-134256098-blog-115503937.235^v43^pc_blog_bottom_relevance_base5&spm=1001.2101.3001.4242.1&utm_relevant_index=3




https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485339&idx=1&sn=b135b3ac7666d630b88ce2ea4d03060c&chksm=c0da33473e9e5476a683f37c3090f2530ebf97f485921763f0d8fdc91aec9a694cc086382205&mpshare=1&scene=1&srcid=0831u291lr0yKBCoz30qYSJz&sharer_shareinfo=0de271fe5b9e36a345c871dbfe27c5c8&sharer_shareinfo_first=0de271fe5b9e36a345c871dbfe27c5c8&exportkey=n_ChQIAhIQfc%2BoPavCLl77t9tbhcuKjhKfAgIE97dBBAEAAAAAABj9ID1zEhkAAAAOpnltbLcz9gKNyK89dVj0vKEApSKdBgS%2BCTf%2BGwZ818xuIzREkSrQAs7iLQVV1SXiaUvsT%2FEOSmu77mdjhMqtSXklGtNmeKdrPt5d37%2FGdInoBYFaZmOeVVT6SZ0fnGuO8YWUk5%2F3y4ZF53Po6Ay4BeZXDF4y5md3LLGL97h9VFAT8AQ0ebt3murAeZjAr4OkRnxKj1MI4f9MsTk28YZpvqPJU31PpHYbOCoVHW6C1yR2eJPEGWYOsmZSAKF%2Ft%2FkkEtiHqSKJOK4hkaeTosdIf9f7JJVL%2F7i5rkex5C0ubSkOmtZQmJY1bsnifoP4YSkcnZLJi%2FUY%2FuDGHLMmATTunM4vBYS720gO&acctmode=0&pass_ticket=7wJTNpQT805vanvdvQ%2FuNtZwzZDZvcxtJ5xYFqyHFh7NiVH9wHxfzJ4OJIgt53kD&wx_header=0#rd

https://github.com/artificialIntelligenceStudy/Kalman-and-Bayesian-Filters-in-Python-chinese?tab=readme-ov-file


https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485974&idx=1&sn=19fd8a052e51ae5b1d58c90f0d003cd9&chksm=c0be54ef5436b0d785d56a14488f85545e8e58f5d2d7c76c0102fab52a673acd902c4cdaed23&mpshare=1&scene=1&srcid=0920DsqVnH9yCUA5l9G0vwWD&sharer_shareinfo=b9c8cc13821f77027d5207eda71df00f&sharer_shareinfo_first=b9c8cc13821f77027d5207eda71df00f&exportkey=n_ChQIAhIQJZvTmaMZ3j16lnOYkpz%2FjRKfAgIE97dBBAEAAAAAABvsNvBDFIgAAAAOpnltbLcz9gKNyK89dVj0QzG7cK9WTpQEmBvhgzDRqacFS5f6SYIUCYgukp17iE0GkopoIwz2kJEIvOX31ql42mA8%2Fi7MzBTMkTXz2g8etC6AmS77Ds579lLzRWffU5Cdde6MDAVw%2BpKyUL8aYOT8LXkra9IG07JclLperqXaR0yCMv8M4BR4zHvpXW3ohjksBITUbHQhyswSjhzxTl90hvCXzlqS16tbUkVGLJ075wJgKTbllT57rHkbq3n2%2FUeL1fa%2B5eJIDvqnpn3IWAQXgHGM2p4Jb23C8u4GRpi1Ribu0cfjYOoZsD6%2F0P7KEaO6HEyC2STo1MRaaHWpHBAlhbsdBeP5sJBA&acctmode=0&pass_ticket=EJImrC01GRqrn0cLq8b%2Bl%2F9eDdpYuPNlRtI0mVed5RD%2BgCVPPJc823cBz2L6pDYI&wx_header=0#rd

# https://github.com/loveuav/Kalman-and-Bayesian-Filters-in-Python

# https://github.com/dougszumski/KalmanFilter/tree/master

# https://github.com/zziz/kalman-filter

# https://github.com/rlabbe/filterpy

# https://github.com/milsto/robust-kalman/tree/master


"""




#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485956&idx=1&sn=abac1ec541bf3c8f51cf38114c22fc0d&chksm=c1ae6c993e3698811a85a706a4389cc6d8d8ffbdab05eb7462452868215344fa76287aa3da3b&mpshare=1&scene=1&srcid=0822GQ8QZOOaAbFcHOYS6ozJ&sharer_shareinfo=792b67ed4ccef1bee1c59d47e8285e91&sharer_shareinfo_first=792b67ed4ccef1bee1c59d47e8285e91&exportkey=n_ChQIAhIQNvSF0aFR9HfJVszrGewg%2BhKfAgIE97dBBAEAAAAAALyqIGEjphAAAAAOpnltbLcz9gKNyK89dVj0M7UjKheWmvj7E62WcFIB2ejlGDIP%2BD39Lj7wRBB%2FKOBEHidvMsrcbhWl3CfGBb9ThSAfcizDTQW1OWhW8npbVbzVLLFl2k%2B8vjBde50MIWp6Mnl02PbpqaBbY7H3r8zrV9PYsJ5cgYd4XLTg11uFwmTUIW6L%2Fm4P34sLQIjSyCSPWPT5tDbJpkR7rph3%2F9qRmqmBzzArTLwkM2RMS0SeAVWTtTZfOLpNNuMh%2Bd7UksOLQI7rl581mG8FcM9ts9zGEoYAK%2Biyj7%2FsHhjITPyU0AsrS6aql4%2BPx1tiUYF2qe8FUB5BCawmA8QOJmXbfbeTS6FB4zkI%2FNVk&acctmode=0&pass_ticket=4JksxdRP9ZK%2BlfuY8ugpV87Z0z2yUPk0b12Oab%2FDhhovXUUPFL8AK5gxyje%2FNWTP&wx_header=0#rd
# 卡尔曼滤波器通过将当前时刻的观测值与上一时刻的状态预测进行结合，来动态更新信号的估计值，从而实现对状态的最佳估计。这种方法不仅能有效地平滑噪声，还能根据系统的动态特性进行预测，使其在各种实时系统中表现出色。

# 这个案例模拟了一辆汽车沿直线运动的过程，我们用卡尔曼滤波器来估计汽车的速度和位置。假设我们可以测量汽车的位置，但测量中存在噪声。

import numpy as np
import matplotlib.pyplot as plt

# 模拟数据集生成
np.random.seed(999)  # 固定随机数种子

# 时间步数
n_timesteps = 200

# 实际初始状态 (位置=0，速度=1)
true_initial_position = 0
true_initial_velocity = 1

# 实际状态 (位置和速度) 的噪声
process_noise_std_position = 0.1
process_noise_std_velocity = 0.1

# 观测噪声 (位置的测量噪声)
measurement_noise_std = 0.5

# 状态和观测矩阵
A = np.array([[1, 1],  # 状态转移矩阵 (状态更新)
              [0, 1]])  # 速度保持恒定
H = np.array([[1, 0]])  # 观测矩阵 (仅观测位置)

# 初始化状态
true_states = np.zeros((n_timesteps, 2))
true_states[0] = [true_initial_position, true_initial_velocity]

# 初始化观测
measurements = np.zeros(n_timesteps)

# 生成实际状态和观测值
for t in range(1, n_timesteps):
    process_noise = np.random.normal(0, [process_noise_std_position, process_noise_std_velocity])
    true_states[t] = A @ true_states[t-1] + process_noise
    measurements[t] = (H @ true_states[t])[0] + np.random.normal(0, measurement_noise_std)

# 卡尔曼滤波器
# 初始化估计的状态
estimated_states = np.zeros((n_timesteps, 2))
estimated_states[0] = [0, 0]  # 初始位置和速度的估计值
P = np.eye(2)                 # 初始估计误差协方差矩阵

# 定义过程噪声协方差矩阵和观测噪声协方差矩阵。
Q = np.array([[process_noise_std_position**2, 0], [0, process_noise_std_velocity**2]])  # 过程噪声协方差矩阵
R = measurement_noise_std**2                                                            # 观测噪声协方差矩阵。

# 卡尔曼滤波器迭代
for t in range(1, n_timesteps):
    ###>>>>>>>>>>>>>>>> 预测阶段
    predicted_state = A @ estimated_states[t-1]  # 预测下一个状态(状态预测,当前时刻的状态预测): hat{x}_{t|t-1} = A @ hat{x}_{t-1|t-1}
    P = A @ P @ A.T + Q                          # 预测下一个协方差(更新误差协方差): P_{t|t-1} = A@P_{t-1|t-1}@A^T + Q
    ###>>>>>>>>>>>>>>>> 更新阶段
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)      # 计算卡尔曼增益: K_t = P_{t|t-1} @ H^T @ (H @ P_{t|t-1} @ H^T + R)^{-1}
    ### 结合观测值修正预测
    estimated_states[t] = predicted_state + K @ (measurements[t] - H @ predicted_state)  # 更新状态估计: hat{x}_{t|t} = hat{x}_{t|t-1} + K_t @ (z_t - H @ hat{x}_{t|t-1})
    P = (np.eye(2) - K @ H) @ P                          # 更新误差协方差：P_{t|t} = (I - K_t @ H) @ P_{t|t-1}

# 绘制结果
plt.figure(figsize=(12, 12))

# 图1：真实位置 vs 估计位置 vs 观测位置
plt.subplot(2, 1, 1)
plt.plot(true_states[:, 0], label='True Position', color='g')
plt.plot(estimated_states[:, 0], label='Estimated Position', color='b', linestyle='--')
plt.scatter(range(n_timesteps), measurements, label='Measured Position', color='r', marker='o')
plt.title('Position: True vs Estimated vs Measured')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()

# 图2：真实速度 vs 估计速度
plt.subplot(2, 1, 2)
plt.plot(true_states[:, 1], label='True Velocity', color='g')
plt.plot(estimated_states[:, 1], label='Estimated Velocity', color='b', linestyle='--')
plt.title('Velocity: True vs Estimated')
plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>
# https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485974&idx=1&sn=19fd8a052e51ae5b1d58c90f0d003cd9&chksm=c0be54ef5436b0d785d56a14488f85545e8e58f5d2d7c76c0102fab52a673acd902c4cdaed23&mpshare=1&scene=1&srcid=0920DsqVnH9yCUA5l9G0vwWD&sharer_shareinfo=b9c8cc13821f77027d5207eda71df00f&sharer_shareinfo_first=b9c8cc13821f77027d5207eda71df00f&exportkey=n_ChQIAhIQJZvTmaMZ3j16lnOYkpz%2FjRKfAgIE97dBBAEAAAAAABvsNvBDFIgAAAAOpnltbLcz9gKNyK89dVj0QzG7cK9WTpQEmBvhgzDRqacFS5f6SYIUCYgukp17iE0GkopoIwz2kJEIvOX31ql42mA8%2Fi7MzBTMkTXz2g8etC6AmS77Ds579lLzRWffU5Cdde6MDAVw%2BpKyUL8aYOT8LXkra9IG07JclLperqXaR0yCMv8M4BR4zHvpXW3ohjksBITUbHQhyswSjhzxTl90hvCXzlqS16tbUkVGLJ075wJgKTbllT57rHkbq3n2%2FUeL1fa%2B5eJIDvqnpn3IWAQXgHGM2p4Jb23C8u4GRpi1Ribu0cfjYOoZsD6%2F0P7KEaO6HEyC2STo1MRaaHWpHBAlhbsdBeP5sJBA&acctmode=0&pass_ticket=EJImrC01GRqrn0cLq8b%2Bl%2F9eDdpYuPNlRtI0mVed5RD%2BgCVPPJc823cBz2L6pDYI&wx_header=0#rd
# 1 | 数据滤波：探讨卡尔曼滤波、SG滤波与组合滤波
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

np.random.seed(1)  # 固定随机数种子

def find_chinese_font():
    font_list = font_manager.fontManager.ttflist
    for font in font_list:
        if "SimHei" in font.name:
            return font.fname
        elif "SimSun" in font.name:
            return font.fname
    return None

font_path = find_chinese_font()
if font_path:
    my_font = font_manager.FontProperties(fname=font_path)
else:
    print("未找到中文字体")

def predict(F, B, u, x_est, P_est, Q):
    x_pred = F @ x_est + B * u
    P_pred = F @ P_est @ F.T + Q
    return x_pred, P_pred

def update(x_pred, P_pred, H, z, R):
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_est = x_pred + K @ (z - H @ x_pred)
    P_est = (np.eye(H.shape[0]) - K @ H) @ P_pred
    return x_est, P_est

Q_vals = [0.1, 1, 10]
R_vals = [0.1, 1, 10]
dt = 1
t1 = np.linspace(0, 100, 101)
n_timesteps = len(t1)
F = np.array([[1, dt], [0, 1]]) # 状态转移矩阵，描述了状态如何从  时刻转移到  时刻
H = np.array([[1, 0]]) # 观测矩阵，将状态空间映射到观测空间。
B = np.array([0, 0]) # 控制矩阵（如果有控制输入）。
u = 0

errs = {}
ests = {}
leg = {}
idx = 0

# % 生成真实轨迹
x_true = np.zeros((n_timesteps, 2))
x_true[0] = np.array([0, 1])
measurements = np.zeros(n_timesteps)

for t in range(1, n_timesteps):
    x = F @ x_true[t-1] + B * u + np.random.randn(2 )
    z = H @ x + np.random.randn()
    x_true[t] = x
    measurements[t] = z[0]

for Q1 in Q_vals:
    for R in R_vals: # # 测量噪声协方差
        Q = Q1 * np.eye(2) # 过程噪声协方差
        P_est = np.eye(2)  # 初始估计误差协方差矩阵
        x_est = np.zeros((n_timesteps, 2))
        x_est[0] = np.array([0, 0])  # 初始位置和速度的估计值

        for t in range(1, n_timesteps):
            x_pred, P_pred = predict(F, B, u, x_est[t-1], P_est, Q)
            x_est_tmp, P_est = update(x_pred, P_pred, H, measurements[t], R)
            x_est[t] = x_est_tmp
        err = np.sqrt(np.sum(np.abs(x_true - x_est)**2, axis = 1), )
        errs[idx] = err
        ests[idx] = x_est
        leg[idx] = f"Q={Q1:.2f}, R = {R:.2f}"
        idx = idx + 1

############## plot_errors
fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout = True)# constrained_layout=True
for i in range(len(errs)):
    axs.plot(t1, errs[i], lw = 2, label = leg[i])

font1 = {'family':'Times New Roman','style':'normal','size':18, }
legend1 = axs.legend(loc = 'upper right', borderaxespad = 0, edgecolor = 'black', prop = font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# font = {'family':'Times New Roman','style':'normal','size':22}
axs.set_xlabel("时间", fontproperties = my_font, fontsize = 18)
axs.set_ylabel("误差 (欧几里得范数)", fontproperties = my_font, fontsize = 18 )
plt.show()
plt.close()
############## plot_trajectories

fig, axs = plt.subplots(2, 1, figsize=(10, 12), constrained_layout = True)
axs[0].plot(t1, x_true[:,0], lw = 2, label = "真实位置", color = 'k', ls = 'none',  marker = 'o', mfc = 'none', ms = 18, markevery = 10)
axs[0].plot(t1, measurements, lw = 2, label = "测量值", color = 'r', ls = 'none',  marker = '*', mfc = 'none', ms = 18, markevery = 10)
for i in range(len(ests)):
    axs[0].plot(t1, ests[i][:,0], lw = 2, label = leg[i])

font1 = {'family':'Times New Roman','style':'normal','size':14, }
legend1 = axs[0].legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = my_font, fontsize = 18)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# font = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_xlabel("时间", fontproperties = my_font, fontsize = 18)
axs[0].set_ylabel("位置", fontproperties = my_font, fontsize = 18 )
axs[0].set_title("位置估计", fontproperties = my_font, fontsize = 18 )

axs[1].plot(t1, x_true[:,1], lw = 2, label = "真实速度", color = 'r', ls = 'none',  marker = '*', mfc = 'none', ms = 18, markevery = 10)
for i in range(len(ests)):
    axs[1].plot(t1, ests[i][:,1], lw = 2, label = leg[i])

font1 = {'family':'Times New Roman','style':'normal','size':12, }
legend1 = axs[1].legend(loc = 'best', borderaxespad = 0, edgecolor = 'black', prop = my_font, fontsize = 18)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# font = {'family':'Times New Roman','style':'normal','size':22}
axs[1].set_xlabel("时间", fontproperties = my_font, fontsize = 18)
axs[1].set_ylabel("速度", fontproperties = my_font, fontsize = 18 )
axs[1].set_title("速度估计", fontproperties = my_font, fontsize = 18 )

plt.show()
plt.close()



#%% https://mp.weixin.qq.com/s?__biz=MzE5MTA2MjYzNQ==&mid=2247483736&idx=1&sn=a4ac67987f01a7baa88150dfb733c1f6&chksm=970dc060caf402200b11fc2b9c92a8b02563f63e38c7ebe801ceeded8da4b293ae2c9cd22630&mpshare=1&scene=1&srcid=0801oosha62yStaDQ4ScPmre&sharer_shareinfo=9ea083510294f4274298cb74da57918a&sharer_shareinfo_first=b82205342fc71e1a759b98d32523bbaa&exportkey=n_ChQIAhIQegj27IrssjU2amqGdC%2BRYRKfAgIE97dBBAEAAAAAAGXaFMUT%2FNwAAAAOpnltbLcz9gKNyK89dVj0v0PPNBwG6rlY5OJbpiQN4K3zgkojyIhAPm%2FQ4vCXe8xcNOI4s9rqqxvj7M4J8pYk%2FObQnzEfSEHJBxK6tKoP4eGiMC9Ervpysw%2BNoME%2BGBQwCFiiHYUPrAxuH%2BclEZzg%2FyKIabOF5vuz1qGZyfENkDwG9SaVLNGdtrMiA7N6ZmxE5o%2Bxxgx6OwOjLI3tM2TUSHN%2FnM5G9RZbUOldXWlcoi1ZrNHTIXf0YKUY52z91wRzkPHMJPj0ZyFSuO5e3g8%2FXRi5%2FSKsRGll2TeIfw7fmBuxBMzG04o3DfbQx%2BKcybLGKqNA2i6PUv%2BkVs519Xadk0sNrD4qe8NU&acctmode=0&pass_ticket=2SGNrPnbdvuqKkqnHSF%2B%2FE7eNlWfLOSMHS5TD2M1y6Fs8NZbdLn8CnNGfNqrot9j&wx_header=0#rd

#%% 卡尔曼滤波——滤波就是降噪吗？

import numpy as np
import matplotlib.pyplot as plt
# 设置支持中文的字体（SimHei 是黑体，Microsoft YaHei 是雅黑）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False# 正常显示负号

dt = 0.1# 采样时间10Hz

# 状态转移矩阵
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 测量矩阵
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# 过程噪声输入矩阵
G = np.array([
    [dt*dt, 0],
    [0, dt*dt],
    [dt, 0],
    [0, dt]
])

# 加速度噪声标准差 (m/s²)
accel_noise = 0.2
# GPS噪声标准差 (m)
gps_noise = 2

# 过程噪声协方差矩阵
Q = np.diag([(0.5*dt**2)**2, (0.5*dt**2)**2, dt**2, dt**2]) * accel_noise**2
# 测量噪声协方差矩阵
R = np.array([
    [gps_noise**2, 0],
    [0, gps_noise**2]
])

# 初始化真实状态
true_x = np.array([0, 0, 0.5, 0.5])  # 初始状态 [px, py, vx, vy]

# 初始化卡尔曼滤波器
x_est = np.array([0, 0, 0.5, 0.5])  # 初始估计状态 [px, py, vx, vy]
P_est = np.eye(4)  # 初始协方差矩阵

# 存储结果
true_states, estimates, observations = [], [], []

# 模拟100个时间步
for t in range(1000):
    # 1. 生成真实状态 (含过程噪声)
    true_x = F @ true_x + G @ np.random.normal(0, accel_noise, 2)

    # 2. 生成GPS观测 (含噪声)
    z = H @ true_x + np.random.normal(0, gps_noise, 2)

    # 3. 卡尔曼预测
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q

    # 4. 卡尔曼更新
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_est = x_pred + K @ (z - H @ x_pred)
    P_est = (np.eye(4) - K @ H) @ P_pred

    # 存储数据
    true_states.append(true_x[:2])
    estimates.append(x_est[:2])
    observations.append(z)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(*zip(*true_states), 'g-', label="真实轨迹")
plt.plot(*zip(*observations), 'ro', markersize=3, label="GPS观测")
plt.plot(*zip(*estimates), 'b--', linewidth=2, label="卡尔曼估计")
plt.legend()
plt.title("CV模型下的卡尔曼滤波跟踪")
plt.xlabel("X位置 (m)")
plt.ylabel("Y位置 (m)")
plt.grid(True)
plt.show()


#%% 更通用的卡尔曼滤波方程形式
# https://mp.weixin.qq.com/s?__biz=MzE5MTA2MjYzNQ==&mid=2247483720&idx=1&sn=ce7bafcbced28d597ac2bff68b29de61&chksm=97d7b4efed4db82930a7aa8be44a69e96f33dd343cbc7ca84599375ca7a5717c887185574807&mpshare=1&scene=1&srcid=0801fMySvTr3CU8YnZxO4q5A&sharer_shareinfo=09fb5b73866e3cb1282823d9cd98c5be&sharer_shareinfo_first=09fb5b73866e3cb1282823d9cd98c5be&exportkey=n_ChQIAhIQ%2Bgn6uCvEz2A5G2TOOlZXiRKfAgIE97dBBAEAAAAAALMEKV5FOIkAAAAOpnltbLcz9gKNyK89dVj0vF1IvpvTlP0i3H0jfAjhTqjmRwvBd5VEMKkIMDC3UApp%2F4z0l4imMGLuoGKvt4TSoNI5OCTQQnugkG4BGpFd3kVUh82trjz4etGhBpzpvmG8wcqCD6gZY8s5T57tqR1B1uVM%2F7ISkz%2FwMO2wHusztRye3wK7OMypBGNFduye4OEXZppbTbmlwzFjxxKopgJrTw9lfeQiTXeanVAT81GnA%2BL5RWddDtwTBj62XTnQdZ0MmhvTKOZSjnatquS%2FqBtexX32djBOU6JJpqwbFKyJ%2BaVTHUAee7hhmluhnvtshVifJqR1AHNgZyf1QLBQorOq%2FJ7ccJcI5w2B&acctmode=0&pass_ticket=wq75DY0B2rbuj8aMm5QCSx9rFyyR5NTg%2BoZj4lvZtcGmnz0iVaOsPq0h6Lc0rx5n&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt

# 指定默认字体为黑体（SimHei），Windows 上一般预装；macOS 上可用 'Hei'
plt.rcParams['font.sans-serif'] = ['SimHei']
# 取消负号“−”显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import multivariate_normal

# ————————————— 参数设置 —————————————
dt = 1.0           # 采样时间间隔
N = 50             # 总步数
u = 1.0            # 恒定加速度 (m/s^2)

# 状态转移矩阵 A 和控制输入矩阵 B
A = np.array([[1, dt],
              [0, 1]])
B = np.array([[0.5 * dt**2],
              [dt]])

# 观测矩阵 H：这里只测位置 p_k
H = np.array([[1, 0]])

# 过程噪声协方差 Q，观测噪声协方差 R
Q = 0.01 * np.eye(2)
R = np.array([[1.0]])

# ————————————— 生成真值和观测 —————————————
x_true = np.zeros((2, N))
z = np.zeros(N)
x = np.zeros((2, 1))

for k in range(N):
    # 过程噪声 w ~ N(0, Q)
    w = multivariate_normal.rvs(mean=[0, 0], cov=Q).reshape(2, 1)
    # 真值演化：匀加速运动 + 过程噪声
    x = A @ x + B * u + w
    x_true[:, k] = x.flatten()
    # 观测：位置 + 观测噪声 v ~ N(0, R)
    z[k] = (H @ x).item() + np.sqrt(R) * np.random.randn()

# —————— 卡尔曼滤波：不含控制输入 ——————
x_est_nc = np.zeros((2, N))
P = np.eye(2)
x_nc = np.zeros((2, 1))

for k in range(N):
    # 预测（无 u 项）
    x_pred = A @ x_nc
    P_pred = A @ P @ A.T + Q

    # 更新
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_nc = x_pred + K @ (z[k] - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred

    x_est_nc[:, k] = x_nc.flatten()

# —————— 卡尔曼滤波：含控制输入 ——————
x_est_wc = np.zeros((2, N))
P = np.eye(2)
x_wc = np.zeros((2, 1))

for k in range(N):
    # 预测（加入 u 项）
    x_pred = A @ x_wc + B * u
    P_pred = A @ P @ A.T + Q

    # 更新
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_wc = x_pred + K @ (z[k] - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred

    x_est_wc[:, k] = x_wc.flatten()

# —————— 误差与 RMSE ——————
err_nc = x_est_nc[0, :] - x_true[0, :]
err_wc = x_est_wc[0, :] - x_true[0, :]

rmse_nc = np.sqrt(np.mean(err_nc**2))
rmse_wc = np.sqrt(np.mean(err_wc**2))
print(f'RMSE without input: {rmse_nc:.4f}')
print(f'RMSE with    input: {rmse_wc:.4f}')

# —————— 绘图 ——————
fig, axes = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)

# 上图：位置对比
axes[0].plot(x_true[0, :], 'k-', linewidth=2, label='真实位置')
axes[0].plot(x_est_nc[0, :], 'r--', linewidth=1.5, label='KF 无输入')
axes[0].plot(x_est_wc[0, :], 'b-.', linewidth=1.5, label='KF 含输入')
axes[0].set_ylabel('位置')
axes[0].set_title('卡尔曼滤波——位置估计对比')
axes[0].legend(loc='upper left')
axes[0].grid(True)

# 下图：误差对比
axes[1].plot(err_nc, 'r--', linewidth=1.5, label='误差：无输入')
axes[1].plot(err_wc, 'b-.', linewidth=1.5, label='误差：含输入')
axes[1].set_xlabel('时间步 k')
axes[1].set_ylabel('位置误差')
axes[1].set_title('卡尔曼滤波——位置误差对比')
axes[1].legend(loc='upper left')
axes[1].grid(True)

plt.show()



#%% 什么是集合卡尔曼滤波EnKF，如何应用EnKF实现数据同化？
# https://mp.weixin.qq.com/s?__biz=MzE5MTA2MjYzNQ==&mid=2247483736&idx=1&sn=a4ac67987f01a7baa88150dfb733c1f6&chksm=979dcea87124240036b6b1f39147744a246d2c9a83aebb00599be6c909abda48508058bd208d&mpshare=1&scene=1&srcid=0801oosha62yStaDQ4ScPmre&sharer_shareinfo=aa7e4a308f213d82406b607992d2042f&sharer_shareinfo_first=b82205342fc71e1a759b98d32523bbaa&exportkey=n_ChQIAhIQc4upmm2qulKLI2xdk2NqPRKfAgIE97dBBAEAAAAAADayM%2FQHy%2FQAAAAOpnltbLcz9gKNyK89dVj0aXQJhlZXR2ny5iQkbttIxI3AtBsKWyzt3Ail%2F0QqafovQXnZjdKa4rf4cObrevd1omtmVqI7APTVxOwW9Je6QU2vajq4H3slU3u%2FFEeaKUgbQZ0TdWUiMV13CEKqOGHHNnjosmsLbSeRHDVtbqBAAsRdx4waQVIXUAb5NdJCV5pzAdEmQ0T6wexz%2B1UTptPqTyUpg5TGZmllejkBzs7qHRxalTROYV31XLX6892YLcyvOGv6AA0GaR3zQ7pP1QnrB7YH62c9HwBhjH8hH9Ht7BSoOWeIfVEE8SmjHXLqCDVZSMV9FV3nPrPjNSFpmuRARuKIvYYUVYMl&acctmode=0&pass_ticket=c2JAUiy1xaSHCdTq0d0aDCW5HCQs%2ByWoJgwZreEbr%2F1CfUSpbA1xzTVftWMFN4Mh&wx_header=0#rd
import numpy as np
import matplotlib.pyplot as plt

def EnKF_fast(A, B, H, Q, R, X0_ens, u_seq, y_seq):
    n, N = X0_ens.shape
    K     = u_seq.shape[1]

    # 预分解 Q 和 R
    Lq = np.linalg.cholesky(Q)
    Lr = np.linalg.cholesky(R)

    X_est  = np.zeros((n, K+1, N))
    x_mean = np.zeros((n, K+1))
    X_est[:, 0, :] = X0_ens
    x_mean[:, 0]   = X0_ens.mean(axis=1)

    for k in range(K):
        # —— 向量化预测 ——
        W = Lq @ np.random.randn(n, N)                           # (n, N)
        Xf = A @ X_est[:, k, :] + (B @ u_seq[:, k:k+1]) + W      # (n, N)

        # —— 计算增益 ——
        xf_mean = Xf.mean(axis=1, keepdims=True)                 # (n,1)
        Exf     = Xf - xf_mean                                   # (n,N)
        P_f     = Exf @ Exf.T / (N-1)                            # (n,n)

        S       = H @ P_f @ H.T + R                              # (p,p)
        K_gain  = P_f @ H.T @ np.linalg.inv(S)                    # (n,p)

        # —— 向量化分析 ——
        V   = Lr @ np.random.randn(R.shape[0], N)                # (p,N) 扰动观测噪声
        Yp  = y_seq[:, k+1:k+2] + V                              # (p,N)
        Innov = Yp - H @ Xf                                       # (p,N)
        Xa    = Xf + K_gain @ Innov                              # (n,N)

        X_est[:, k+1, :] = Xa
        x_mean[:, k+1]   = Xa.mean(axis=1)

    return X_est, x_mean

# ========== 参数设定 ==========
n = 100
Q = 0.5 * np.eye(n)
R = 0.01 * np.eye(9)
alpha = 1.0
dx = 1.0
dt = 0.2
r = alpha * dt / dx**2

# 状态转移矩阵A（三对角）
A = (1 - 2 * r) * np.eye(n)
for i in range(n-1):
    A[i, i+1] = r
    A[i+1, i] = r

# 输入矩阵B
B = np.zeros((n, 2))
B[32, 0] = 1  # 注意Python下标从0开始
B[66, 1] = 1

# 观测矩阵H
H = np.zeros((9, n))
for i in range(9):
    H[i, 10*(i+1)-1] = 1  # Python下标从0开始

K = 1000  # 时间步数

# 生成热源输入：两个正弦信号
t = np.arange(K)
u1 = 10 * np.sin(2 * np.pi * t / 50)
u2 = 10 * np.sin(2 * np.pi * t / 80)
u_seq = np.vstack([u1, u2])

# 生成真值初始状态
np.random.seed(0)

x0_true = 300 + 10 * np.random.randn(n)
X_true = np.zeros((n, K+1))
Y = np.zeros((9, K+1))
X_true[:, 0] = x0_true - 300  # 状态使用T-300
for k in range(K):
    w = np.sqrt(0.5) * np.random.randn(n)
    X_true[:, k+1] = A @ X_true[:, k] + B @ u_seq[:, k] + w
    v = 0.1 * np.random.randn(9)
    Y[:, k] = H @ X_true[:, k] + v
v = 0.1 * np.random.randn(9)
Y[:, K] = H @ X_true[:, K] + v

# 标准卡尔曼滤波初始化
x_kf = np.zeros((n, K+1))
P = 1000 * np.eye(n)
x_kf[:, 0] = np.zeros(n)
for k in range(K):
    # 预测
    x_pred = A @ x_kf[:, k] + B @ u_seq[:, k]
    P_pred = A @ P @ A.T + Q
    # 卡尔曼增益
    S = H @ P_pred @ H.T + R
    K_gain = P_pred @ H.T @ np.linalg.inv(S)
    # 更新
    x_kf[:, k+1] = x_pred + K_gain @ (Y[:, k+1] - H @ x_pred)
    P = (np.eye(n) - K_gain @ H) @ P_pred

# 运行EnKF并计算MSE
skip1 = 5
N_list = np.arange(10, 101, skip1)
MSE_kf = np.mean((x_kf - X_true)**2, axis=0)
MSE_enk = np.zeros((len(N_list), K+1))

for idx, N in enumerate(N_list):
    np.random.seed(0)
    X0_ens = 10 * np.random.randn(n, N)
    _, x_mean_enk = EnKF_fast(A, B, H, Q, R, X0_ens, u_seq, Y)
    MSE_enk[idx, :] = np.mean((x_mean_enk - X_true)**2, axis=0)

# 绘制图1：时间序列上的MSE曲线（对数坐标）
plt.figure()
plt.semilogy(np.arange(K+1), MSE_kf, 'k-', linewidth=1.5, label='KF')
plt.rcParams['font.sans-serif'] = ['SimHei']          # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False            # 使 “-” 号正常显示

show_list = [10, 20, 100]
show_list1 = [(N-10)//skip1 for N in show_list]
colors = ['r--', 'g-.', 'b:']
for idx, showidex in enumerate(show_list1):
    plt.semilogy(np.arange(K+1), MSE_enk[showidex, :], colors[idx], linewidth=1.5, label=f'EnKF, N={show_list[idx]}')
plt.legend(loc='best')
plt.xlabel('时间步 k')
plt.ylabel('均方误差 (MSE)')
plt.title('图1: 不同EnKF集合规模与Kalman滤波的状态估计MSE对比')
plt.grid(True)

# 绘制图2：MSE随集合规模的变化（取最后时刻的MSE均值）
plt.figure()
MSE_final = np.mean(MSE_enk, axis=1)
plt.semilogy(N_list, MSE_final, 'o-', linewidth=1.5)
plt.xlabel('集合规模 N')
plt.ylabel('最终状态估计MSE')
plt.title('图2: 不同集合规模下的平均MSE(最后时刻)')
plt.grid(True)
plt.show()

print("EnKF 运行完成，结果已绘制。")


#%% 非线性系统如何应用卡尔曼滤波？扩展卡尔曼滤波EKF篇(含原理、案例与代码实现)
# https://mp.weixin.qq.com/s?__biz=MzE5MTA2MjYzNQ==&mid=2247483746&idx=1&sn=2e9844e7dee0bc83a50bf56dd65e129f&chksm=9727f6794b9e86773e1f02f95c9653f74c82fda2650671f73c42140daa3a88951da695cc9aac&mpshare=1&scene=1&srcid=0811Sts1JIulLacJKFRkt6m6&sharer_shareinfo=61c39115e6b7d0ea6d635f37ad1c9268&sharer_shareinfo_first=61c39115e6b7d0ea6d635f37ad1c9268&exportkey=n_ChQIAhIQB%2BxksmGhwJXMtrP93bGq3xKfAgIE97dBBAEAAAAAAOSSGXW2mkcAAAAOpnltbLcz9gKNyK89dVj0Pzm7G0LXcmL8FkuxDszrh03C6k905YXmvTngyVclcUxOk06%2B3GURloiCz0mSChW5JREgJ9QeYZT5yu5ADj6a89fslW4T4HppGpJuuh8zynBefgl%2Bt5UjNISE3WpdXTlKSaL5rtxcET8Rw7IyVKle08quS4DCbZh34IYDPWRbouzDdJqxif3bKFvTzF%2BrTIgsGt862ZX250oFZyzNgp8zHVfAVMBbijo%2FZP1nMZIF5eWW9zcvKJ1Zhjo%2FnGOVwBJqznbDJFmCOm81HEM6cygxHYbKVFpXqt7%2Bx0uoLjiCpFM3so4F4RKyYIFEEbM9xcyK8PUpJ8t6yGgY&acctmode=0&pass_ticket=4eZfvv6a2DCccxNDcYRMICL%2FcBDHDdX%2F3KvvulbtqBGm7M42EGoFK2IqPV10Kwtl&wx_header=0&poc_token=HMi4mWijMQXs6CamXP5nCaX2SgatQnsl_CMVHnXP


import numpy as np
import matplotlib.pyplot as plt

def wrapToPi(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Set simulation parameters
dt = 0.1           # Time step (s)
N = 500            # Number of steps
R = 20             # Circle radius (m)
omega_true = 0.1   # True angular velocity (rad/s)
v_true = R * omega_true  # True linear speed (m/s)

# Noise characteristics
overlay_std = 0.05   # Speed sensor noise (m/s)
omega_std = 0.005    # Yaw rate noise (rad/s)
gps_std = 1.0        # GPS position noise (m)

# Preallocate arrays
true_state = np.zeros((3, N))   # [x; y; theta]
v_meas = np.zeros(N)
omega_meas = np.zeros(N)
gps_meas = np.zeros((2, N))

# Initial condition: start at angle 0
phi = np.zeros(N)
phi[0] = 0
true_state[:, 0] = [R, 0, phi[0] + np.pi/2]  # [x, y, heading]

# Generate true trajectory and measurements
np.random.seed(0)
for k in range(1, N):
    # Update true heading (polar angle)
    phi[k] = phi[k-1] + omega_true * dt
    # Compute true position on circle
    true_state[0, k] = R * np.cos(phi[k])
    true_state[1, k] = R * np.sin(phi[k])
    true_state[2, k] = phi[k] + np.pi/2
    # Odometry measurements: linear speed + yaw rate
    v_meas[k] = v_true + overlay_std * np.random.randn()
    omega_meas[k] = omega_true + omega_std * np.random.randn()
    # GPS measurement: noisy x, y
    gps_meas[:, k] = true_state[0:2, k] + gps_std * np.random.randn(2)

# EKF initialization
x_ekf = np.zeros((3, N))
P = np.eye(3)
Q = np.diag([0.1**2, 0.1**2, 0.01**2])  # process noise covariance
Rk = gps_std**2 * np.eye(2)             # measurement noise covariance
x_ekf[:, 0] = true_state[:, 0]          # initial state estimate

# State transition function
def motion_model(x, u):
    return np.array([
        x[0] + u[0]*dt * np.cos(x[2]),
        x[1] + u[0]*dt * np.sin(x[2]),
        x[2] + u[1]*dt
    ])

# EKF loop
for k in range(1, N):
    # Prediction
    u = np.array([v_meas[k], omega_meas[k]])
    x_pred = motion_model(x_ekf[:, k-1], u)
    # Jacobian F
    theta = x_ekf[2, k-1]
    F = np.array([
        [1, 0, -u[0]*dt*np.sin(theta)],
        [0, 1,  u[0]*dt*np.cos(theta)],
        [0, 0, 1]
    ])
    P_pred = F @ P @ F.T + Q

    # Update with GPS
    H = np.array([[1, 0, 0], [0, 1, 0]])
    z = gps_meas[:, k]
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + Rk
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_ekf[:, k] = x_pred + K @ y
    P = (np.eye(3) - K @ H) @ P_pred

# Plotting results
plt.figure()
plt.plot(true_state[0], true_state[1], 'k-', linewidth=1.5, label='True trajectory')
plt.plot(gps_meas[0, 1:], gps_meas[1, 1:], 'r.', markersize=4, label='GPS measurements')
plt.plot(x_ekf[0], x_ekf[1], 'b--', linewidth=1.5, label='EKF estimate')
plt.legend()
plt.title('Circular Trajectory: EKF Position Estimation')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.axis('equal')

# Error analysis: position and heading errors
errors_ekf = true_state - x_ekf
theta_ekf_error = wrapToPi(errors_ekf[2, :])

plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(np.arange(N), errors_ekf[0, :], linewidth=1.2)
plt.ylabel(r'$\delta X$ (m)'); plt.grid(True)

plt.subplot(3,1,2)
plt.plot(np.arange(N), errors_ekf[1, :], linewidth=1.2)
plt.ylabel(r'$\delta Y$ (m)'); plt.grid(True)

plt.subplot(3,1,3)
plt.plot(np.arange(N), theta_ekf_error, linewidth=1.2)
plt.ylabel(r'$\delta \theta$ (rad)'); plt.xlabel('Time step'); plt.grid(True)

plt.tight_layout()
plt.show()






























































































