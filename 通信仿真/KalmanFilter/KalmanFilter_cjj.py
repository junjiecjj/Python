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
    predicted_state = A @ estimated_states[t-1]  # 预测下一个状态(当前时刻的状态预测): hat{x}_{t|t-1} = A @ hat{x}_{t-1|t-1}
    P = A @ P @ A.T + Q                          # 预测下一个协方差(更新预测误差协方差): P_{t|t-1} = A@P_{t-1|t-1}@A^T + Q
    ###>>>>>>>>>>>>>>>> 更新阶段
    y_tilde = measurements[t] - H @ predicted_state   # z_t - H @ hat{x}_{t|t-1}
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)      # 计算卡尔曼增益: K_t = P_{t|t-1} @ H^T @ (H @ P_{t|t-1} @ H^T + R)^{-1}
    ### 结合观测值修正预测
    estimated_states[t] = predicted_state + K @ y_tilde  # 更新状态估计: hat{x}_{t-1|t-1} = hat{x}_{t|t-1} + K_t @ (z_t - H @ hat{x}_{t|t-1})
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

# https://github.com/loveuav/Kalman-and-Bayesian-Filters-in-Python

# https://github.com/dougszumski/KalmanFilter/tree/master

# https://github.com/zziz/kalman-filter

# https://github.com/rlabbe/filterpy

# https://github.com/milsto/robust-kalman/tree/master






























































