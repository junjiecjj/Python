#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 20:51:50 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247484144&idx=1&sn=73025dd1696c6c54ab3c3bf741627505&chksm=97406f74657a912e5c340c1eb9f98c2d8d288d4c950e122d0ee14cce3ce123b8cce066e5e210&mpshare=1&scene=1&srcid=10112mG2UYzYDC0CgVMwQhiU&sharer_shareinfo=dc6b55b470847bc323d5e5be5241530e&sharer_shareinfo_first=dc6b55b470847bc323d5e5be5241530e&exportkey=n_ChQIAhIQPBFrhpT%2BsF5Bj6aK2pWsCRKfAgIE97dBBAEAAAAAAAjCDoajuHcAAAAOpnltbLcz9gKNyK89dVj0N8zepvrY07g9aLopftOb%2B88NUJI50P6VYGKJsHi%2B4r6pBKdDFrKXVMhpEXR7jn43lhGnV5axArLi6%2FPZxvkqamV%2BMUb0HLzeUHP7oiuVnjRBUFFnQL6HS321DSIQ3GDfPIuwCRyG6DTfecOEHksy5PmhDoM0G5sVATSAvhyAMkIa%2BmEHzW5%2B%2BxYw0kNhGgxtlpkdwtBLXyrwELNEfmedhDCrCBgHPsCgOx3NLMAdjaNG0M4FiWUQMFhkFZGqchwrfeWD8etMYtT4Zb5QdhVn9XdmStKzvsv3xlQKA14zSufFxAyMnDumIZH%2B795FLbNr3CoySFTu8Va%2F&acctmode=0&pass_ticket=yN8HbJj2WhwsIeDVtFmuiOcFUwgHzJsgBZXI%2FwxirKYOAFt8xb5e8n8yawg00di6&wx_header=0#rd


"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)



def generate_RD_map(N_range, N_doppler, clutter_power):
    """生成距离-多普勒图，包含瑞利分布杂波"""
    real_part = np.random.randn(N_range, N_doppler)
    imag_part = np.random.randn(N_range, N_doppler)
    RD_map = np.sqrt(clutter_power/2) * (real_part + 1j * imag_part)
    return RD_map

def get_reference_cells(data, center_i, center_j, guard_r, guard_d, ref_r, ref_d):
    """获取参考单元（排除保护单元）"""
    ref_cells = []

    for i in range(center_i - ref_r - guard_r, center_i + ref_r + guard_r + 1):
        for j in range(center_j - ref_d - guard_d, center_j + ref_d + guard_d + 1):
            # 跳过保护单元
            if abs(i - center_i) <= guard_r and abs(j - center_j) <= guard_d:
                continue
            # 跳过中心单元本身
            if i == center_i and j == center_j:
                continue
            # 确保在边界内
            if i >= 0 and i < data.shape[0] and j >= 0 and j < data.shape[1]:
                ref_cells.append(data[i, j])

    return np.array(ref_cells)

def ml_cfar_detector_2D(data, guard_r, guard_d, ref_r, ref_d, P_fa):
    """2D ML-CFAR检测器（基于瑞利分布假设）"""
    N_range, N_doppler = data.shape
    detection_map = np.zeros((N_range, N_doppler), dtype=np.int32)  # 明确指定数据类型

    # 计算阈值系数（瑞利分布）
    alpha = np.sqrt(-2 * np.log(P_fa))

    for i in range(guard_r + ref_r, N_range - guard_r - ref_r):
        for j in range(guard_d + ref_d, N_doppler - guard_d - ref_d):
            # 提取参考单元
            ref_cells = get_reference_cells(data, i, j, guard_r, guard_d, ref_r, ref_d)

            if len(ref_cells) > 0:
                # ML估计：瑞利分布的尺度参数
                sigma_ml = np.sqrt(np.sum(ref_cells**2) / (2 * len(ref_cells)))

                # 计算检测阈值
                threshold = alpha * sigma_ml

                # 检测判决
                if data[i, j] > threshold:
                    detection_map[i, j] = 1

    return detection_map

def ca_cfar_detector_2D(data, guard_r, guard_d, ref_r, ref_d, P_fa):
    """2D CA-CFAR检测器用于比较"""
    N_range, N_doppler = data.shape
    detection_map = np.zeros((N_range, N_doppler), dtype=np.int32)  # 明确指定数据类型

    # 计算阈值系数（基于瑞利分布假设）
    N_ref_total = (2*ref_r + 1)*(2*ref_d + 1) - (2*guard_r + 1)*(2*guard_d + 1)
    alpha = N_ref_total * (P_fa**(-1/N_ref_total) - 1)

    for i in range(guard_r + ref_r, N_range - guard_r - ref_r):
        for j in range(guard_d + ref_d, N_doppler - guard_d - ref_d):
            # 提取参考单元
            ref_cells = get_reference_cells(data, i, j, guard_r, guard_d, ref_r, ref_d)

            if len(ref_cells) > 0:
                # CA-CFAR：平均功率估计
                Z = np.mean(ref_cells)

                # 计算检测阈值
                threshold = alpha * Z

                # 检测判决
                if data[i, j] > threshold:
                    detection_map[i, j] = 1

    return detection_map

def calculate_ml_threshold_line(data, range_idx, guard_r, guard_d, ref_r, ref_d, P_fa):
    """计算ML-CFAR在特定距离单元的阈值线"""
    N_doppler = data.shape[1]
    threshold_line = np.zeros(N_doppler)
    alpha = np.sqrt(-2 * np.log(P_fa))

    for j in range(guard_d + ref_d, N_doppler - guard_d - ref_d):
        ref_cells = get_reference_cells(data, range_idx, j, guard_r, guard_d, ref_r, ref_d)
        if len(ref_cells) > 0:
            sigma_ml = np.sqrt(np.sum(ref_cells**2) / (2 * len(ref_cells)))
            threshold_line[j] = alpha * sigma_ml

    return threshold_line

def calculate_ca_threshold_line(data, range_idx, guard_r, guard_d, ref_r, ref_d, P_fa):
    """计算CA-CFAR在特定距离单元的阈值线"""
    N_doppler = data.shape[1]
    threshold_line = np.zeros(N_doppler)
    N_ref_total = (2*ref_r + 1)*(2*ref_d + 1) - (2*guard_r + 1)*(2*guard_d + 1)
    alpha = N_ref_total * (P_fa**(-1/N_ref_total) - 1)

    for j in range(guard_d + ref_d, N_doppler - guard_d - ref_d):
        ref_cells = get_reference_cells(data, range_idx, j, guard_r, guard_d, ref_r, ref_d)
        if len(ref_cells) > 0:
            Z = np.mean(ref_cells)
            threshold_line[j] = alpha * Z

    return threshold_line

def calculate_detection_stats(ml_detection, ca_detection, target_range, target_doppler, N_doppler):
    """计算检测性能统计"""
    # 确保检测图是整数类型
    ml_detection = ml_detection.astype(np.int32)
    ca_detection = ca_detection.astype(np.int32)

    # 真实目标位置
    true_targets = np.zeros_like(ml_detection, dtype=np.int32)
    for i in range(len(target_range)):
        doppler_idx = target_doppler[i] + N_doppler//2
        if 0 <= doppler_idx < N_doppler:
            true_targets[target_range[i], doppler_idx] = 1

    # ML-CFAR统计
    ml_detected = np.sum(ml_detection & true_targets)
    ml_total_targets = np.sum(true_targets)
    ml_detection_rate = ml_detected / ml_total_targets if ml_total_targets > 0 else 0
    ml_false_alarms = np.sum(ml_detection) - ml_detected

    # CA-CFAR统计
    ca_detected = np.sum(ca_detection & true_targets)
    ca_total_targets = np.sum(true_targets)
    ca_detection_rate = ca_detected / ca_total_targets if ca_total_targets > 0 else 0
    ca_false_alarms = np.sum(ca_detection) - ca_detected

    ml_stats = {'detection_rate': ml_detection_rate, 'false_alarms': ml_false_alarms}
    ca_stats = {'detection_rate': ca_detection_rate, 'false_alarms': ca_false_alarms}

    return ml_stats, ca_stats

def main():
    """主函数 - ML-CFAR检测仿真：低速小目标检测"""
    # 参数设置
    N_range = 256      # 距离单元数
    N_doppler = 64     # 多普勒单元数
    P_fa_desired = 1e-4  # 期望虚警概率

    # 目标参数
    target_range = [80, 150, 200]      # 目标距离单元
    target_doppler = [2, 3, -2]        # 目标多普勒单元（低速）
    target_amplitude = [3.5, 3.0, 2.8] # 目标幅度（小目标）

    # CFAR参数
    N_guard_range = 4      # 距离维保护单元
    N_guard_doppler = 8    # 多普勒维保护单元
    N_ref_range = 32       # 距离维参考单元
    N_ref_doppler = 16     # 多普勒维参考单元

    # 杂波参数
    clutter_power = 1.0    # 杂波功率
    SNR_dB = 10            # 信噪比(dB)

    print('生成距离-多普勒图...')
    RD_map = generate_RD_map(N_range, N_doppler, clutter_power)

    # 添加目标
    for i in range(len(target_range)):
        range_idx = target_range[i]
        doppler_idx = target_doppler[i] + N_doppler//2  # 转换为正索引
        amplitude = target_amplitude[i]

        # 确保索引在范围内
        if 0 <= range_idx < N_range and 0 <= doppler_idx < N_doppler:
            RD_map[range_idx, doppler_idx] += amplitude

    # 添加噪声
    noise_power = clutter_power / (10**(SNR_dB/10))
    noise_real = np.random.randn(N_range, N_doppler) * np.sqrt(noise_power/2)
    noise_imag = np.random.randn(N_range, N_doppler) * np.sqrt(noise_power/2)
    RD_map = RD_map + noise_real + 1j * noise_imag

    # 取幅度
    RD_map_mag = np.abs(RD_map)

    print('执行ML-CFAR检测...')
    detection_map_ml = ml_cfar_detector_2D(RD_map_mag, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired)

    print('执行CA-CFAR检测用于比较...')
    detection_map_ca = ca_cfar_detector_2D(RD_map_mag, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired)

    # 结果显示
    plt.figure(figsize=(16, 10))

    # 原始距离-多普勒图
    plt.subplot(2, 3, 1)
    plt.imshow(20*np.log10(RD_map_mag), aspect='auto', extent=[1, N_doppler, N_range, 1], cmap='hsv')
    plt.colorbar()
    plt.title('距离-多普勒图 (dB)')
    plt.xlabel('多普勒单元')
    plt.ylabel('距离单元')
    for i in range(len(target_range)):
        doppler_idx = target_doppler[i] + N_doppler//2 + 1
        plt.plot(doppler_idx, target_range[i], 'ro', markersize=10, linewidth=2)

    # ML-CFAR检测结果
    plt.subplot(2, 3, 2)
    plt.imshow(detection_map_ml, aspect='auto',
               extent=[1, N_doppler, N_range, 1], cmap='hot')
    plt.colorbar()
    plt.title('ML-CFAR检测结果')
    plt.xlabel('多普勒单元')
    plt.ylabel('距离单元')
    for i in range(len(target_range)):
        doppler_idx = target_doppler[i] + N_doppler//2 + 1
        plt.plot(doppler_idx, target_range[i], 'ro', markersize=10, linewidth=2)

    # CA-CFAR检测结果
    plt.subplot(2, 3, 3)
    plt.imshow(detection_map_ca, aspect='auto',
               extent=[1, N_doppler, N_range, 1], cmap='hot')
    plt.colorbar()
    plt.title('CA-CFAR检测结果')
    plt.xlabel('多普勒单元')
    plt.ylabel('距离单元')
    for i in range(len(target_range)):
        doppler_idx = target_doppler[i] + N_doppler//2 + 1
        plt.plot(doppler_idx, target_range[i], 'ro', markersize=10, linewidth=2)

    # 距离维切片比较
    range_slice = target_range[0]
    plt.subplot(2, 3, 4)
    plt.plot(range(1, N_doppler+1), RD_map_mag[range_slice, :], 'b-', linewidth=2, label='信号')

    ml_threshold = calculate_ml_threshold_line(RD_map_mag, range_slice, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired)
    ca_threshold = calculate_ca_threshold_line(RD_map_mag, range_slice, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired)

    plt.plot(range(1, N_doppler+1), ml_threshold, 'r--', linewidth=2, label='ML-CFAR阈值')
    plt.plot(range(1, N_doppler+1), ca_threshold, 'g--', linewidth=2, label='CA-CFAR阈值')
    plt.title(f'距离单元 {range_slice} 的检测情况')
    plt.xlabel('多普勒单元')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)

    # 性能统计
    plt.subplot(2, 3, 5)
    ml_stats, ca_stats = calculate_detection_stats(detection_map_ml, detection_map_ca, target_range, target_doppler, N_doppler)

    performance_data = np.array([
        [ml_stats['detection_rate'], ca_stats['detection_rate']],
        [ml_stats['false_alarms'], ca_stats['false_alarms']]
    ])

    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, performance_data[0, :], width, label='ML-CFAR')
    plt.bar(x + width/2, performance_data[1, :], width, label='CA-CFAR')
    plt.xticks(x, ['检测率', '虚警数'])
    plt.ylabel('性能指标')
    plt.title('ML-CFAR vs CA-CFAR 性能比较')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 低速目标区域放大
    plt.subplot(2, 3, 6)
    low_speed_doppler_start = N_doppler//2 - 5
    low_speed_doppler_end = N_doppler//2 + 5
    low_speed_doppler_range = range(low_speed_doppler_start, low_speed_doppler_end + 1)

    plt.imshow(RD_map_mag[:, low_speed_doppler_start:low_speed_doppler_end+1], aspect='auto', extent=[low_speed_doppler_start+1, low_speed_doppler_end+1, N_range, 1], cmap='hsv')
    plt.colorbar()
    plt.title('低速目标区域放大')
    plt.xlabel('多普勒单元')
    plt.ylabel('距离单元')
    for i in range(len(target_range)):
        doppler_idx = target_doppler[i] + N_doppler//2
        if low_speed_doppler_start <= doppler_idx <= low_speed_doppler_end:
            plt.plot(doppler_idx + 1, target_range[i], 'ro', markersize=10, linewidth=2)

    plt.tight_layout()
    plt.show()

    # 输出性能统计
    print('\n=== 检测性能统计 ===')
    print(f"ML-CFAR: 检测率 = {ml_stats['detection_rate']*100:.2f}%, 虚警数 = {ml_stats['false_alarms']}")
    print(f"CA-CFAR: 检测率 = {ca_stats['detection_rate']*100:.2f}%, 虚警数 = {ca_stats['false_alarms']}")

if __name__ == "__main__":
    main()
