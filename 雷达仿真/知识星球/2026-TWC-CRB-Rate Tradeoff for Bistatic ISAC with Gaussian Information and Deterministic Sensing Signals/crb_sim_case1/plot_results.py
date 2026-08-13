"""
结果可视化模块
=============
画 CRB-Rate tradeoff 曲线。

用法:
    from plot_results import plot_tradeoff, plot_comparison
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_tradeoff(gamma_arr, crb_arr, rate_arr, save_path='tradeoff.png'):
    """Case 1 单条曲线（保留，不改动）"""
    sort_idx = np.argsort(rate_arr)
    rate_sort = np.array(rate_arr)[sort_idx]
    crb_sort  = np.array(crb_arr)[sort_idx]
    gamma_sort = np.array(gamma_arr)[sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rate_sort, crb_sort, 'b-o', markersize=3, linewidth=1.5,
             label='Gaussian signals only (Case 1)')
    ax1.set_xlabel('Communication Rate (bps/Hz)')
    ax1.set_ylabel('CRB for DoA Estimation (rad²)')
    ax1.set_xlim(0, 7)
    ax1.set_title('CRB-Rate Tradeoff')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.plot(10 * np.log10(gamma_sort), crb_sort, 'r-s', markersize=3,
             linewidth=1.5)
    ax2.set_xlabel('SINR Threshold gamma_0 (dB)')
    ax2.set_ylabel('CRB for DoA Estimation (rad²)')
    ax2.set_title('CRB vs SINR Constraint')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show(block=False)
    plt.pause(0.5)


def plot_comparison(data1=None, data2=None, data3=None, save_path='comparison.png'):
    """
    在同一张图上对比 2~3 条 CRB-Rate tradeoff 曲线。

    Args:
        data1: Case 1 — rate, crb, gamma (ISAC with Gaussian signals, 黑色实线)
        data2: Case 2 — rate, crb, gamma (ISAC with both signals, 红色点虚线)
        data3: Case 3 — rate, crb, gamma (ISAC with given realizations, 蓝色点线)
        save_path: 图片保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图: CRB vs Rate ----
    if data1 is not None:
        s = np.argsort(data1['rate'])
        ax1.plot(data1['rate'][s], data1['crb'][s], 'k-', linewidth=1.8,
                 label='ISAC with Gaussian signals')
        ax2.plot(10 * np.log10(data1['gamma'][s]), data1['crb'][s], 'k-', linewidth=1.8,
                 label='ISAC with Gaussian signals')

    if data2 is not None:
        s = np.argsort(data2['rate'])
        ax1.plot(data2['rate'][s], data2['crb'][s], 'r--', linewidth=1.8,
                 label='ISAC with both Gaussian and deterministic signals')
        ax2.plot(10 * np.log10(data2['gamma'][s]), data2['crb'][s], 'r--', linewidth=1.8,
                 label='ISAC with both Gaussian and deterministic signals')

    if data3 is not None:
        s = np.argsort(data3['rate'])
        ax1.plot(data3['rate'][s], data3['crb'][s], 'b:', linewidth=1.8,
                 label='ISAC with given realizations of information signals')
        ax2.plot(10 * np.log10(data3['gamma'][s]), data3['crb'][s], 'b:', linewidth=1.8,
                 label='ISAC with given realizations of information signals')

    ax1.set_xlabel('Communication Rate (bps/Hz)')
    ax1.set_ylabel('CRB for DoA Estimation (rad²)')
    ax1.set_xlim(0, 7)
    ax1.set_title('CRB-Rate Tradeoff')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.set_xlabel('SINR Threshold γ₀ (dB)')
    ax2.set_ylabel('CRB for DoA Estimation (rad²)')
    ax2.set_title('CRB vs SINR Constraint')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show(block=False)
    plt.pause(0.5)
