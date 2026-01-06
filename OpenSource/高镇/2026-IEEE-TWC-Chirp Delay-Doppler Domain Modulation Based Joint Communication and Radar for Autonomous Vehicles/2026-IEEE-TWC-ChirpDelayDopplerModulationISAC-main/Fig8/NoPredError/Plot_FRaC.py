#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重绘 FRaC.json 中的 BER / SER 曲线（2bit 天线选择版本）"""

import os, json, matplotlib.pyplot as plt, numpy as np

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的文件夹路径
script_dir = os.path.dirname(script_path)
# 构造最终要保存的文件的完整路径
output_path = os.path.join(script_dir, 'FRaC.json')
with open(output_path, "r", encoding="utf-8") as f:
    res = json.load(f)

# 与仿真一致的 SNR 轴
snrs = np.arange(-30, 0, 5)               # [-30, -25, ..., -5]

modes = ["spatial", "qpsk", "all"]
markers = {"spatial": "s-", "qpsk": "^-", "all": "o-"}
marker_perm = "d-"

def _get(bucket, m, s, key):
    """读取 res[m][s][key]；兼容 JSON 中键为字符串的情况"""
    if s in bucket[m]:
        return bucket[m][s][key]
    else:
        return bucket[m][str(s)][key]

# -------- BER --------
plt.figure(figsize=(9,6))
for m in modes:
    if m == "qpsk":
        # QPSK 共 4 bit
        ber_num = np.array([_get(res, m, s, "BER_QPSK") for s in snrs], dtype=float)
        ber_den = 10_000 * 4
        label = "QPSK-bits BER (4 bits)"
    elif m == "spatial":
        # 天线选择现为 2 bit（从 6 种中选 4 种）
        ber_num = np.array([_get(res, m, s, "BER_AntSel") for s in snrs], dtype=float)
        ber_den = 10_000 * 2
        label = "Antenna-sel BER (2 bits)"
    else:  # all
        # 有效比特总数：2（天线选择）+1（swap）+4（两段 QPSK）= 7
        ber_num = np.array([_get(res, m, s, "BER") for s in snrs], dtype=float)
        ber_den = 10_000 * 7
        label = "All-bits BER (7 bits)"
    ber = ber_num / ber_den
    ber[ber == 0] = np.nan
    plt.semilogy(snrs, ber, markers[m], label=label)

# 单独画 swap 位（1 bit），从 spatial 桶中取
ber_perm = np.array([_get(res, "spatial", s, "BER_perm") for s in snrs], dtype=float) / (10_000 * 1)
ber_perm[ber_perm == 0] = np.nan
plt.semilogy(snrs, ber_perm, marker_perm, label="Swap-bit BER (1 bit)")

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.grid(True, which="both")
plt.legend()
plt.title("BER vs SNR (2-bit antenna selection)")

# -------- SER --------
plt.figure(figsize=(9,6))
for m in modes:
    if m == "qpsk":
        ser_num = np.array([_get(res, m, s, "SER_QPSK") for s in snrs], dtype=float)
        label = "QPSK-symbol SER"
    elif m == "spatial":
        ser_num = np.array([_get(res, m, s, "SER_AntSel") for s in snrs], dtype=float)
        label = "Antenna-sel SER"
    else:  # all
        ser_num = np.array([_get(res, m, s, "SER") for s in snrs], dtype=float)
        label = "All-symbol SER"
    ser = ser_num / 10_000.0
    ser[ser == 0] = np.nan
    plt.semilogy(snrs, ser, markers[m], label=label)

ser_perm = np.array([_get(res, "spatial", s, "SER_perm") for s in snrs], dtype=float) / 10_000.0
ser_perm[ser_perm == 0] = np.nan
plt.semilogy(snrs, ser_perm, marker_perm, label="Swap-bit SER")

plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate")
plt.grid(True, which="both")
plt.legend()
plt.title("SER vs SNR (2-bit antenna selection)")
plt.tight_layout()
plt.show()
