#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重绘 Mine.json 中的 BER / SER 曲线"""

import json, matplotlib.pyplot as plt, numpy as np

# 读取文件
with open("Mine.json", "r", encoding="utf-8") as f:
    res = json.load(f)

snrs = np.arange(-30, 0, 5)               # 与仿真保持一致

# -------- BER --------
ber_all   = np.array(res["case1_both"]["BER"])
ber_tone  = np.array(res["case2_tone_only"]["BER"])
ber_qpsk  = np.array(res["case3_qpsk_only"]["BER"])

plt.figure()
plt.semilogy(snrs, ber_all,  "o-", label="Total BER")
plt.semilogy(snrs, ber_tone, "x-", label="Delay-only BER")
plt.semilogy(snrs, ber_qpsk, "s-", label="QPSK-only BER")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.grid(True, which="both")
plt.legend()
plt.title("BER vs SNR for Mine.json")

# -------- SER --------
ser_tone1 = np.array(res["case1_both"]["SER_tone"])
ser_qpsk1 = np.array(res["case1_both"]["SER_qpsk"])
ser_all1 = np.array(res["case1_both"]["SER"]) 
ser_tone2 = np.array(res["case2_tone_only"]["SER_tone"])
ser_qpsk3 = np.array(res["case3_qpsk_only"]["SER_qpsk"])

plt.figure()
plt.semilogy(snrs, ser_all1,  "o-", label="Total SER")
plt.semilogy(snrs, ser_tone2, "x-", label="Delay-only SER")
plt.semilogy(snrs, ser_qpsk3, "s-", label="QPSK-only SER")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate")
plt.grid(True, which="both")
plt.legend()
plt.title("SER vs SNR for Mine.json")
plt.show()
