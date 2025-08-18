#  https://github.com/bb16177/OTFS-Simulation
#  https://github.com/ironman1996/OTFS-simple-simulation
# https://github.com/eric-hs-rou/doubly-dispersive-channel-simulation


# https://zhuanlan.zhihu.com/p/608867803

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, ifft2
from scipy.special import erfc
from Modulations import modulator


def otfs_modulation(data, M, N):
    """OTFS调制：DD域 → TF域 → 时域 (修正归一化)"""
    # ISFFT: (M,N) DD域 → (M,N) TF域
    X_tf = ifft(fft(data, axis=0), axis=1) / np.sqrt(N)

    # 海森堡变换: TF域 → 时域
    s = np.zeros(M*N, dtype=complex)
    for n in range(N):
        for m in range(M):
            s[n*M + m] = X_tf[m, n] * np.exp(1j*2*np.pi*m*n/M)
    return s

def otfs_demodulation(r, M, N):
    """OTFS解调：时域 → TF域 → DD域 (修正归一化)"""
    # 维格纳变换: 时域 → TF域
    Y_tf = np.zeros((M, N), dtype=complex)
    for n in range(N):
        for m in range(M):
            Y_tf[m, n] = r[n*M + m] * np.exp(-1j*2*np.pi*m*n/M)

    # SFFT: TF域 → DD域
    y_dd = fft(ifft(Y_tf, axis=0), axis=1) * np.sqrt(N)
    return y_dd

def qpsk_theoretical_ser(snr_db):
    """理论QPSK的SER公式"""
    snr_linear = 10**(snr_db / 10)
    return erfc(np.sqrt(snr_linear)) - 0.25 * erfc(np.sqrt(snr_linear))**2

def simulate_otfs_awgn(M=32, N=8, snr_db_range=np.arange(0, 21, 2)):
    """OTFS在AWGN信道下的SER仿真（精确对比QPSK理论值）"""
    ser_otfs, ser_qpsk = [], []
    mod_order = 4  # QPSK
    MOD_TYPE = "psk"
    modem, Es, bps = modulator(MOD_TYPE, mod_order)
    map_table, demap_table = modem.getMappTable()
    for snr_db in snr_db_range:
        errors_otfs = 0
        for _ in range(1000):  # 蒙特卡洛仿真
            # 生成QPSK符号 (Gray编码)
            # 发射端
            uu = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
            qam_symbols = modem.modulate(uu)
            data = np.array([demap_table[sym] for sym in qam_symbols]).reshape(M,N)
            qam_symbols = qam_symbols.reshape(M,N)
            # data = np.random.randint(0, mod_order, (M, N))
            # qam_symbols = np.exp(1j*(np.pi/4 + 2*np.pi*data/mod_order))

            # OTFS调制
            tx_signal = otfs_modulation(qam_symbols, M, N)

            # AWGN信道
            noise_var = 10**(-snr_db/10)
            noise = np.sqrt(noise_var/2) * (np.random.randn(*tx_signal.shape) + 1j*np.random.randn(*tx_signal.shape))
            rx_signal = tx_signal + noise

            # OTFS解调
            rx_symbols = otfs_demodulation(rx_signal, M, N).flatten()

            # QPSK解调 (相位判决)
            sCap = modem.demodulate(rx_symbols, 'hard')
            rx_data = []
            for j in range(M*N):
                rx_data.append( int(''.join([str(num) for num in sCap[j*bps:(j+1)*bps]]), base = 2) )
            rx_data = np.array(rx_data).reshape(M,N)

            # rx_phase = (np.angle(rx_symbols) + np.pi) % (2*np.pi)
            # rx_data = np.floor(rx_phase * mod_order / (2*np.pi)).astype(int)
            errors_otfs += np.sum(rx_data != data)

        ser_otfs.append(errors_otfs / (M*N*1000))
        ser_qpsk.append(qpsk_theoretical_ser(snr_db))

    return ser_otfs, ser_qpsk

def plot_ser_comparison():
    """绘制OTFS与理论QPSK的SER对比曲线"""
    M, N = 32, 8
    snr_db_range = np.arange(0, 16, 2)
    ser_otfs, ser_qpsk = simulate_otfs_awgn(M, N, snr_db_range)

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_range, ser_otfs, 'bo-', label='OTFS (AWGN)')
    plt.semilogy(snr_db_range, ser_qpsk, 'r--', linewidth=2, label='Theoretical QPSK')
    plt.grid(True, which="both", ls="--")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Symbol Error Rate (SER)")
    plt.title("OTFS vs QPSK SER Performance in AWGN Channel")
    plt.legend()
    plt.show()

plot_ser_comparison()



#%%

















