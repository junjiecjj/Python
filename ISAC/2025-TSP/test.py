import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.special import erfc

def otfs_modulation(data, M, N):
    """完全修正的OTFS调制"""
    # 能量归一化的ISFFT
    X_tf = ifft(fft(data, axis=0), axis=1) / np.sqrt(N)

    # 精确相位补偿的海森堡变换
    s = np.zeros(M*N, dtype=complex)
    for n in range(N):
        for m in range(M):
            s[n*M + m] = X_tf[m, n] * np.exp(1j*2*np.pi*(m*n)/(M*N))
    return s

def otfs_demodulation(r, M, N):
    """完全修正的OTFS解调"""
    # 精确相位补偿的维格纳变换
    Y_tf = np.zeros((M, N), dtype=complex)
    for n in range(N):
        for m in range(M):
            Y_tf[m, n] = r[n*M + m] * np.exp(-1j*2*np.pi*(m*n)/(M*N))

    # 能量归一化的SFFT
    y_dd = fft(ifft(Y_tf, axis=0), axis=1) * np.sqrt(N)
    return y_dd

def qpsk_theory(snr_db):
    """精确QPSK理论性能"""
    snr = 10**(snr_db/10)
    return erfc(np.sqrt(snr/2))*(1 - 0.25*erfc(np.sqrt(snr/2)))

def run_simulation():
    M, N = 32, 8  # 必须满足M > 最大时延，N > 最大多普勒
    snr_db_range = np.arange(0, 13, 2)
    trials = 1000
    ser_otfs = []

    for snr_db in snr_db_range:
        errors = 0
        for _ in range(trials):
            # 生成标准QPSK符号（能量=1）
            bits = np.random.randint(0, 2, (M, N, 2))
            symbols = (2*bits[:,:,0]-1 + 1j*(2*bits[:,:,1]-1))/np.sqrt(2)

            # OTFS调制
            tx_signal = otfs_modulation(symbols, M, N)

            # 计算实际符号能量
            Es = np.mean(np.abs(tx_signal)**2)

            # 添加精确噪声
            N0 = Es * 10**(-snr_db/10)
            noise = np.sqrt(N0/2)*(np.random.randn(*tx_signal.shape) + 1j*np.random.randn(*tx_signal.shape))
            rx_signal = tx_signal + noise

            # OTFS解调
            rx_dd = otfs_demodulation(rx_signal, M, N)

            # QPSK硬判决
            rx_bits = np.zeros((M,N,2))
            rx_bits[:,:,0] = (np.real(rx_dd) > 0).astype(int)
            rx_bits[:,:,1] = (np.imag(rx_dd) > 0).astype(int)
            errors += np.sum(bits != rx_bits)

        ser_otfs.append(errors/(2*M*N*trials))

    # 理论曲线
    ser_theory = [qpsk_theory(snr) for snr in snr_db_range]

    # 绘图
    plt.figure(figsize=(10,6))
    plt.semilogy(snr_db_range, ser_otfs, 'bo-', label='OTFS Simulation')
    plt.semilogy(snr_db_range, ser_theory, 'r--', label='QPSK Theory')
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    plt.legend()
    plt.title('OTFS Performance Verification (Final Corrected)')
    plt.show()

# 执行仿真
run_simulation()





