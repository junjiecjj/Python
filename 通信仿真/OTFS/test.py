import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import scipy
import commpy
from Modulations import modulator

def Qfun(x):
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

def ser_awgn(EbN0dB, MOD_TYPE, M, COHERENCE = None):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = Qfun(np.sqrt(2 * EbN0))
    elif MOD_TYPE == "psk":
        if M == 2:
            SER = Qfun(np.sqrt(2 * EbN0))
        else:
            if M == 4:
                SER = 2 * Qfun(np.sqrt(2* EbN0)) - Qfun(np.sqrt(2 * EbN0))**2
            else:
                SER = 2 * Qfun(np.sin(np.pi/M) * np.sqrt(2 * EsN0))
    elif MOD_TYPE.lower() == "qam":
        SER = 1 - (1 - 2*(1 - 1/np.sqrt(M)) * Qfun(np.sqrt(3 * EsN0/(M - 1))))**2
    elif MOD_TYPE.lower() == "pam":
        SER = 2*(1-1/M) * Qfun(np.sqrt(6*EsN0/(M**2-1)))
    return SER

def ISFFT(X):
    """
    Inverse Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after ISFFT
    """
    M, N = X.shape
    # ISFFT: DFT along rows (delay domain) and IDFT along columns (Doppler domain)
    X_out = np.fft.ifft(np.fft.fft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(N / M)
    return X_out

def SFFT(X):
    """
    Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after SFFT
    """
    M, N = X.shape
    # SFFT: IDFT along rows (delay domain) and DFT along columns (Doppler domain)
    X_out = np.fft.fft(np.fft.ifft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(M / N)
    return X_out

def Heisenberg(M, N, X_tf):
    # 海森堡变换: TF域 → 时域
    s = np.zeros(M*N, dtype=complex)
    for n in range(N):
        for m in range(M):
            s[n*M + m] = X_tf[m, n] * np.exp(1j*2*np.pi*m*n/M)
    return s

def Wigner(M, N, r):
    # 维格纳变换: 时域 → TF域
    Y_tf = np.zeros((M, N), dtype=complex)
    for n in range(N):
        for m in range(M):
            Y_tf[m, n] = r[n*M + m] * np.exp(-1j*2*np.pi*m*n/M)
    return Y_tf

def otfs_simulation(M=32, N=16, EbN0dB=np.arange(0, 17, 2), N_frames=2000):
    """OTFS系统仿真 (修正功率和噪声计算)"""
    SER_sim = np.zeros_like(EbN0dB, dtype=float)
    # 创建QAM调制器
    QAM_mod = 4
    bps = int(np.log2(QAM_mod))

    EsN0dB = 10 * np.log10(bps) + EbN0dB
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()

    for idx, snr in enumerate(EsN0dB):
        print(f"{idx+1}/{EsN0dB.size}")
        sigma2 = 10 ** (-snr / 10)  # 噪声方差
        errors = 0
        total_symbols = 0

        for _ in range(N_frames):
            # === 发射端 ===
            bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
            X_dd = modem.modulate(bits)
            d = np.array([demap_table[sym] for sym in X_dd])
            X_dd = X_dd.reshape(M, N)

            # OTFS调制
            X_tf = ISFFT(X_dd)
            # s = np.fft.ifft(X_tf, axis=0).flatten()
            ## or
            s = Heisenberg(M, N, X_tf)
            # === 信道 ===
            # 多普勒信道 (单径)
            nu = 0.05  # 归一化多普勒
            dop = np.exp(1j * 2 * np.pi * nu * np.arange(len(s)) / len(s))
            r = s * dop  # 忽略时延，仅测试多普勒

            r = s
            # 添加噪声 (修正噪声功率)
            noise_power = np.mean(np.abs(r)**2) * sigma2
            noise = np.sqrt(noise_power/2) * (np.random.randn(*r.shape) + 1j*np.random.randn(*r.shape))
            r += noise

            # === 接收端 ===
            # Y_tf = np.fft.fft(r.reshape(M, N), axis=0)
            ## or
            Y_tf = Wigner(M, N, r)
            Y_dd = SFFT(Y_tf)

            # === 解调与SER计算 ===
            # QPSK解调 (相位判决)
            uu_hat = modem.demodulate(Y_dd.flatten(), 'hard')
            d_hat = []
            for j in range(M*N):
                d_hat.append( int(''.join([str(num) for num in uu_hat[j*bps:(j+1)*bps]]), base = 2) )
            d_hat = np.array(d_hat)

            errors += np.sum(d != d_hat)
            total_symbols += d.size

        SER_sim[idx] = errors / total_symbols

    # 理论QPSK SER
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, QAM_mod)

    return  SER_sim, SER_theory

# 运行仿真
EbN0dB = np.arange(0, 12, 2)
SER_sim, SER_theory = otfs_simulation(EbN0dB=EbN0dB, N_frames=2000)

# 绘图
plt.figure()
plt.semilogy(EbN0dB, SER_sim, 'bo-', label='OTFS (仿真)')
plt.semilogy(EbN0dB, SER_theory, 'r--', label='QPSK (理论)')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.grid(True)
plt.legend()
plt.title('OTFS-QPSK 符号错误率性能')
plt.show()
