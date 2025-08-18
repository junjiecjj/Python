# https://github.com/eric-hs-rou/doubly-dispersive-channel-simulation
# https://github.com/jerryzhang124/AFDM-simple-simulation

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import erf
from Modulations import modulator

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

def AFDM_mod(x, c1, c2):
    """
    AFDM调制函数 - Python版本
    参数:
        x : 输入信号 (Nx1 列向量)
        c1 : 第一个调频参数
        c2 : 第二个调频参数
    返回:
        out : 调制输出信号
    """
    N = x.shape[0]

    # 创建DFT矩阵并归一化
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)

    # 创建L1和L2对角矩阵
    n = np.arange(N)
    L1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * (n**2)))
    L2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * (n**2)))

    # 构建AFDM矩阵
    A = L2 @ F @ L1
    # 计算调制输出 (注意MATLAB的'是共轭转置，Python用.conj().T)
    out = A.conj().T @ x
    return out

def AFDMdemod(S, c1, c2):
    """
    AFDM解调函数 - Python版本
    参数:
        S : 输入信号 (N x T 矩阵，N为子载波数，T为符号数)
        c1 : 第一个调频参数
        c2 : 第二个调频参数
    返回:
        X : 解调输出信号 (N x T 矩阵)
    """
    N = S.shape[0]  # 获取子载波数

    # 创建DFT矩阵并归一化
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)

    # 创建L1和L2对角矩阵
    n = np.arange(N)
    L1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * (n**2)))
    L2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * (n**2)))

    # 构建AFDM解调矩阵
    A = L2 @ F @ L1

    # 解调计算
    X = A @ S
    return X

def Gen_channel_mtx(N, taps, chan_coef, delay_taps, Doppler_freq, c1):
    """
    生成AFDM信道矩阵 - Python版本

    参数:
        N : 矩阵大小/子载波数
        taps : 多径数
        chan_coef : 信道系数数组 (长度=taps)
        delay_taps : 时延抽头数组 (长度=taps)
        Doppler_freq : 多普勒频率数组 (长度=taps)
        c1 : AFDM调频参数

    返回:
        H : 生成的信道矩阵 (N x N)
    """
    # 创建Pi矩阵 (方程25)
    Pi_row = np.zeros(N)
    Pi_row[-1] = 1
    Pi = toeplitz(np.concatenate(([Pi_row[0]], np.flip(Pi_row[1:]))), Pi_row)

    H = np.zeros((N, N), dtype=complex)

    for i in range(taps):
        h_i = chan_coef[i]
        l_i = delay_taps[i]
        f_i = Doppler_freq[i]

        # 创建D_i矩阵 (多普勒)
        D_i = np.diag(np.exp(-1j * 2 * np.pi * f_i * np.arange(N)))

        # 创建G_i矩阵 (方程26)
        temp = np.ones(N, dtype=complex)
        for n in range(N):
            if n < l_i:
                temp[n] = np.exp(-1j * 2 * np.pi * c1 * (N**2 - 2 * N * (l_i - n)))

        G_i = np.diag(temp)

        # 计算Pi^l_i (矩阵幂)
        Pi_pow = np.linalg.matrix_power(Pi, l_i)

        # 累加到信道矩阵
        H += h_i * G_i @ D_i @ Pi_pow

    return H


def main():
    np.random.seed(1)  # 固定随机种子

    # 系统参数
    M_mod = 4          # QAM调制阶数
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, M_mod)
    map_table, demap_table = modem.getMappTable()
    N = 64             # 子载波数
    car_fre = 4e9      # 载波频率 (Hz)
    delta_f = 15e3     # 子载波间隔 (Hz)
    T = 1/delta_f      # 符号持续时间 (s)

    eng_sqrt = 1 if M_mod == 2 else np.sqrt((M_mod-1)/6*4)  # 符号平均功率
    SNR_dB = np.arange(0, 17, 2)  # SNR范围 (dB)
    SNR = 10**(SNR_dB/10)
    sigma_2 = (abs(eng_sqrt)**2)/SNR  # 噪声功率

    N_frame = 10000    # 仿真帧数

    # 生成合成时延-多普勒信道
    taps = 9           # 多径数
    l_max = 3          # 最大归一化时延索引
    k_max = 4          # 最大归一化多普勒索引

    # 生成复高斯信道系数 (瑞利分布)
    chan_coef = 1/np.sqrt(2)*(np.random.randn(taps) + 1j*np.random.randn(taps))
    delay_taps = np.random.randint(0, l_max+1, taps)
    delay_taps = np.sort(delay_taps - np.min(delay_taps))  # 时延抽头 [0, l_max-1]
    Doppler_taps = k_max - 2*k_max*np.random.rand(taps)    # 多普勒抽头 [-k_max, k_max]
    Doppler_freq = Doppler_taps/(N*T)                      # 实际多普勒频率 (Hz)

    # AFDM参数计算
    max_Doppler = np.max(Doppler_taps)
    max_delay = np.max(delay_taps)

    CPP_len = max_delay              # CPP长度 >= l_max-1
    N_data = N - CPP_len             # 数据符号长度

    k_v = 1  # 保护间隔对抗分数多普勒
    if (2*(max_Doppler+k_v)*(max_delay+1)+max_delay) > N_data:
        raise ValueError('子载波正交性不满足条件')

    c1 = (2*(max_Doppler+k_v)+1)/(2*N_data)  # 调频参数c1
    c2 = 1/(N_data**2)                       # 调频参数c2

    # 生成信道矩阵 (离散时间)
    L_set = np.unique(delay_taps)
    gs = np.zeros((max_delay+1, N), dtype=complex)

    for q in range(N):
        for i in range(taps):
            g_i = chan_coef[i]
            l_i = delay_taps[i]
            f_i = Doppler_freq[i]
            gs[l_i, q] += g_i * np.exp(-1j*2*np.pi*f_i*q)
    # 生成完整信道矩阵
    H = Gen_channel_mtx(N, taps, chan_coef, delay_taps, Doppler_freq, c1)
    # 主仿真循环
    ber = np.zeros(len(SNR_dB))
    for iesn0 in range(len(SNR_dB)):
        err_count = 0

        for iframe in range(N_frame):
            # 发射端
            uu = np.random.randint(0, 2, size = N_data * bps).astype(np.int8)
            x_qam = modem.modulate(uu)
            x = np.array([demap_table[sym] for sym in x_qam])
            # x = np.random.randint(0, M_mod, N_data)  # 生成随机数据
            # x_qam = qammod(x, M_mod)                 # QAM调制
            s = AFDM_mod(x_qam, c1, c2)              # AFDM调制

            # 生成CPP
            cpp_indices = np.arange(N_data-CPP_len, N_data)
            cpp = s[cpp_indices] * np.exp(-1j*2*np.pi*c1*(N**2 + 2*N*(-CPP_len + np.arange(0, CPP_len))))
            s_cpp = np.concatenate([cpp, s])         # 添加CPP

            # 信道传输
            r = np.zeros(N, dtype=complex)
            for q in range(len(s_cpp)):
                for l in L_set:
                    if q >= l:
                        r[q] += gs[l, q] * s_cpp[q-l]

            # 添加噪声
            w = np.sqrt(sigma_2[iesn0]/2) * (np.random.randn(len(s_cpp)) + 1j*np.random.randn(len(s_cpp)))
            r += w

            # 接收端处理
            # MMSE均衡 (理想信道估计)
            x_est = H.conj().T @ np.linalg.inv(H @ H.conj().T + sigma_2[iesn0]*np.eye(N)) @ r
            x_est_no_cpp = x_est[CPP_len:]           # 去除CPP
            y = AFDMdemod(x_est_no_cpp, c1, c2)       # AFDM解调
            # x_est_bit = qamdemod(y, M_mod)            # QAM解调

            sCap = modem.demodulate(y, 'hard')
            x_est_bit = []
            for j in range(N_data):
                x_est_bit.append( int(''.join([str(num) for num in sCap[j*bps:(j+1)*bps]]), base = 2) )
            x_est_bit = np.array(x_est_bit)

            # 错误计数
            err_count += np.sum(x_est_bit != x)

        ber[iesn0] = err_count / (N_data * N_frame)  # 计算BER

    print("BER结果:", ber)

    # 绘制BER曲线
    plt.figure(figsize=(10, 10))
    plt.semilogy(SNR_dB, ber, 'b-o', linewidth=2, markersize=8)
    plt.legend(['MMSE均衡'])
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('AFDM系统误码率性能')
    plt.grid(True, which="both", ls="-")
    plt.show()

# 需要确保以下函数已在之前定义或导入:
# AFDM_mod(), AFDMdemod(), Gen_channel_mtx(), qammod(), qamdemod()

if __name__ == "__main__":
    main()














