import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import math

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

# 参数设置
# 基础参数
c = 3.0e8  # 光速(m/s)
Fc = 35e9  # 雷达射频
Br = 10e6  # 发射信号带宽
fs = 20 * 1e6  # 采样频率
PRF = 2e3  # 脉冲重复频率
PRT = 1 / PRF  # 脉冲重复周期
lamda = c / Fc  # 雷达工作波长，用于计算多普勒频移
N_pulse = 128  # 回波脉冲数
N_sample = round(fs * PRT)  # 每个脉冲周期的采样点数；
Tr = 3 * 1e-6  # 发射信号时宽
t1 = np.arange(0, N_sample) / fs  # 时间序列
RangeMax = c * t1[-1] / 2  # 最大不模糊距离
Range = c * t1 / 2  # 距离序列
Vmax = lamda * PRF / 2  # 最大可检测速度
Velocity = np.linspace(-Vmax/2, Vmax/2 - Vmax/N_pulse, N_pulse)  # 速度序列
searching_doa = np.arange(-15, 15, 0.01)  # 角度搜索区间

# 阵列参数
M = 16  # 阵元数量
SourceNum = 1  # 信号源数量
d = lamda / 2  # 阵元间隔
d_LinearArray = np.arange(M).reshape(-1, 1) * d  # 阵元间距
SNR = 10
SNR = 10 ** (SNR / 10)  # 信杂比 Pt=10w RCS=20m^2

# 场景参数
V = 50
T = 10  # 采样间隔，没有采用PRT是因为：T=PRT时，nT=300001,跑一次代码太久
nT = len(np.arange(-4e3, 4e3 + V * T, V * T))  # 采样帧数

# 目标状态变化
Xk = np.zeros((4, nT))
Xk[0, :] = np.arange(-4e3, 4e3 + V * T, V * T)
Xk[1, :] = 64e3 * np.ones(nT)
Xk[2, :] = V * np.ones(nT)
Xk[3, :] = np.zeros(nT)

Zk = np.zeros((3, nT))
for i in range(nT):
    r = np.sqrt(Xk[0, i] ** 2 + Xk[1, i] ** 2)  # 径向距离
    v = -(Xk[0, i] * Xk[2, i] + Xk[1, i] * Xk[3, i]) / r  # 径向速度
    phi = -(np.arctan2(Xk[1, i], Xk[0, i]) * 180 / np.pi - 90)  # 角度
    Zk[:, i] = [r, v, phi]  # i时刻目标极坐标状态

# 波束鉴角曲线的生成
theta = np.arange(-90, 90, 0.01)
theta1 = -3  # 波形A指向的方向（度）
theta2 = 3  # 波束B指向的方向
theta_min = -3.8
theta_max = 3.8
look_a = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(theta * np.pi / 180) / lamda)  # 导向矢量
w_1 = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(theta1 * np.pi / 180) / lamda)  # 波束A加权权向量
w_2 = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(theta2 * np.pi / 180) / lamda)  # 波束B加权权向量
yA = np.abs(w_1.conj().T @ look_a)  # 波束A的方向图
yB = np.abs(w_2.conj().T @ look_a)  # 波束B的方向图
ABSum = yA + yB  # 和波束的方向图
ABDiff = yA - yB  # 差波束的方向图
AB_ybili = ABDiff / ABSum  # 差和比

# 绘制两波束
plt.figure(1)
plt.plot(theta, yA.flatten() / np.max(yA), linewidth=1)  # 绘制波束A
plt.plot(theta, yB.flatten() / np.max(yB), linewidth=1)  # 绘制波束B
plt.xlabel('方位角/°')
plt.ylabel('归一化方向图')
plt.legend(['波束A', '波束B'])
plt.title('波束A、B示意图')
plt.tight_layout()
plt.grid(True)
plt.show()

# 绘制和差波束
plt.figure(2)
plt.plot(theta, ABSum.flatten(), linewidth=1)  # 绘制和波束
plt.plot(theta, ABDiff.flatten(), linewidth=1)  # 绘制差波束
plt.xlabel('方位角/°')
plt.ylabel('功率增益')
plt.legend(['和波束', '差波束'])
plt.title('和差波束示意图')
plt.tight_layout()
plt.grid(True)
plt.show()

# 绘制鉴角曲线
plt.figure(3)
plt.plot(theta, AB_ybili.flatten())
plt.xlim([theta_min, theta_max])
plt.xlabel('方位角/°')
plt.ylabel('差和比')
plt.title('鉴角曲线')
plt.grid(True)
plt.show()

# 主程序

def rectpuls(t, width):
    """实现MATLAB的rectpuls函数"""
    return np.where((t >= -width/2) & (t <= width/2), 1, 0)

def ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET):
    """CA-CFAR检测算法"""
    numTrain2D = numTrain * numTrain - numGuard * numGuard
    RDM_mask = np.zeros_like(RDM_dB)

    rows, cols = RDM_mask.shape
    for r in range(numTrain + numGuard, rows - (numTrain + numGuard)):
        for d in range(numTrain + numGuard, cols - (numTrain + numGuard)):
            # 计算噪声功率
            Pn = (np.sum(RDM_dB[r - (numTrain + numGuard):r + (numTrain + numGuard) + 1, d - (numTrain + numGuard):d + (numTrain + numGuard) + 1]) -
                  np.sum(RDM_dB[r - numGuard:r + numGuard + 1, d - numGuard:d + numGuard + 1])) / numTrain2D

            a = numTrain2D * (P_fa ** (-1 / numTrain2D) - 1)  # scaling factor of T = α*Pn
            threshold = a * Pn

            if (RDM_dB[r, d] > threshold) and (RDM_dB[r, d] > SNR_OFFSET):
                RDM_mask[r, d] = 1

    # 找到检测到的目标位置
    cfar_ranges, cfar_dopps = np.where(RDM_mask == 1)

    # 去除冗余检测
    rem_range = []
    rem_dopp = []
    for i in range(1, len(cfar_ranges)):
        if (abs(cfar_ranges[i] - cfar_ranges[i-1]) <= 5) and (abs(cfar_dopps[i] - cfar_dopps[i-1]) <= 5):
            rem_range.append(i)
            rem_dopp.append(i)

    cfar_ranges = np.delete(cfar_ranges, rem_range)
    cfar_dopps = np.delete(cfar_dopps, rem_dopp)
    K = len(cfar_dopps)  # 检测到的目标数量

    return RDM_mask, cfar_ranges, cfar_dopps, K

def lookup_angle_from_sumdiffratio(AB_ybili, sum_diff_ratio, theta, theta_min, theta_max):
    """根据和差比查找角度"""
    if len(AB_ybili) != len(theta):
        raise ValueError('AB_ybili 的长度必须与 theta 的长度一致')

    # 限制查找范围
    theta_range_idx = (theta >= theta_min) & (theta <= theta_max)
    theta_limited = theta[theta_range_idx]
    AB_ybili_limited = AB_ybili[theta_range_idx]

    # 计算差值
    diff = np.abs(AB_ybili_limited - sum_diff_ratio)

    # 找到最小差值的位置
    idx = np.argmin(diff)

    # 获取对应的theta值
    theta_closest = theta_limited[idx]

    return theta_closest

def plotTrajectory(Detect_Result):
    """绘制航迹动态显示"""
    distance = Detect_Result[0, :]  # 距离
    angle_xy = Detect_Result[2, :]  # 与y轴夹角
    angle_polar = -1 * angle_xy + 90  # 转换为极坐标下的角度

    # 极坐标系动画
    fig = plt.figure()
    for i in range(len(distance)):
        plt.clf()
        ax = plt.subplot(111, projection='polar')
        ax.plot(angle_polar[i] * np.pi / 180, distance[i], 'bo')
        ax.set_theta_zero_location('N')  # 0度在顶部
        ax.set_theta_direction(-1)  # 角度顺时针增加
        ax.set_ylim(0, np.max(distance))
        plt.title('航迹动态显示')
        plt.draw()
        plt.pause(0.05)

    # 直角坐标系动画
    X = distance * np.sin(angle_xy * np.pi / 180)  # x 坐标
    Y = distance * np.cos(angle_xy * np.pi / 180)  # y 坐标

    fig = plt.figure()
    for i in range(len(distance)):
        plt.clf()
        plt.plot(X[i], Y[i], 'bo')
        plt.grid(True)
        plt.xlabel('水平距离（m）')
        plt.ylabel('垂直距离（m）')
        plt.axis([-4500, 4500, 0, 70000])  # 设置坐标轴范围
        plt.title('航迹动态显示')
        plt.draw()
        plt.pause(0.05)

# 初始化数组
Detect_Result = np.zeros((3, nT))  # 最终测量结果

signal_LFM = np.zeros((M, N_pulse, N_sample), dtype=complex)  # 信号矩阵

signal_i = np.ones((N_pulse, N_sample), dtype=complex)  # 中间累加矩阵变量
y1_out = np.ones((N_pulse, N_sample), dtype=complex)
y2_out = np.ones((N_pulse, N_sample), dtype=complex)

FFT_y1out_all = np.ones((N_pulse, N_sample, nT), dtype=complex)  # 保存每个nT时刻下MTD、CFAR结果
FFT_y2out_all = np.ones((N_pulse, N_sample, nT), dtype=complex)
RDM_mask_A_all = np.ones((N_pulse, N_sample, nT))
RDM_mask_B_all = np.ones((N_pulse, N_sample, nT))

# 匹配滤波系数生成
sr = rectpuls(t1 - Tr/2, Tr) * np.exp(1j * np.pi * (Br/Tr) * (t1 - Tr/2) ** 2)  # LFM发射信号
win = windows.hamming(N_sample)  # 匹配滤波加窗
win2 = np.tile(windows.hamming(N_pulse), (N_sample, 1)).T  # MTD加窗
h_w = np.conj(sr[::-1]) * win
h_w_freq = np.fft.fft(h_w)

# 噪声
clutter = (np.sqrt(2)/2 * np.random.randn(M, N_sample) +
           np.sqrt(2)/2 * 1j * np.random.randn(M, N_sample))  # 噪声

for t in range(nT):
    print(f"{t+1}/{nT}")
    data = Zk[:, t]  # 读取目标真实位置
    a_tar_LinearArray = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(data[2] * np.pi / 180) / lamda)  # 期望信号的导向矢量，线性阵列

    for i_n in range(N_pulse):
        ta = i_n * PRT
        tao = 2 * (data[0] - data[1] * (ta + t1)) / c
        signal_i[i_n, :] = (SNR * rectpuls(t1 - tao - Tr/2, Tr) *
                           np.exp(1j * 2 * np.pi * Fc * (t1 - tao - Tr/2) +
                                  1j * np.pi * (Br/Tr) * (t1 - tao - Tr/2) ** 2))

        signal_LFM[:, i_n, :] = a_tar_LinearArray * signal_i[i_n, :] + clutter
        st = signal_LFM[:, i_n, :]

        y1 = w_1.conj().T @ st  # 波束A回波
        y2 = w_2.conj().T @ st  # 波束B回波

        # 脉冲压缩
        y1_out[i_n, :] = np.fft.ifft(np.fft.fft(y1) * h_w_freq)
        y2_out[i_n, :] = np.fft.ifft(np.fft.fft(y2) * h_w_freq)

    # MTD
    FFT_y1out = np.fft.fftshift(np.fft.fft(y1_out * win2, axis=0), axes=0)
    FFT_y2out = np.fft.fftshift(np.fft.fft(y2_out * win2, axis=0), axes=0)
    FFT_y1out_all[:, :, t] = FFT_y1out
    FFT_y2out_all[:, :, t] = FFT_y2out

    # CA-CFAR
    numGuard = 2  # # of guard cells
    numTrain = numGuard * 2  # # of training cells
    P_fa = 1e-5  # desired false alarm rate
    SNR_OFFSET = -5  # dB

    RDM_dB_y1 = 10 * np.log10(np.abs(FFT_y1out) / np.max(np.abs(FFT_y1out)))
    RDM_dB_y2 = 10 * np.log10(np.abs(FFT_y2out) / np.max(np.abs(FFT_y2out)))

    # 对波束 A 和波束 B 分别执行 CA-CFAR 检测
    RDM_mask_A, cfar_ranges_A, cfar_dopps_A, K_A = ca_cfar(RDM_dB_y1, numGuard, numTrain, P_fa, SNR_OFFSET)
    RDM_mask_B, cfar_ranges_B, cfar_dopps_B, K_B = ca_cfar(RDM_dB_y2, numGuard, numTrain, P_fa, SNR_OFFSET)
    RDM_mask_A_all[:, :, t] = RDM_mask_A
    RDM_mask_B_all[:, :, t] = RDM_mask_B

    # 感觉cfar没最佳，RDM_mask_A、B存在几个点
    cfar_ranges_A = cfar_ranges_A
    cfar_ranges_B = cfar_ranges_B
    cfar_dopps_A = cfar_dopps_A
    cfar_dopps_B = cfar_dopps_B
    print(f"   {t} -> {len(cfar_dopps_A)}")
    if len(cfar_dopps_A) > 0 and len(cfar_ranges_A) > 0:
        TrgtR = Range[cfar_dopps_A[0]]  # 取第一个检测目标
        TrftV = Velocity[cfar_ranges_A[0]]  # 取第一个检测目标

        # 获取对应目标在波束 A 和 B 中的强度
        intensity_A = np.abs(FFT_y1out[cfar_ranges_A[0], cfar_dopps_A[0]])
        intensity_B = np.abs(FFT_y2out[cfar_ranges_B[0], cfar_dopps_B[0]])

        # 计算和（Σ）和差（Δ）
        sum_val = intensity_A + intensity_B
        diff_val = intensity_A - intensity_B

        # 计算和差比 Δ/Σ
        sum_diff_ratio = diff_val / sum_val

        # 根据和差比估计角度
        TrgtAngle = lookup_angle_from_sumdiffratio(AB_ybili.flatten(), sum_diff_ratio, theta, theta_min, theta_max)

        # 存下测距测速测角结果
        TrgtInform = np.array([TrgtR, TrftV, TrgtAngle])
        Detect_Result[:, t] = TrgtInform

# 航迹解算
r_all = Detect_Result[0, :]
theta_all = Detect_Result[2, :] + 90
xk_out = np.array([r_all * np.cos(theta_all * np.pi / 180),
                   r_all * np.sin(theta_all * np.pi / 180)])

RMSE_R_ave = np.mean(np.abs(Detect_Result[0, :] - Zk[0, :]))
RMSE_V_ave = np.mean(np.abs(Detect_Result[1, :] - Zk[1, :]))
RMSE_phi_ave = np.mean(np.abs(Detect_Result[2, :] - Zk[2, :]))

print(f'平均测距误差{RMSE_R_ave:.2f} m')
print(f'平均测速误差{RMSE_V_ave:.3f} m/s')
print(f'平均测角误差{RMSE_phi_ave:.4f} °')

# 绘图
plt.figure(4)
plt.plot(d_LinearArray.flatten(), np.zeros(M), 'g^', linewidth=1.1)
plt.plot(Xk[0, :], Xk[1, :], 'b--', linewidth=1.1)
plt.plot(xk_out[0, :], xk_out[1, :], 'rx', linewidth=1.1)
plt.legend(['雷达位置', '真实航迹', '点迹估计结果'], loc='southeast')
plt.xlabel('X（m)')
plt.ylabel('Y（m)')
plt.title('航迹检测结果')
plt.show()

plt.figure(5)
plt.plot(Xk[0, :], Xk[1, :], 'b--', linewidth=1.1)
plt.plot(xk_out[0, :], xk_out[1, :], 'rx', linewidth=1.1)
plt.legend(['真实航迹', '点迹估计结果'], loc='southeast')
plt.xlabel('X（m)')
plt.ylabel('Y（m)')
plt.title('航迹放大图')
plt.axis('equal')
plt.show()

# 绘制MTD和CFAR结果
t_index = 39  # 对应MATLAB中的40（Python索引从0开始）

fig = plt.figure(6)
X, Y = np.meshgrid(Range, Velocity)
plt.pcolormesh(X, Y, np.abs(FFT_y1out_all[:, :, t_index]))
plt.xlabel('距离（m)')
plt.ylabel('速度（m/s)')
plt.title('MTD-波束A-距离速度检测')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

fig = plt.figure(7)
plt.pcolormesh(X, Y, np.abs(FFT_y2out_all[:, :, t_index]))
plt.xlabel('距离（m)')
plt.ylabel('速度（m/s)')
plt.title('MTD-波束B-距离速度检测')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

fig = plt.figure(8)
plt.pcolormesh(X, Y, np.abs(RDM_mask_A_all[:, :, t_index]))
plt.xlabel('距离（m)')
plt.ylabel('速度（m/s)')
plt.title('CFAR-波束A-距离速度检测')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

fig = plt.figure(9)
plt.pcolormesh(X, Y, np.abs(RDM_mask_B_all[:, :, t_index]))
plt.xlabel('距离（m)')
plt.ylabel('速度（m/s)')
plt.title('CFAR-波束B-距离速度检测')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

# 绘制航迹的动态显示
plotTrajectory(Detect_Result)
