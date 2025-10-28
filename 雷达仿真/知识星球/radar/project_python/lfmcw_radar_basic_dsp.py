# %%
import numpy as np
import scipy
import pandas as pd
import scipy.signal
from tabulate import tabulate
import drawhelp.draw as dh
import plotly.graph_objects as go
from joblib import Parallel, delayed


# %%

"""1. 加载数据，设置一些雷达参数"""
data = scipy.io.loadmat("../data/RadarData_2024_05_31_09_34_52.mat")
radarDataCube = scipy.fft.ifft(data["frames"], axis=3)
numFrame, numChannel, numChrip, numSampling = radarDataCube.shape
print(f"帧数    \t{numFrame}\n通道数\t{numChannel}\nchrip数\t{numChrip}\n采样点数\t{numSampling}")

from scipy.constants import speed_of_light as c
from steering_vector import steering_vector

bandwidth = 1000e6  # 带宽
fc = 24e9  # 载波频率
T_chirp = 420e-6  # chirp 持续时间
T_idle = 2588e-6  # 两个 chirp 之间的间隔时间
T_nop = 118e-6
Fs = 2.5e6 / 8  # 采样频率

numTx = 1  # 发射天线数量
numRx = 2  # 接收天线数量

lambda_ = c / fc
d_rx = lambda_ / 2  # 接收天线之间的距离，单位米
d_tx = 34e-3  # 发射天线之间的距离，单位米

# 目标初始化信息
target_info = np.array(
    [
        [10 * np.cos(np.radians(10)), 10 * np.sin(np.radians(10)), 0, 0, 0, 0],
        [20 * np.cos(np.radians(-45)), 20 * np.sin(np.radians(-45)), 0, 1 * np.cos(np.radians(-45)), 1 * np.sin(np.radians(-45)), 0],
    ]
).T

# 发射天线的位置
tx_pos = np.array([[0] * numTx, np.linspace(-0.5, 0.5, numTx) * d_tx * (numTx - 1), [0] * numTx])

# 接收天线的位置
rx_pos = np.array([[0] * numRx, np.linspace(-0.5, 0.5, numRx) * d_rx * (numRx - 1), [0] * numRx])


max_velocity = (numChrip - 1) * c / (2 * fc * (T_chirp + T_idle) * numChrip)
axis_t = np.arange(0, (T_chirp + T_idle) * numFrame * numChrip + 1 / Fs, 1 / Fs)  # 时间轴
axis_range = np.arange(numSampling) * c / 2 / (bandwidth * numSampling * (1 / Fs) / T_chirp)  # 输入信号混频后是负频率，因此坐标轴反向
axis_velocity = -1 * (np.arange(-np.floor(numChrip / 2), -np.floor(numChrip / 2) + 1)) / (numChrip - 1) * max_velocity
axis_angle = np.arange(-90, 91)  # 角度范围
T_frame = (T_chirp + T_idle) * numChrip + T_nop


# %%
"""去静态杂波(去均值)"""
windwosSize = 32
radarDataCube_mean = np.mean(radarDataCube, axis=(0, 2), keepdims=True)
radarDataCube_move = radarDataCube - radarDataCube_mean


# %% [markdown]
# 观察单通道的RDM

# %%
"""观察单通道的RDM"""
frames = radarDataCube_move[:, 0, :, :]
rdms = [np.abs(np.fft.fft2(f)) for f in frames]

dh.draw_2d_spectrumlist(rdms[::10]).show()

# %%
"""使用capon算法绘制 角度距离谱"""


# 定义处理每一帧的函数
def process_frame(i, radarDataCube_move, steering_vectors):
    frame = radarDataCube_move[i]
    frame = np.fft.fft(frame, axis=2)
    range_az_mvdr = np.zeros((frame.shape[2], steering_vectors.shape[1]))

    # 计算每个距离单元的capon谱
    for j in range(frame.shape[2]):
        x = frame[:, :, j]
        Cxx = np.matmul(x, x.T.conj())
        sv_conj = steering_vectors.conj().T
        numerator = np.sum(np.matmul(sv_conj, np.linalg.inv(Cxx)) * steering_vectors.T, axis=1)
        range_az_mvdr[j, :] = 1 / np.abs(numerator)

    return range_az_mvdr


# 导向矢量
steering_vectors = steering_vector(rx_pos, np.vstack([axis_angle, np.zeros_like(axis_angle)]), c / (fc + bandwidth / 2))

# 使用joblib对最外层循环进行并行化
rams = Parallel(n_jobs=-1)(delayed(process_frame)(i, radarDataCube_move, steering_vectors) for i in range(numFrame))
rams = np.array(rams)


axis_angle_rad = np.radians(axis_angle)
polarX = np.outer(axis_range, np.cos(axis_angle_rad))
polarY = np.outer(axis_range, np.sin(axis_angle_rad))

dh.draw_2d_spectrumlist(x=polarX, y=polarY, z=rams[::40]).show()


# %%
""" 使用相位差方法计算角度
对于1发2收雷达来说，使用capon算法或者是music等阵列信号DOA算法和直接计算两个通道的相位差并没有明显区别。
"""

positionCapon = []
for i in range(len(rams)):
    f = rams[i]
    f[0, :] = 0
    t = np.argmax(f)
    pa = axis_angle[t % f.shape[1]]
    pr = axis_range[t // f.shape[1]]
    positionCapon.append([np.array([pr * np.cos(np.radians(pa)), pr * np.sin(np.radians(pa))])])

# %% 2.先通过距离维的幅度谱找到目标所在的距离单元，然后通过相位差计算角度


def calcPosition_phase(frame, resRange) -> np.ndarray:
    ampSpecRDM = np.abs(np.fft.fft2(scipy.signal.detrend(frame[0], axis=0)))
    ampSpecRDM = ampSpecRDM[:, :64]
    t = np.argmax(ampSpecRDM)
    indexRangeRDM = t % ampSpecRDM.shape[1]
    indexVelocityRDM = t // ampSpecRDM.shape[1]

    phase0 = np.angle(np.fft.fft2(frame[0])[indexVelocityRDM, indexRangeRDM])
    phase1 = np.angle(np.fft.fft2(frame[1])[indexVelocityRDM, indexRangeRDM])
    phaseDelta = (phase1 - phase0) % (2 * np.pi)
    if phaseDelta > np.pi:
        phaseDelta -= 2 * np.pi
    elif phaseDelta < -np.pi:
        phaseDelta += 2 * np.pi
    theta = np.arcsin(phaseDelta / (np.pi))
    r = indexRangeRDM * resRange
    res = np.array([r * np.cos(theta), r * np.sin(theta)])
    return res


resRange = axis_range[1]

positionPhase = []
for i in range(len(radarDataCube_move)):
    pos = calcPosition_phase(radarDataCube_move[i], resRange)
    positionPhase.append([pos])

# %%
"""比较相位差法和Capon算法的结果 """


listData = []
for i in range(len(positionPhase)):
    data = []
    x = [j[0] for j in positionPhase[i]]
    y = [j[1] for j in positionPhase[i]]
    data.append(go.Scatter(x=x, y=y, mode="markers", name="Phase"))

    x = [j[0] for j in positionCapon[i]]
    y = [j[1] for j in positionCapon[i]]
    data.append(go.Scatter(x=x, y=y, mode="markers", name="Capon"))

    listData.append(data)

fig = dh.draw_animation(listData, title="单目标检测——相位差法和Capon算法峰值检测结果")
fig.update_layout(yaxis_range=[-5, 5], xaxis_range=[0, 8], xaxis_title="前后方向", yaxis_title="左右方向")
fig.show()
# dh.save_plotly_animation_as_video(fig, "plotly_animation.mp4", fps=20)

# 计算标准差
posPhase = np.array([j for i in positionPhase for j in i])
posCapon = np.array([j for i in positionCapon for j in i])
distances = np.linalg.norm(posPhase - posCapon, axis=1)
distances = distances[distances <= np.percentile(distances, 90)]
std_dev = np.std(distances)
mean_dev = np.mean(distances)
print(f"标准差: {std_dev}m 平均差: {mean_dev}m")
# %%
