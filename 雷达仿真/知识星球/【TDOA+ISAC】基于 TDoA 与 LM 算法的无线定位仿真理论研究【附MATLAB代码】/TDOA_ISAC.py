#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:28:01 2025

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import time
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

# ======================= 用户区参数（可改） ==========================
np.random.seed(42)                      # 固定随机种子
c = 299792458                           # 光速 (m/s)

areaSize = np.array([60, 40])           # 场地大小 [X宽, Y高] (m)
TRP = np.array([[0, 0],                  # 4 个基站（可改/可增减）
               [areaSize[0], 0],
               [areaSize[0], areaSize[1]],
               [0, areaSize[1]]])
refIdx = 0                              # 参考基站索引（做差的基站）

N_UE = 200                              # 本次仿真的 UE 数量
SNRdB = 25                              # 接收信噪比 (dB)
enableNLoS = True                       # 是否启用 NLoS
useLoSOnly = True                       # TDoA 解算时仅用 LoS 基站（需≥3个LoS+参考）

losProb = 0.65                          # 每条链路为 LoS 的概率（仅用于仿真标签）
reflAtten = 0.6                         # NLoS 反射额外衰减系数（幅度）
fs = 61.44e6                            # 采样率（越高→TOA分辨率越高）
zcLen = 1023                            # Zadoff-Chu 长度（探针序列）
guardZeros = 2048                       # 序列前后补零，避免卷积越界

# 环境散射体
Nscatter = 20                           # 散射体数量
showOneUEPaths = True                   # 额外绘出一个 UE 的 NLoS 路径
uShow = 0                               # 显示第几个 UE 的路径（若存在 NLoS）

# LM（Levenberg–Marquardt）参数（稳健求解 TDoA）
maxIter = 50
tol = 1e-6
lambda0 = 1e-2
lambdaGrow = 10
lambdaShrink = 0.3
lambdaMax = 1e9
wFloor = 0.05
stepClip = 5.0

# 可视化选项
plotGeometry = True                     # 几何图（含散射体、真值/估计/误差向量）
plotCDF = True                          # 误差 CDF
plotHist = True                         # 误差直方图
plotHeat = True                         # 区域误差热力图（中位数）
showVectors = True
vecMaxShow = 400
heatBins = [12, 8]

# ======================= 生成散射体 ================================
SC = np.column_stack([areaSize[0] * np.random.rand(Nscatter),
                     areaSize[1] * np.random.rand(Nscatter)])  # Nscatter×2
SC_used_count = np.zeros(Nscatter)      # 记录被选中次数（全部 UE 的所有 NLoS 链路）

# ======================= 预先生成"探针序列" =========================
root = 29
Nzc = zcLen
n = np.arange(Nzc)
# ZC 基带复序列（CAZAC）
zc = np.exp(-1j * np.pi * root * n * (n + 1) / Nzc)
tx = np.concatenate([np.zeros(guardZeros), zc, np.zeros(guardZeros)])
Nt = len(tx)
tgrid = np.arange(Nt) / fs

M = TRP.shape[0]
TRP = TRP.astype(float)
SNRlin = 10 ** (SNRdB / 10)

# ======================= 结果缓存 ==========================
UE_true = np.zeros((N_UE, 2))
UE_est = np.full((N_UE, 2), np.nan)
posErr = np.full(N_UE, np.nan)
iters = np.zeros(N_UE)
usedK = np.zeros(N_UE)
fallbackCnt = 0
failSolve = 0

# 记录每个 UE / 基站使用了哪个散射体（0 表示 LoS）
usedSIdx = np.zeros((N_UE, M))

# ======================= 主循环：逐 UE 仿真 ==========================
print("开始主循环仿真...")
start_time = time.time()

for u in range(N_UE):
    if u % 50 == 0:
        print(f"处理第 {u}/{N_UE} 个 UE...")

    # ---------- 随机 UE 位置 ----------
    UE = np.array([2 + (areaSize[0] - 4) * np.random.rand(),
                   2 + (areaSize[1] - 4) * np.random.rand()])
    UE_true[u, :] = UE

    # ---------- LoS/NLoS 标签 ----------
    isLoS = np.random.rand(M) < losProb
    isLoS[refIdx] = True

    # ---------- 直达距离 ----------
    dist_LOS = np.sqrt(np.sum((TRP - UE) ** 2, axis=1))  # M×1

    # ---------- 生成接收信号（LoS：直达；NLoS：经单次散射） ----------
    rx = np.zeros((Nt, M), dtype=complex)

    for m in range(M):
        if enableNLoS and not isLoS[m]:
            # 计算所有散射体的单跳路径长度：|UE-S| + |S-TRP_m|
            dxUE = SC[:, 0] - UE[0]
            dyUE = SC[:, 1] - UE[1]
            rUE = np.sqrt(dxUE ** 2 + dyUE ** 2)

            dxTR = SC[:, 0] - TRP[m, 0]
            dyTR = SC[:, 1] - TRP[m, 1]
            rTR = np.sqrt(dxTR ** 2 + dyTR ** 2)

            totalL = rUE + rTR  # Nscatter×1
            idxS = np.argmin(totalL)
            d_bounce = totalL[idxS]

            if d_bounce <= dist_LOS[m] + 1e-6:  # 三角不等式的等号保护
                d_bounce = dist_LOS[m] + 1e-6

            tau_m = d_bounce / c
            usedSIdx[u, m] = idxS + 1  # 记录所用散射体（+1 为了与MATLAB索引对应）
            SC_used_count[idxS] += 1

            # 振幅：路径损耗 ~ 1/d，附加反射衰减
            amp = reflAtten / max(d_bounce, 1)
        else:
            tau_m = dist_LOS[m] / c  # LoS
            amp = 1 / max(dist_LOS[m], 1)
            usedSIdx[u, m] = 0

        # 生成分数时延的接收波形 + 加噪
        tm = tgrid - tau_m

        # 使用线性插值
        # real_interp = interp1d(tgrid, tx.real, kind='linear', bounds_error=False, fill_value=0)
        # imag_interp = interp1d(tgrid, tx.imag, kind='linear', bounds_error=False, fill_value=0)
        # sm_real = real_interp(tm)
        # sm_imag = imag_interp(tm)
        # sm = sm_real + 1j * sm_imag
        interp = interp1d(tgrid, tx.real, kind='linear', bounds_error=False, fill_value=0)
        # imag_interp = interp1d(tgrid, tx.imag, kind='linear', bounds_error=False, fill_value=0)
        # sm_real = real_interp(tm)
        sm = interp(tm)
        # sm = sm_real + 1j * sm_imag

        # 以 amp 缩放，同时统一噪声方差基准（按 amp^2 近似）
        sigPow = np.mean(np.abs(tx) ** 2) * amp ** 2
        noiseVar = sigPow / SNRlin
        noise = np.sqrt(noiseVar / 2) * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        rx[:, m] = amp * sm + noise

    # ---------- 匹配滤波 + 三点插值估 TOA ----------
    est_tau = np.zeros(M)
    peakAmp = np.zeros(M)

    for m in range(M):
        cc = signal.correlate(rx[:, m], tx, mode='full')
        lags = signal.correlation_lags(len(rx[:, m]), len(tx), mode='full')
        abscc = np.abs(cc)
        I = np.argmax(abscc)

        if 0 < I < len(cc) - 1:
            y1 = abscc[I - 1]
            y2 = abscc[I]
            y3 = abscc[I + 1]
            denom = (y1 - 2 * y2 + y3)
            if abs(denom) > np.finfo(float).eps:
                delta = 0.5 * (y1 - y3) / denom  # 分数偏移（-1..+1）
            else:
                delta = 0
        else:
            delta = 0

        lagRefined = lags[I] + delta
        est_tau[m] = lagRefined / fs
        peakAmp[m] = max(abscc[I], np.finfo(float).eps)

    # ---------- 选择用于 TDoA 的基站 ----------
    useIdx = np.arange(M)
    if useLoSOnly:
        useIdx = np.where(isLoS)[0]
        if refIdx not in useIdx:
            useIdx = np.unique(np.concatenate(([refIdx], useIdx)))

    if len(useIdx) < 3:
        useIdx = np.arange(M)
        fallbackCnt += 1

    K = len(useIdx)
    usedK[u] = K

    refLocal = np.where(useIdx == refIdx)[0]
    if len(refLocal) == 0:
        useIdx = np.unique(np.concatenate(([refIdx], useIdx)))
        refLocal = [0]
        K = len(useIdx)
        usedK[u] = K
    else:
        refLocal = refLocal[0]

    tauSel = est_tau[useIdx]
    TRPSel = TRP[useIdx, :]
    peakSel = peakAmp[useIdx]

    # ---------- 相对延时与差距程 ----------
    dTau = tauSel - tauSel[refLocal]  # K×1
    meas = c * dTau[1:]  # (K-1)×1

    # ---------- 简单量测门控 ----------
    diagArea = np.hypot(areaSize[0], areaSize[1])
    # 基站间最大间距
    maxPair = 0
    for i in range(K - 1):
        dx = TRPSel[i, 0] - TRPSel[i + 1:, 0]
        dy = TRPSel[i, 1] - TRPSel[i + 1:, 1]
        v = np.sqrt(dx ** 2 + dy ** 2)
        if len(v) > 0:
            maxPair = max(maxPair, np.max(v))

    geoBound = diagArea + maxPair
    if np.any(~np.isfinite(meas)) or np.any(np.abs(meas) > 2 * geoBound):
        failSolve += 1
        continue

    # ---------- LM 求解（矢量化） ----------
    p = np.mean(TRPSel, axis=0).reshape(2, 1)
    lam = lambda0
    ok = False

    idxNR = np.arange(K)
    idxNR = np.delete(idxNR, refLocal)

    w = peakSel.copy()
    w = w / (np.max(w) + (np.max(w) == 0))
    w = np.maximum(w, wFloor)
    wnr = w[idxNR]
    sqrtw = np.sqrt(wnr)

    # 初值残差/J
    s1 = TRPSel[refLocal, :].reshape(2, 1)
    dx1 = p[0] - s1[0]
    dy1 = p[1] - s1[1]
    d1 = max(np.sqrt(dx1 ** 2 + dy1 ** 2), 1e-9)

    sk = TRPSel[idxNR, :].T
    dxk = p[0] - sk[0, :]
    dyk = p[1] - sk[1, :]
    dk = np.sqrt(dxk ** 2 + dyk ** 2)
    dk = np.maximum(dk, 1e-9)

    r = (dk - d1) - meas
    J = np.column_stack([(dxk / dk) - dx1 / d1, (dyk / dk) - dy1 / d1])

    Jw = J * sqrtw.reshape(-1, 1)
    rw = r * sqrtw
    cost0 = rw.T @ rw

    for it in range(maxIter):
        H = Jw.T @ Jw
        g = Jw.T @ rw

        if not np.isfinite(np.linalg.cond(H)) or np.linalg.cond(H) < 1e-12:
            lam = min(lam * lambdaGrow, lambdaMax)

        Hd = H + lam * np.diag(np.maximum(np.diag(H), 1))

        if (not np.isfinite(np.linalg.cond(Hd)) or
            np.linalg.cond(Hd) < 1e-15 or
            np.any(~np.isfinite(g))):
            lam = min(lam * lambdaGrow, lambdaMax)
            if lam >= lambdaMax:
                break
            continue

        try:
            dp = -np.linalg.solve(Hd, g)
        except np.linalg.LinAlgError:
            lam = min(lam * lambdaGrow, lambdaMax)
            if lam >= lambdaMax:
                break
            continue

        ndp = np.linalg.norm(dp)
        if ndp > stepClip:
            dp = dp * (stepClip / ndp)

        p_new = p + dp.reshape(2, 1)

        # 新 r/J/cost
        dx1 = p_new[0] - s1[0]
        dy1 = p_new[1] - s1[1]
        d1 = max(np.sqrt(dx1 ** 2 + dy1 ** 2), 1e-9)

        dxk = p_new[0] - sk[0, :]
        dyk = p_new[1] - sk[1, :]
        dk = np.sqrt(dxk ** 2 + dyk ** 2)
        dk = np.maximum(dk, 1e-9)

        r_new = (dk - d1) - meas
        J_new = np.column_stack([(dxk / dk) - dx1 / d1, (dyk / dk) - dy1 / d1])

        rw_new = r_new * sqrtw
        cost_new = rw_new.T @ rw_new

        if np.isfinite(cost_new) and cost_new < cost0:
            p = p_new
            r = r_new
            J = J_new
            Jw = J * sqrtw.reshape(-1, 1)
            rw = rw_new
            cost0 = cost_new
            lam = max(lam * lambdaShrink, 1e-12)

            if np.linalg.norm(dp) < tol:
                ok = True
                break
        else:
            lam = min(lam * lambdaGrow, lambdaMax)
            if lam >= lambdaMax:
                break

    if not ok or np.any(~np.isfinite(p)):
        failSolve += 1
        continue

    UE_est[u, :] = p.flatten()
    posErr[u] = np.linalg.norm(UE - UE_est[u, :])
    iters[u] = it + 1  # +1 因为Python从0开始计数

end_time = time.time()
print(f"主循环完成，耗时: {end_time - start_time:.2f} 秒")

# ======================= 统计结果（忽略 NaN） =======================
e = posErr
good = np.isfinite(e)
Nvalid = np.sum(good)

if Nvalid == 0:
    raise ValueError('全部解算失败；请检查几何/参数。')

es = np.sort(e[good])
cdfy = np.arange(1, len(es) + 1) / len(es)

def get_percentile(data, perc):
    idx = max(0, int(np.ceil(perc * len(data))) - 1)
    return data[idx]

P10 = get_percentile(es, 0.10)
P50 = get_percentile(es, 0.50)
P90 = get_percentile(es, 0.90)

print('\n=== TDoA 多UE统计（含散射体的NLoS模型） ===')
print(f'UE 总数: {N_UE}, 成功: {Nvalid}, 失败: {failSolve}（LoSOnly兜底: {fallbackCnt} 次）')
print(f'误差:  均值={np.mean(e[good]):.3f} m, 中位={P50:.3f} m, P10={P10:.3f} m, P90={P90:.3f} m, 最大={np.max(e[good]):.3f} m')
print(f'迭代次数: 均值={np.mean(iters[good]):.2f}, 中位={np.median(iters[good]):.0f}（成功样本）')
print(f'每次解算参与基站数（均值，成功样本）: {np.mean(usedK[good]):.2f}')

# ======================= 可视化 =======================

# 可视化 1：几何图（含散射体）
if plotGeometry:
    plt.figure('Geometry with Scatterers')
    plt.clf()
    plt.axis('equal')
    plt.grid(True)

    # 绘制场地边界
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), areaSize[0], areaSize[1], fill=False, edgecolor=[0.6, 0.6, 0.6])
    plt.gca().add_patch(rect)

    # 散射体使用热度
    sCount = SC_used_count
    hSC = plt.scatter(SC[:, 0], SC[:, 1], s=20 + 3 * sCount, c=sCount, cmap='viridis')
    plt.colorbar(hSC, label='散射体被使用次数')

    hTRP = plt.plot(TRP[:, 0], TRP[:, 1], 'ks', markersize=8, linewidth=1.5, label='TRP')[0]
    hTrue = plt.plot(UE_true[good, 0], UE_true[good, 1], 'g.', markersize=12, label='UE真值(成功)')[0]
    hEst = plt.plot(UE_est[good, 0], UE_est[good, 1], 'rx', markersize=6, linewidth=1, label='UE估计')[0]

    if showVectors:
        idxv = np.where(good)[0]
        nShow = min(vecMaxShow, len(idxv))
        if nShow > 0:
            idxv_show = np.random.choice(idxv, nShow, replace=False)
            for i in idxv_show:
                plt.arrow(UE_true[i, 0], UE_true[i, 1],
                         UE_est[i, 0] - UE_true[i, 0],
                         UE_est[i, 1] - UE_true[i, 1],
                         color=[0.85, 0.2, 0.2], linewidth=0.8, head_width=0.5, head_length=0.5)

    plt.legend(handles=[hTRP, hSC, hTrue, hEst], loc='best')
    plt.title(f'TDoA 多UE（含散射体）：N={N_UE}(成功 {Nvalid}), SNR={SNRdB} dB, fs={fs/1e6:.2f} MHz')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()

# 可视化 1b：单UE的NLoS路径（可选）
if showOneUEPaths:
    u0 = min(max(0, uShow), N_UE - 1)
    plt.figure('One UE NLoS Paths')
    plt.clf()
    plt.axis('equal')
    plt.grid(True)

    # 绘制场地边界
    rect = Rectangle((0, 0), areaSize[0], areaSize[1], fill=False, edgecolor=[0.7, 0.7, 0.7])
    plt.gca().add_patch(rect)

    plt.plot(TRP[:, 0], TRP[:, 1], 'ks', markersize=8, linewidth=1.5, label='TRP')
    plt.plot(UE_true[u0, 0], UE_true[u0, 1], 'go', markersize=8, linewidth=1.5, label='UE真值')
    plt.plot(UE_est[u0, 0], UE_est[u0, 1], 'rx', markersize=8, linewidth=1.5, label='UE估计')
    plt.scatter(SC[:, 0], SC[:, 1], s=12, c=[0.6, 0.6, 0.6], label='散射体(灰)')

    # 把该 UE 的 NLoS 链路画出来（UE->S->TRP）
    for m in range(M):
        k = int(usedSIdx[u0, m])
        if k > 0:
            S = SC[k - 1, :]  # -1 因为之前加了1
            plt.plot([UE_true[u0, 0], S[0]], [UE_true[u0, 1], S[1]], 'm--', linewidth=1.2)
            plt.plot([S[0], TRP[m, 0]], [S[1], TRP[m, 1]], 'm--', linewidth=1.2)
            plt.plot(S[0], S[1], 'md', markersize=6, linewidth=1.2, label='NLoS路径' if m == 0 else "")

    plt.legend(loc='best')
    plt.title(f'单UE的NLoS路径示意（UE #{u0 + 1}）')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()

# 可视化 2：误差 CDF
if plotCDF:
    plt.figure('Error CDF')
    plt.clf()
    plt.grid(True)
    plt.plot(es, cdfy, linewidth=1.5)

    yl = plt.ylim()
    plt.plot([P10, P10], yl, 'k--')
    plt.text(P10, 0.05, f'P10={P10:.2f} m', ha='left', va='bottom')
    plt.plot([P50, P50], yl, 'k--')
    plt.text(P50, 0.35, f'P50={P50:.2f} m', ha='left', va='bottom')
    plt.plot([P90, P90], yl, 'k--')
    plt.text(P90, 0.75, f'P90={P90:.2f} m', ha='left', va='bottom')

    plt.xlabel('定位误差 (m)')
    plt.ylabel('概率')
    plt.title('误差 CDF（成功样本）')
    plt.tight_layout()

# 可视化 3：误差直方图
if plotHist:
    plt.figure('Error Histogram')
    plt.clf()
    plt.grid(True)
    plt.hist(e[good], bins=max(10, int(np.sqrt(Nvalid))), density=True)
    plt.xlabel('定位误差 (m)')
    plt.ylabel('概率密度')
    plt.title('误差直方图（成功样本）')
    plt.tight_layout()

# 可视化 4：区域误差热力图（中位数）
if plotHeat:
    nx, ny = heatBins
    xedges = np.linspace(0, areaSize[0], nx + 1)
    yedges = np.linspace(0, areaSize[1], ny + 1)

    # 离散化坐标
    xi = np.digitize(UE_true[good, 0], xedges) - 1
    yi = np.digitize(UE_true[good, 1], yedges) - 1

    # 确保索引在有效范围内
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    medMap = np.full((nx, ny), np.nan)

    for i in range(nx):
        for j in range(ny):
            mask = (xi == i) & (yi == j)
            if np.any(mask):
                medMap[i, j] = np.median(e[good][mask])

    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])

    plt.figure('Median Error Heatmap')
    plt.clf()
    im = plt.imshow(medMap.T, extent=[xcent[0], xcent[-1], ycent[0], ycent[-1]],
                   origin='lower', aspect='auto')
    plt.colorbar(im, label='中位误差 (m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('各网格中位误差 (m)（成功样本）')
    plt.tight_layout()

plt.show()



























































































































































































































