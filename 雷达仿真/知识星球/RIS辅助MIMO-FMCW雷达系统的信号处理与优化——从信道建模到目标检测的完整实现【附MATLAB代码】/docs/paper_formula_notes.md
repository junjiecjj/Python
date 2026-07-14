# Paper Formula Notes

## Stage 4.6：N_RIS 扫描指标定义

当前 RIS 单元数扫描不把四目标检测概率作为主指标，而是直接在等效信道和 ZF 归一化链路上比较相位优化收益：

```text
Heff(v) = Hsr' * diag(v) * Hrd * diag(v)' * Hsr
B(v) = normalized ZF precoder from pinv(Heff(v))
G_ZF(v) = ||Heff(v) * B(v)||_F^2
SNR_ZF(v) = G_ZF(v) / noisePower_W
```

在同一个 `N_RIS`、同一个 trial、同一组 `Hsr/Hrd` 上定义：

```text
SNR gain dB = SNR_ZF(v_opt)_dB - SNR_ZF(v_random)_dB
G_ZF gain dB = 10*log10(G_ZF(v_opt) / G_ZF(v_random))
```

当前 `noisePower_W` 在两个方法之间保持一致，因此 `SNR gain dB` 与 `G_ZF gain dB` 数值应一致；两者同时保存，是为了区分“信道输出功率指标”和“带噪声归一化后的 SNR 指标”。

维度随 `N_RIS = Nr` 同步变化：

```text
Hsr: Nr x Nt
Hrd: Nr x Nr
v:   Nr x 1
Heff: Nt x Nt
B: Nt x Nt
```

该实验的优化器目标仍是 `evaluate_ris_objective(..., "zf_snr")`，不是 `Pd`、不是 `path_gain`，也不是 quadratic ADMM 代理目标。

## Stage 4.5：Pd-vs-SNR 统计定义

当前检测概率实验对每个回波 SNR 点和每个目标统计 CA-CFAR 关联命中：

```text
Pd_q(SNR_i, method) = hitCount_q(SNR_i, method) / Nmc
Pd_avg(SNR_i, method) = mean_q Pd_q(SNR_i, method)
```

其中 `hitCount_q` 只累计 `detect_rd_targets_cfar.m` 在目标 `q` 真值邻域内关联到的全图 CFAR 峰。

当前主横轴采用 optimized RIS 参考回波定义：

```text
Pref,opt = mean(|Yopt,noiseless[n,m]|^2)
echoNoisePower_W(SNR_i) = Pref,opt / 10^(SNR_i/10)
```

同一个 `SNR_i` 点下，三组方法使用同一 `echoNoisePower_W`：

```text
No RIS:        Y = noise only
Random RIS:    Y = target echo with sqrt(G_ZF_random) + noise
Optimized RIS: Y = target echo with sqrt(G_ZF_optimized) + noise
```

这样 `Random RIS` 与 `Optimized RIS` 的 Pd 差异保留了当前 RIS/ZF 链路增益差异，而不是把每个方法单独归一到相同自身 SNR。

## Stage 4.4：二维 CA-CFAR 与目标关联

CFAR 检测在 RD 线性功率域执行：

```text
P_RD[k,l] = |RD_complex[k,l]|^2
noise_hat[k,l] = mean(training cells around CUT)
threshold[k,l] = alpha_cfar * noise_hat[k,l]
detection[k,l] = P_RD[k,l] > threshold[k,l]
```

对 CA-CFAR，当前代码按训练单元数 `Ntrain` 和虚警概率 `Pfa` 使用：

```text
alpha_cfar = Ntrain * (Pfa^(-1/Ntrain) - 1)
```

Stage 4.4 当前流程为：

```text
1. 对完整 RD 功率图执行二维 CA-CFAR
2. 在 CFAR 检测单元中提取局部极大值
3. 在真值目标的距离/速度邻域内关联最近的 CFAR 峰
4. 关联成功才记录目标 CFAR 峰值、距离误差和速度误差
```

当前四目标脚本使用的 CFAR 参数为：

```text
trainingCells = [6, 6]
guardCells = [2, 2]
Pfa = 1e-4
localMaxRadiusCells = [1, 1]
```

其中真值邻域只用于关联和验收，不参与 CFAR 门限计算，也不替代全图检测结果。

## Stage 4.3：三组 RD 幅度约定

```text
G_ZF(v) = ||Heff(v) * B(v)||_F^2
A_q(random/optimized) = sqrt(G_ZF(v)) * alpha_q
A_q(no RIS) = 0
```

`No RIS` 当前表示遮挡 NLOS 零目标回波基线；random 与 optimized 组仍沿用同一等效信道和 ZF 归一化链路。

## Stage 4：FMCW Beat Signal 与距离-多普勒处理

当前 Stage 4 采用基础解调后 FMCW beat signal 模型：

```text
y[n,m] = sum_q A_q exp(j 2*pi*(fb_q*n*Ts + fD_q*m*Tc)) + w[n,m]
```

其中：

```text
S = B / Tchirp
lambda = c / fc
fb_q = 2*S*R_q/c
fD_q = 2*v_q/lambda
Ts = 1/fs
Tc = Tchirp
```

矩阵约定：

- `Y`: `Nfast x Nchirp`
- 行：fast-time sample index `n`
- 列：slow-time chirp index `m`
- `Nfast = round(fs * Tchirp)`
- `Nchirp = params.radar.numChirps`

RIS/ZF 链路增益用于回波幅度：

```text
Heff(v) = Hsr' * diag(v) * Hrd * diag(v)' * Hsr
B(v) = ZF precoder after power normalization
G_ZF(v) = ||Heff(v) * B(v)||_F^2
A_q(v) = sqrt(G_ZF(v)) * alpha_q
```

距离-多普勒处理：

```text
Range FFT:   FFT along fast-time dimension
Doppler FFT: FFT along slow-time dimension, followed by fftshift
RD_dB = 20*log10(abs(RD_complex) + eps)
```

注意：当前 `generate_fmcw_echo.m` 生成的是复数解调 beat signal，因此 `range_doppler_fft.m` 保留完整 range-frequency 区间，而不是只取实信号半谱。距离轴为：

```text
f_range[k] = k * fs / NfftRange
rangeAxis[k] = f_range[k] * c / (2*S)
```

速度轴为：

```text
f_doppler[l] = shifted_l / (NfftDoppler * Tc)
velocityAxis[l] = f_doppler[l] * lambda / 2
```

## Stage 3.4：工程优化目标与相位搜索公式

当前工程主目标不再使用 `trace(Heff)` 或单纯路径增益代理，而是直接使用 ZF 预编码后的 SNR：

```text
Phi = diag(v), |v_i| = 1
Heff = Hsr' * Phi * Hrd * Phi' * Hsr
B_zf = normalized_pinv_precoder(Heff, Ptx)
SNR_zf = ||Heff * B_zf||_F^2 / sigma^2
```

其中：

- `Hsr`: `Nr x Nt`
- `Hrd`: `Nr x Nr`
- `v`: `Nr x 1`
- `Phi`: `Nr x Nr`
- `Heff`: `Nt x Nt`
- `B_zf`: `Nt x Nt`
- `Ptx`: 线性发射功率，单位 W
- `sigma^2`: 线性噪声功率，单位 W

`evaluate_ris_objective.m` 当前支持的主目标：

```text
path_gain = ||Heff||_F^2
zf_snr = SNR_zf
zf_snr_with_condition_penalty = SNR_zf / (1 + alpha * log10(cond(Heff))^2)
```

Stage 3.4 中 `optimize_ris_objective_driven.m` 采用多初值坐标相位搜索。对第 `i` 个 RIS 单元，固定其他相位，只在候选相位集合中选择使指定 objective 最大的相位：

```text
v_i <- arg max_{exp(j theta), theta in candidateGrid} objective(v)
```

`searchMode = "fixed_grid"` 使用固定全局相位网格；`searchMode = "coarse_to_fine"` 使用三层候选集：

```text
第1层：全局粗搜，例如 16 或 24 个相位点
第2层：围绕当前最优相位 ±pi/12 局部细搜
第3层：围绕当前最优相位 ±pi/48 更细局部搜索
```

该优化器不是论文严格 ADMM，也不声称优化论文二次型代理目标。它是当前工程主线的 ZF-SNR-driven RIS phase optimizer。

## Stage 3.3：ZF-SNR 作为主目标

当前主线目标不再是单纯路径增益：

```text
path_gain = ||Heff||_F^2
```

而是 ZF 归一化后的 SNR：

```text
B_zf = sqrt(P / ||pinv(Heff)||_F^2) * pinv(Heff)
zf_snr = ||Heff * B_zf||_F^2 / noisePower
```

因此 `cond(Heff)` 和 `||pinv(Heff)||_F^2` 会直接影响最终 SNR。当前实验已经说明：

- 增大 `path_gain` 可能恶化 `cond(Heff)`，从而降低 ZF-SNR。
- 直接优化 `zf_snr` 可以降低 `cond(Heff)` 和 ZF raw power，从而提高 ZF-SNR。

后续图3/图4若以 SNR 为纵轴，主算法应优先使用 `objectiveType = "zf_snr"`，而不是 `path_gain` 或 quadratic ADMM proxy。

## 第三阶段目标函数统一

当前统一目标函数由 `evaluate_ris_objective.m` 提供：

```text
Heff = Hsr^H * diag(v) * Hrd * diag(v)^H * Hsr
path_gain = ||Heff||_F^2
zf_snr = ||Heff * B_zf||_F^2 / noisePower
B_zf = sqrt(P / ||pinv(Heff)||_F^2) * pinv(Heff)
zf_snr_with_condition_penalty = zf_snr / (1 + alpha * log10(cond(Heff))^2)
```

这次整改后的核心原则是：优化器必须明确优化目标，不能再优化 `trace(Heff)` 代理目标，却用 `path_gain` 或 `zf_snr` 作为主要结论。

`quadratic_admm_approximation` 的目标仍是二次代理：

```text
Q = (Hsr*Hsr^H) .* transpose(Hrd)
Qh = (Q + Q^H)/2
Qh = Qh / max(||Qh||_F, eps)
T(1:Nr,1:Nr) = -Qh
```

该 ADMM 只用于诊断代理目标行为。真实工程目标由 `optimize_ris_objective_driven.m` 直接优化。

## 第三阶段整改后的 ADMM 公式说明

当前工程固定使用：

```text
Hsr: Nr x Nt
Hrd: Nr x Nr
Phi = diag(v): Nr x Nr
Heff = Hsr^H * Phi * Hrd * Phi^H * Hsr
gain = ||Heff||_F^2
```

严格检查后，`gain = ||Heff||_F^2` 在当前 `Hrd = Nr x Nr` 设定下是关于 `v` 和 `conj(v)` 的四次目标，不能直接构造成论文中的二次型 `x^H T x`。因此本轮实现的是“二次型 ADMM 近似”，不是声称已经完全复现论文原始 `T` 矩阵。

本轮采用的二次代理来自：

```text
trace(Heff) = v^H * Q * v
Q = (Hsr*Hsr^H) .* transpose(Hrd)
Qh = (Q + Q^H)/2
```

为了最大化 `real(v^H Qh v)`，ADMM 写成最小化：

```text
min 0.5 * x^H * T * x
T(1:Nr,1:Nr) = -Qh
T(Nr+1,Nr+1) = 0
s.t. |u_i| = 1, u = x
```

闭式更新：

```text
u^{k+1} = exp(1j * angle(x^k - mu^k/rho))
x^{k+1} = (rho*I + T)^(-1) * (rho*u^{k+1} + mu^k)
mu^{k+1} = mu^k + rho*(u^{k+1} - x^{k+1})
```

相位恢复：

```text
v = exp(1j * angle(x(1:Nr) / x(Nr+1)))
```

`optimize_ris_admm.m` 不再使用有限差分梯度。有限差分方法被移动到 `optimize_ris_surrogate.m`，仅用于对照。

## Stage 3 ADMM Phase Optimization Notes

Current executable dimensions:

```text
Hsr: Nr x Nt
Hrd: Nr x Nr
v:   Nr x 1, |v_i| = 1
Phi = diag(v): Nr x Nr
Heff = Hsr^H * Phi * Hrd * Phi^H * Hsr: Nt x Nt
B: Nt x Nt
```

Stage 3 path-gain objective:

```text
gain(v) = ||Heff(v)||_F^2
Heff(v) = Hsr^H * diag(v) * Hrd * diag(v)^H * Hsr
```

This is the executable path-gain objective for the current code convention. It is not claimed to be identical to the paper's printed quadratic `T`-matrix objective, because the current `Hrd: Nr x Nr` convention makes `||Heff||_F^2` quartic in `v`.

Implemented ADMM surrogate:

```text
x  : surrogate phase-update variable
u  : unit-modulus projection variable
mu : consensus multiplier
rho: consensus penalty
```

Iteration outline:

```text
1. initialize u = v0, x = u, mu = 0
2. estimate finite-difference gradient of gain(exp(j theta)) with respect to theta
3. take a conservative backtracking phase step for the x update
4. project u = exp(j angle(x - mu/rho))
5. update mu = mu + rho * (u - x)
6. record objective, primal residual, and dual residual
```

Stage 3 validation used:

```text
maxIter = 50
rho = 1
gradientStep = 0.005
finiteDifferenceStep = 1e-4
```

Important: larger steps increased path gain but worsened `pinv(Heff)` conditioning and reduced ZF-normalized SNR. The conservative step is a stability choice based on actual validation, not a figure-matching adjustment.

## 论文核心问题

论文研究 RIS 辅助 MIMO-FMCW 雷达在非视距场景下的目标距离和速度估计。RIS 用于重构传播链路，使基站发射信号经 RIS 反射后照射遮挡目标，再由回波链路返回并形成可处理的雷达回波。

## 场景模型

- 基站发射天线数：`N_t`。
- 基站接收天线数：`N_b`。
- RIS 反射单元数：`N_r`。
- RIS 部署在基站与目标之间，用于建立非视距辅助传播链路。
- 论文默认存在 MIMO 与 RIS 之间的控制链路。

## 发射信号

FMCW 发射信号形式：

```text
s(t) = A0 * exp(j * (2*pi*fc*t + pi*gamma*t^2 + phi0))
```

其中 `fc` 为起始频率，`T` 为调频周期，`gamma` 为调频斜率，`A0` 为初始幅度，`phi0` 为初始相位。

## 接收信号

论文给出的接收信号模型为：

```text
y = H_sr^H * Phi * H_rd * Phi^H * H_sr * B * s + n
```

需要后续重点检查的地方：

- 论文符号中 `H_rd` 被描述为 RIS 到目标的信道矩阵，但接收模型中涉及双程链路，维度和物理含义需要在代码实现前再次核对。
- `H_sr`、`H_rd`、`Phi`、`B` 的矩阵维度必须通过小尺寸数值测试验证。

## 差拍信号

论文给出混频、滤波后的中频差拍信号：

```text
r_b(t) = channel_gain * B * s * A0^2
         * exp(j * [2*pi*(fc*tau - B/(2T)*tau^2 + B/T*tau*t) + phi0 + phi1])
         + n
```

其中 `tau` 为传播时延，`phi1` 为目标反射引起的相位差。上式在 PDF 中存在排版和符号压缩，后续实现需要重新整理成代码可执行形式。

## 距离和速度估计

静止目标拍频：

```text
f_IF = gamma * tau = 4 * B * R0 / (T * c)
R0 = f_IF * T * c / (4 * B)
```

运动目标上下扫频拍频：

```text
f_up   = gamma * tau - f_d = 4 * B * R0 / (T * c) - 2*v/lambda0
f_down = gamma * tau + f_d = 4 * B * R0 / (T * c) + 2*v/lambda0
```

距离和速度：

```text
R0 = (f_up + f_down) * T * c / (8 * B)
v  = (f_up - f_down) * lambda0 / 4
```

分辨率：

```text
Delta_R = c / (2 * B)
Delta_v = lambda0 / T
```

注意：论文图2描述三角调频，但实验参数表中给出单个 chirp 周期和 chirp 数量。后续需要明确图5、图6代码到底采用三角扫频、锯齿扫频还是等效距离-多普勒处理。

## 优化问题

论文将目标反射后的 SNR 最大化写为：

```text
max_{B, v} || H_sr^H * Phi * H_rd * Phi^H * H_sr * B ||_F^2 / sigma^2
s.t. ||B||_F^2 <= P
     Phi = beta * diag(v), |v(i)| = 1
```

该问题关于 `B` 和 `Phi` 非凸。论文采用交替优化思想：

- 固定 `Phi`，用 ZF 设计 `B`。
- 固定或等效处理 `B`，用路径增益最大化准则通过 ADMM 优化 `Phi`。

## ZF 预编码

论文目标：

```text
H_eff * B = I
B = Pi^dagger = Pi^H * (Pi * Pi^H)^(-1)
```

其中 `Pi` 表示等效信道。实现时需要处理不可逆、病态矩阵和功率归一化。

## RIS ADMM 相移优化

论文将路径增益最大化问题转成：

```text
min 0.5 * x^H * T * x
s.t. |u(i)| = 1, u = x
```

增广拉格朗日变量包括 `x`、`u`、`mu` 和惩罚参数 `rho`。主要迭代：

```text
u_{k+1} = phase(x_k - rho^(-1) * mu_k)
x_{k+1} = (rho * I + T)^(-1) * (rho * u_{k+1} + mu_k)
mu_{k+1} = mu_k + rho * (u_{k+1} - x_{k+1})
```

论文算法表中还给出 `mu_{k+1} = T * x_{k+1}` 的等价更新关系，需要后续确认采用哪一种更稳定。

## 仿真参数初版

来自论文表1和实验段落：

| 参数 | 符号 | 数值 |
| --- | --- | --- |
| 发射天线数 | `N_t` | 4 |
| 接收天线数 | `N_b` | 4 |
| 载波频率 | `f_c` | 77 GHz |
| 扫频带宽 | `B` | 500 MHz |
| 发射功率 | `P` | 10 dBm |
| 单个 chirp 时间 | `T` | 50 us |
| 采样率 | `f_s` | 2 MHz |
| chirp 数量 | `N_chirp` | 256 |
| 光速 | `c` | 3e8 m/s |
| RCS | `RCS` | 1 |
| 最大迭代次数 | `k_max` | 1000 |
| Rician 因子 | `K` | 10 dB |
| 路径损耗指数 | `alpha` | 2 |
| 收敛阈值 | `epsilon` | 1e-2 / 1e-3 / 1e-4 |

多目标设置：

| 目标 | 距离 | 速度 |
| --- | --- | --- |
| Target 1 | 25 m | -1 m/s |
| Target 2 | 20 m | 1 m/s |
| Target 3 | 10 m | -1 m/s |
| Target 4 | 5 m | 1 m/s |

## 图表目标

- 图3：ADMM 与 CD 的 SNR 随 `N_r` 增大而提高，ADMM 始终高于 CD。
- 图4：SNR 随发射功率增大而提高，ADMM 高于 CD；论文文字强调 `N_r = 16` 的 ADMM 可优于 `N_r = 64` 的 CD。
- 图5：四目标距离-时间三维幅度谱，展示多目标距离轨迹和幅度分布。
- 图6：`N_r = 4, 16, 64` 下 ADMM/CD 距离-多普勒图；RIS 单元数增加后目标峰值更清晰，背景噪声相对降低。

## Stage 2 Actual Matrix Convention

第二阶段基础模型采用以下可执行维度约定：

```text
Nt = 4
Nb = 4
Nr = 16 by default
Hsr: Nr x Nt
Hrd: Nr x Nr
Phi: Nr x Nr
Heff = Hsr^H * Phi * Hrd * Phi^H * Hsr
Heff: Nt x Nt
B: Nt x Nt
```

这里的 `Hrd` 暂时不是严格的 RIS-to-target 单程信道，而是 RIS 域的目标散射/回波等效矩阵。采用该约定的原因是它能与论文接收模型中的双 RIS 相移结构形成维度一致的 `Heff`。该处理已经同步记录到 `reproduction_assumptions.md`。

## Stage 2 SNR Formula

代码中 `compute_snr.m` 使用：

```text
SNR_linear = ||Heff * B||_F^2 / noisePower_W
SNR_dB = 10 * log10(SNR_linear)
```

其中 `noisePower_W` 必须是线性瓦特值，不能传入 dBm。

## Stage 2 ZF Formula

代码中 `design_precoder_zf.m` 使用：

```text
B_raw = pinv(Heff)
P_raw = ||B_raw||_F^2
B = sqrt(P_tx_W / P_raw) * B_raw
```

归一化后：

```text
||B||_F^2 = P_tx_W
```

ZF 误差记录为：

```text
rawRelativeError = ||Heff * B_raw - I||_F / ||I||_F
relativeError = ||Heff * B - scale * I||_F / ||scale * I||_F
```

注意：归一化后的 `Heff * B` 等于缩放后的单位阵，而不是单位阵本身。这是功率约束下 ZF 预编码的预期结果。
