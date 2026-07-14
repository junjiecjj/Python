# Stage 4 ZF-SNR Gain vs N_RIS Design

## 目标

新增独立的 RIS 单元数增益扫描实验，用于观察 `N_RIS` 增大时：

- Random RIS 的 ZF 输出 SNR；
- Fixed-grid ZF-SNR optimized RIS 的 ZF 输出 SNR；
- optimized 相对 random 的 SNR / `G_ZF` 增益；
- 优化运行时间。

## 实验主线

本实验不把 RD 检测概率作为主指标。当前工程的 RIS 主优化目标是 ZF 归一化后的 SNR，因此 `N_RIS` 扫描直接在信道与 ZF 预编码输出层统计：

```text
SNR_zf = ||Heff(v) * B_zf(v)||_F^2 / noisePower
G_ZF = ||Heff(v) * B_zf(v)||_F^2
```

在固定 `noisePower` 下，`SNR gain dB` 与 `G_ZF gain dB` 是同一输出链路的两种表达。

## 默认正式参数

```text
NrisAxis = 4:4:64
numTrials = 100
objectiveType = "zf_snr"
searchMode = "fixed_grid"
phaseGridSize = 16
numStarts = 3
maxSweeps = 4
```

## 公平性

每个 `N_RIS` 点和每个 trial：

1. 重新构造对应维度的 `Hsr: Nr x Nt` 和 `Hrd: Nr x Nr`；
2. 在同一信道上比较 random 与 optimized；
3. optimized 的 start phases 包含 random 基线相位；
4. 不跨 `N_RIS` 维度复用旧信道或旧 RIS 相位。

## 输出指标

- random / optimized SNR dB
- SNR gain dB
- random / optimized `G_ZF`
- `G_ZF` gain dB
- random / optimized `cond(Heff)`
- random / optimized ZF raw power
- optimizer runtime

## 图件

- ZF output SNR vs `N_RIS`
- SNR gain vs `N_RIS`
- `G_ZF` gain vs `N_RIS`
- optimizer runtime vs `N_RIS`

均值曲线配 trial 标准差带，保留统计波动信息。
