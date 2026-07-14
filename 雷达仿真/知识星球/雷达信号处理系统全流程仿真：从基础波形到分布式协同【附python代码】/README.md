# Radar Sim — 雷达信号处理仿真平台

从零搭建的雷达信号处理仿真环境，用 Python (numpy/scipy) 验证雷达理论知识。

## 设计原则

1. **验证驱动**：每个模块必须有 `validate()` 函数，对比理论值和仿真值
2. **渐进叠加**：从单脉冲开始，逐步加入脉压、多普勒、CFAR、跟踪
3. **零外部依赖**：只用 numpy/scipy/matplotlib，不依赖雷达仿真框架
4. **知识库联动**：每个仿真模块对应 radar-knowledge-base 中的理论笔记

## 快速开始

```bash
pip install -r requirements.txt
python 01_signal_chain/s01_radar_equation.py          # 雷达方程与 SNR
python 01_signal_chain/s02_lfm_pulse_compression.py   # LFM 脉冲压缩
python 04_phased_array/s10_array_beamforming.py       # 阵列波束形成
python 04_phased_array/s12_doa_estimation.py          # MUSIC/ESPRIT 角度估计
python 05_clutter_stap/s13_clutter_model.py           # 杂波 RCS 与杂波谱
python 05_clutter_stap/s14_stap_processing.py         # MTI/STAP 处理
python 06_mimo_waveform/s15_mimo_virtual_array.py     # MIMO 虚拟阵列
python 06_mimo_waveform/s16_waveform_design.py        # 波形设计
python 07_distributed_deep/s17_quasi_coherent.py      # 准相参积累
python 07_distributed_deep/s18_multi_station_fusion.py # 多站定位
python 07_distributed_deep/s19_sync_protocol.py       # PTP 同步
```

## 项目结构

```
radar-sim/
  README.md
  requirements.txt
  lib/                              # 共享工具库
    __init__.py
    signal_utils.py                 # 信号生成、FFT、窗函数
    radar_params.py                 # 雷达参数数据类
    validation.py                   # 验证工具（对比理论值）
  01_signal_chain/                  # 阶段一：基础信号链
    s01_radar_equation.py           # 雷达方程与 SNR
    s02_lfm_pulse_compression.py    # LFM 波形与脉冲压缩
    s03_doppler_processing.py       # 多普勒处理与 MTD
    s04_cfar_detection.py           # CFAR 检测
    s05_single_pulse_chain.py       # 端到端串联
  02_distributed_coherent/          # 阶段二：分布式相参
    s06_coherent_integration.py     # 相参积累仿真
    s07_sync_error_impact.py        # 同步误差影响量化
  03_software_engineering/          # 阶段三：软件工程
    s08_pipeline_framework.py       # 处理流水线框架
    s09_middleware_prototype.py      # 中间件原型
  04_phased_array/                  # 阶段四：相控阵
    s10_array_beamforming.py        # 阵列波束形成
    s11_adaptive_beamforming.py     # 自适应波束形成
    s12_doa_estimation.py           # MUSIC/ESPRIT 角度估计
  05_clutter_stap/                  # 阶段五：杂波建模与 STAP
    s13_clutter_model.py            # 地面杂波 RCS、杂波谱建模
    s14_stap_processing.py          # MTI 滤波、空时自适应处理
  06_mimo_waveform/                 # 阶段六：MIMO 雷达与波形分集
    s15_mimo_virtual_array.py       # 正交波形分集，虚拟孔径扩展
    s16_waveform_design.py          # OFDM/相位编码波形互相关
  07_distributed_deep/              # 阶段七：分布式相参深化
    s17_quasi_coherent.py           # 准相参 vs 全相参性能对比
    s18_multi_station_fusion.py     # 多站定位融合
    s19_sync_protocol.py            # PTP 协议状态机建模
  output/                           # 生成的验证图像
```

## 模块说明

### 阶段一：单脉冲雷达信号链（验证基础知识）

| 模块 | 文件 | 验证目标 |
|------|------|---------|
| s01 | radar_equation.py | 雷达方程：R_max、SNR vs 距离曲线 |
| s02 | lfm_pulse_compression.py | LFM 波形：匹配滤波、距离分辨率、脉冲增益 |
| s03 | doppler_processing.py | 多普勒：MTD、速度分辨率、Range-Doppler Map |
| s04 | cfar_detection.py | CFAR：虚警率 Pfa、检测概率 Pd |
| s05 | single_pulse_chain.py | 端到端：发射→回波→脉压→多普勒→检测 |

### 阶段二：分布式雷达相参仿真（验证核心知识）

| 模块 | 文件 | 验证目标 |
|------|------|---------|
| s06 | coherent_integration.py | 相参积累：SNR 增益 = 10·log10(N) dB |
| s07 | sync_error_impact.py | 同步误差：相位误差→相干因子衰减 |

### 阶段三：软件工程实践（验证工程知识）

| 模块 | 文件 | 验证目标 |
|------|------|---------|
| s08 | pipeline_framework.py | 生产者-消费者流水线 |
| s09 | middleware_prototype.py | ZeroMQ 节点间数据分发 |

## 验证方法

每个模块的 `validate()` 函数输出 PASS/FAIL + 具体数值对比：

```
=== s01 雷达方程验证 ===
[PASS] 最大探测距离: 理论 157.2 km, 仿真 157.2 km, 误差 0.00%
[PASS] SNR@100km: 理论 18.3 dB, 仿真 18.3 dB, 误差 0.02 dB
=== 全部通过 ===
```

任何 FAIL 都意味着理论理解有偏差，需要回到 radar-knowledge-base 修正。

## 实施进度

- [x] lib/ 共享工具（3 文件）
- [x] s01 雷达方程与 SNR（4/4 PASS）
- [x] s02 LFM 脉冲压缩（5/5 PASS）
- [x] s03 多普勒处理（3/3 PASS）
- [x] s04 CFAR 检测（3/3 PASS）
- [x] s05 端到端串联（4/4 PASS）
- [x] s06 相参积累（12/12 PASS）
- [x] s07 同步误差量化（8/8 PASS）
- [x] s08 流水线框架（4/4 PASS）
- [x] s09 中间件原型（4/4 PASS）
- [x] s10 阵列波束形成（4/4 PASS）
- [x] s11 自适应波束形成（4/4 PASS）
- [x] s12 MUSIC/ESPRIT 角度估计（4/4 PASS）
- [x] s13 杂波 RCS 与杂波谱建模（4/4 PASS）
- [x] s14 MTI/STAP 处理（4/4 PASS）
- [x] s15 MIMO 虚拟阵列（6/6 PASS）
- [x] s16 波形设计（4/4 PASS）
- [x] s17 准相参积累（4/4 PASS）
- [x] s18 多站定位融合（4/4 PASS）
- [x] s19 PTP 同步协议（4/4 PASS）

**总计 81/81 验证项全部 PASS**

---

## 后续探索路线图

### 阶段四：相控阵波束形成（空间维）

| 模块 | 文件 | 验证目标 | 状态 |
|------|------|---------|------|
| s10 | array_beamforming.py | ULA 方向图、波束指向、旁瓣控制 | ✅ 4/4 PASS |
| s11 | adaptive_beamforming.py | MVDR/Capon 波束形成，干扰抑制 | ✅ 4/4 PASS |
| s12 | doa_estimation.py | MUSIC/ESPRIT 角度估计 | ✅ 4/4 PASS |

### 阶段五：杂波建模与抑制

| 模块 | 文件 | 验证目标 | 状态 |
|------|------|---------|------|
| s13 | clutter_model.py | 地面杂波 RCS、杂波谱建模 | ✅ 4/4 PASS |
| s14 | stap_processing.py | MTI 滤波、空时自适应处理 | ✅ 4/4 PASS |

### 阶段六：MIMO 雷达与波形分集

| 模块 | 文件 | 验证目标 | 状态 |
|------|------|---------|------|
| s15 | mimo_virtual_array.py | 正交波形分集，虚拟孔径扩展 | ✅ 6/6 PASS |
| s16 | waveform_design.py | OFDM/相位编码波形互相关 | ✅ 4/4 PASS |

### 阶段七：分布式相参深化

| 模块 | 文件 | 验证目标 | 状态 |
|------|------|---------|------|
| s17 | quasi_coherent.py | 准相参 vs 全相参性能对比 | ✅ 4/4 PASS |
| s18 | multi_station_fusion.py | 多站定位融合 | ✅ 4/4 PASS |
| s19 | sync_protocol.py | PTP 协议状态机建模 | ✅ 4/4 PASS |
