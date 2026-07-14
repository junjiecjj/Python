# Debug Log

## 2026-05-21 Stage 4.6 运行时汇总字段命名修正

- 现象：`tests/test_stage4_snr_gain_vs_nris.m` 首次运行在 `main_stage4_snr_gain_vs_nris.m` 中报错，提示无法识别汇总字段 `runtimeMean_s`。
- 定位：原始矩阵字段命名为 `runtime_s`，通用汇总器生成的是 `runtime_sMean/runtime_sStd`；主脚本验收和测试读取的是 `runtimeMean_s`，两种命名口径不一致。
- 修复：在 `summarize_raw` 中保留通用汇总字段，同时显式增加 `runtimeMean_s` 与 `runtimeStd_s` 别名，并让图和日志统一读取该别名。
- 复核：MATLAB 会话清理函数缓存后，`tests/test_stage4_snr_gain_vs_nris.m` 通过，Code Analyzer 对新主脚本无 issue。

## 2026-05-21 Stage 4.4 CFAR 图件缺失命中柱处理

- 现象：全图 CA-CFAR 后，`No RIS` 组在四个真值邻域内没有关联命中；CFAR 柱状图若直接按三组共同峰值范围设置 `ylim`，缺失值会干扰图件生成。
- 定位：这不是 CFAR 检测回退失败，而是当前遮挡 NLOS 基线按零目标回波建模后的预期结果；问题在于绘图逻辑没有先忽略未命中的 `NaN` 峰值。
- 修复：`main/main_stage4_rd_detection.m` 的 CFAR 柱状图 y 轴范围改为基于有效峰值并使用 `omitnan` 处理，保留 no-RIS 未命中状态。
- 验证：重新运行 Stage 4 主脚本后，`stage4_rd_detection_cfar_20260521_121642.png` 成功生成，no-RIS 未命中没有被局部峰回填。

## 2026-05-21 Stage 4.3 三维颜色条导出布局修正

- 现象：`stage4_rd_four_targets_nature_3d_clean_surface` 的 Magnitude 颜色条仍压近 3D 子图，影响 optimized RIS 面板阅读。
- 根因：三维坐标轴标签和标题在 `bbox_inches="tight"` 导出时会扩张整体边界，导致颜色条即使位于右侧 GridSpec 列，导出后仍被视觉压回图组内部。
- 修复：将 `clean_surface` 改为 `no_ris`、`random_ris`、`optimized_ris` 三张单面板图；每张图使用自己的右侧颜色条，避免三联 3D 布局和颜色条互相挤压。

## 2026-05-20 Stage 4 range FFT 轴修正

问题：最初的 Stage 4 单元测试中，目标距离 `25 m` 的 RD 峰值被检测到 `0 m`，测试失败。

定位：

- `generate_fmcw_echo.m` 生成的是复数解调 beat signal。
- `R = 25 m`、`S = 500 MHz / 50 us` 时，`fb = 2*S*R/c ≈ 1.67 MHz`。
- 当前采样率 `fs = 2 MHz`，复数采样可以表示 `0` 到接近 `fs` 的正频率；但初版 `range_doppler_fft.m` 按实信号处理，只保留了 `0` 到 `fs/2` 的半谱，导致 25 m 目标频率被丢弃。

修复：

- `range_doppler_fft.m` 改为保留完整 range FFT 频谱，range bins 数为 `nfftRange`。
- 文档中明确记录：当前模型是复数 beat signal，不能按实信号单边谱处理。

验证：

- `tests/test_stage4_fmcw_rd.m` 通过，单目标峰值距离和速度均接近真实值。
- `main/main_stage4_rd_detection.m` 通过，峰值位置为 `24.9 m` 和 `3.0438 m/s`。

## 2026-05-20 Stage 4 图像标注修正

问题：峰值柱状图中 MATLAB `categorical` 默认排序导致类别显示为 `optimized/random`，与代码输入顺序不一致；随后文字标注与负值柱状图发生重叠。

修复：

- 显式设置 categorical 顺序为 `random -> optimized`。
- 将提升值移动到柱状图标题中，避免文本覆盖柱子。

验证：

- 重新运行 `main_stage4_rd_detection.m`，输出图 `stage4_rd_detection_20260520_204249.png`，读图顺序正确。

## 2026-05-20 Stage 4.1 四目标图形 QA

问题：四目标 Nature 风格绘图第一版中，三维图的目标标记放在统一顶部高度，和真实谱峰高度不一致，容易误解为检测点悬浮在谱面之外。

修复：

- Python 脚本中将三维目标标记高度改为对应检测峰值 `peakDb + 1.2 dB`。
- 对每个目标在三维图中直接标注 `T1` 至 `T4`。

验证：

- 重新运行 `python scripts/plot_stage4_nature_figures.py`。
- 输出 `stage4_rd_four_targets_nature_2d.*` 和 `stage4_rd_four_targets_nature_3d.*`。
- Python 脚本通过 `py_compile`。

## 2026-05-21 Stage 4.2 三维 RD 渲染整改

问题：原三维 RD 图使用密集彩色 `plot_surface`，把低幅噪声底渲染成连续起伏地形，弱化了峰值位置和优化前后差异。

修复：

- 只修改 Python 绘图脚本，不改变 MATLAB 仿真数据。
- 三维图使用 `vmax - 40 dB` 作为低端压缩参考；低于该参考下限的点做软压缩，保留连续噪声底及小起伏而不再设为 `NaN`。
- 将三维采样密度降低到峰形仍可辨认的规模。
- 新增 `clean_surface` 和 `wireframe` 两版输出。
- clean surface 使用浅蓝单色曲面；wireframe 使用蓝色细线框；两版均保留红色峰值标记。

验证：

- 二维 RD 主图仍可正常导出。
- Python 绘图脚本通过 `py_compile` 和实际运行。

## 2026-05-17 Stage 3.4 优化器公平性与运行时间调整

问题 1：初版 `main_stage3_optimizer_comparison.m` 中，`random_best_of_numStarts` 和优化器使用的随机初值数量相同，但优化器内部除第一个初值外会自行生成其余初值，导致两者并非完全相同的 start phases。

定位：
- 该问题不会使结果硬编码或失效，但会削弱“公平对比”的严格性。
- 如果优化器恰好生成了更好的初值，部分收益可能被误归因于优化过程。

修复：
- 修改 `functions/optimize_ris_objective_driven.m`，允许 `options.initialV` 传入 `Nr x K` 初值矩阵。
- 修改 `main/main_stage3_optimizer_comparison.m`，让 `random_best_of_numStarts` 和所有优化器复用同一组 `startPhases`。

验证：
- MATLAB Code Analyzer 对 `optimize_ris_objective_driven.m` 和 `main_stage3_optimizer_comparison.m` 均无 warning。
- 重新运行 30 trial 对照，`coarse_to_fine_zf_snr` 和带 condition penalty 的版本均明显优于两个随机基线。

问题 2：初版 30 trial 参数较重，交互式工具内运行接近或超过 120 秒。

修复：
- 将 Stage 3.4 诊断参数设为 `numStarts = 3`、`maxSweeps = 4`、`phaseGridSize = 16`、`coarseGridSize = 16`、`fineGridSize = 8`、`finerGridSize = 6`。
- 该设置用于交互式阶段诊断；后续正式图3/图4可根据运行时间提高 Monte Carlo 数和搜索精度。

## 2026-05-17 Stage 3.3 自检问题

### 多 start history 串接造成图像误读

- 现象：旧图中 `objective_path_gain` 和 `objective_zf_snr` 曲线出现跳变，且曲线终点不一定等于最终 best。
- 根因：多个 start 的普通 history 被直接串接，不应被解释为单条连续优化曲线。
- 修复：`optimize_ris_objective_driven.m` 增加 best-so-far 历史字段；诊断图改画 best-so-far 曲线。
- 验证：`main_stage3_zf_snr_stability.m` 中 `Best-so-far endpoints match final best: true`。

### path gain 与 ZF-SNR 不一致

- 现象：有些 trial 优化后 path gain 不一定高于 random，但 SNR 明显更高。
- 根因：ZF-SNR 主要受 `pinv(Heff)` 和条件数影响，而不仅是 `||Heff||_F^2`。
- 处理：后续主线以 `zf_snr` 为目标，path gain 仅作为辅助诊断指标。

## 2026-05-17 目标统一整改中的问题

### ADMM 数值发散

- 现象：`rhoScale = 0.2` 时，`x` 和 `mu` 范数快速指数增长，后续 `Heff` 出现 NaN/Inf。
- 根因：`Q/T` 归一化后，`rho` 仍然相对过小，`rho*I + T` 虽可逆但动力学不稳定。
- 修复：默认 `rhoScale` 改为 `2`，即按 `2*norm(T,2)` 量级设置 `rho`。
- 验证：最终 `rho = 0.97179`、`normT = 0.4859`，ADMM 收敛标志为 `true`，primal residual `1.0051e-07`，dual residual `4.047e-07`。

### 代理目标与工程目标冲突

- 现象：quadratic ADMM 和 path_gain optimizer 都会提高 path gain，但降低 ZF-SNR。
- 根因：ZF-SNR 受 `pinv(Heff)` 和条件数影响，`||Heff||_F^2` 不是充分目标。
- 处理：新增 `evaluate_ris_objective.m` 和 `optimize_ris_objective_driven.m`，直接优化 `zf_snr`。

## 2026-05-17 第三阶段 ADMM 整改问题记录

### 问题 1：原 `optimize_ris_admm.m` 不是严格 ADMM

- 现象：原实现使用 finite-difference phase-gradient surrogate，并令 `x = candidateU`，随后 `u = project(x - mu/rho)`。
- 根因：`x` 已经接近单位模，导致 `u` 近似等于 `x`，primal residual 接近 0，`mu` 基本不起作用。
- 修复：删除该主体逻辑，改为闭式 ADMM：
  - `u = exp(1j * angle(x - mu/rho))`
  - `x = (rho*I + T)^(-1) * (rho*u + mu)`
  - `mu = mu + rho*(u - x)`

### 问题 2：论文 `T` 矩阵无法直接等价到当前 `||Heff||_F^2`

- 现象：当前 `Heff = Hsr^H * diag(v) * Hrd * diag(v)^H * Hsr`，`||Heff||_F^2` 是四次目标。
- 根因：论文 ADMM 的闭式 `x` 更新依赖二次型 `x^H T x`；当前真实 path gain 不是二次型。
- 修复：构造维度自洽的二次代理：
  - `Q = (Hsr*Hsr^H) .* transpose(Hrd)`
  - `Qh = (Q + Q^H)/2`
  - `T(1:Nr,1:Nr) = -Qh`
  - `T(Nr+1,Nr+1) = 0`
- 标注：该实现为 `quadratic_admm_approximation`，不是严格论文 ADMM。

### 问题 3：surrogate 优于 ADMM 近似

- 现象：最终验证中 surrogate path gain 和 SNR 均高于 quadratic ADMM approximation。
- 判断：如实保留该结果，不强行调参或放大 ADMM 输出。
- 后续：若要复现论文图3/图4，需要继续推导更接近论文物理模型的 `T`，或重新定义与 ZF SNR 一致的优化目标。

## 2026-05-17 Stage 3 Validation Issues

### Missing path-gain function

- Script: `main/main_stage3_admm_validation.m`
- Symptom: first validation run failed because `compute_path_gain` was undefined.
- Root cause: validation was written before the production function, as intended.
- Fix: added `functions/compute_path_gain.m`.

### Path gain improvement can reduce ZF SNR

- Functions: `optimize_ris_admm.m`, `design_precoder_zf.m`, `compute_snr.m`
- Symptom: with `gradientStep = 0.25`, path gain increased from `8.9838e-08` to `5.2076e-07`, but ZF-normalized SNR dropped from `-82.614 dB` to `-83.5887 dB`.
- Evidence: `pinv(Heff)` raw power increased from `7.3023e+08 W` to `9.1396e+08 W`, and condition number increased from about `5.24` to about `19.41`.
- Root cause: maximizing `||Heff||_F^2` alone does not guarantee better ZF-normalized SNR; aggressive phase steps can make `Heff` more ill-conditioned.
- Fix: use conservative validation settings, `gradientStep = 0.005` and `maxIter = 50`.
- Verification: rerun improved path gain by `1.1034 dB` and ZF-normalized SNR by `0.24576 dB`.

本文件记录每次 bug、错误信息、定位过程、修复方案和遗留问题。

## 记录模板

```text
日期：
相关脚本/函数：
错误信息：
复现步骤：
定位过程：
根因判断：
修复方案：
验证方式：
遗留问题：
```

## 2026-05-17 工程初始化

- 当前未进行算法实现，暂无运行错误。
- 已识别后续高风险 debug 点：
  - 接收信号模型的矩阵维度。
  - `H_rd` 的物理含义和维度。
  - ADMM 扩展变量 `x`、`u`、`mu` 的维度。
  - ZF 伪逆在病态信道下的稳定性。
  - FMCW 三角扫频与标准距离-多普勒 FFT 的对应关系。

## 2026-05-17 Stage 2 Validation Issues

### Expected failing validation before implementation

- 相关脚本：`main/main_stage2_model_validation.m`
- 现象：首次运行失败在 `generate_channels.m`，错误为 `generate_channels is a placeholder`。
- 判断：这是 validation-first 的预期失败，证明新验证脚本能够捕获基础函数未实现状态。
- 处理：实现 `paper_params.m`、`generate_channels.m`、`design_precoder_zf.m`、`compute_snr.m` 后重新运行。

### Empty diary log

- 相关脚本：`main/main_stage2_model_validation.m`
- 现象：MATLAB 控制台显示验证通过，但 `diary` 生成的成功日志文件为空。
- 根因判断：当前 MCP MATLAB 执行环境下 `diary` 捕获不可靠。
- 修复方案：移除对 `diary` 的依赖，改为在验证结束后显式构造 `logLines` 并用 `writelines(logLines, logPath)` 写入日志。
- 验证：重新运行后 `outputs/logs/stage2_model_validation_20260517_153924.txt` 包含完整维度、ZF 和 SNR 验证结果。
