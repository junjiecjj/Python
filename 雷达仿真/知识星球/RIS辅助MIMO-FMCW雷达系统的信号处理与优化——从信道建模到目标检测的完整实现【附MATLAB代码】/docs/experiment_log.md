# Experiment Log

## 2026-05-22 Stage 4.6 N_RIS 与 ZF-SNR 增益正式大样本实验

- 运行主函数：`main_stage4_snr_gain_vs_nris()`
- 扫描轴：`NrisAxis = 4:4:64`，共 `16` 个 RIS 单元数点。
- Monte Carlo：每个 `N_RIS` 点 `100` 次，共 `1600` 个 trial。
- 主优化器：固定网格 ZF-SNR 相位搜索，`phaseGridSize = 16`，`numStarts = 3`，`maxSweeps = 4`。
- 输出图：`outputs/figures/stage4_snr_gain_vs_nris_20260522_085344.png`
- 输出数据：`outputs/data/stage4_snr_gain_vs_nris_20260522_085344.mat`
- 输出日志：`outputs/logs/stage4_snr_gain_vs_nris_20260522_085344.txt`
- 日志格式：逐 trial 打印 random SNR、optimized SNR、SNR gain、`G_ZF` gain、条件数和 runtime，长跑时可直接在 MATLAB 命令行观察趋势。

正式统计摘要：

| 指标 | 结果 |
| --- | --- |
| 平均 SNR 增益最小值 | `8.2752 dB`，出现在 `N_RIS = 4` |
| 平均 SNR 增益最大值 | `17.4857 dB`，出现在 `N_RIS = 60` |
| `N_RIS = 64` 平均 SNR 增益 | `17.1692 dB` |
| `N_RIS = 64` random SNR mean | `-72.1103 dB` |
| `N_RIS = 64` optimized SNR mean | `-54.9411 dB` |
| 所有 `N_RIS` 点 optimized mean SNR 是否高于 random | `true` |
| 汇总是否有限 | `true` |

结论：

- 相比 random RIS，fixed-grid ZF-SNR optimized RIS 在全部 `16` 个 `N_RIS` 点上都给出正平均增益。
- 随着 `N_RIS` 增大，random 与 optimized 的绝对 ZF 输出 SNR 整体抬升，优化增益均值也由约 `8 dB` 上升到约 `17 dB` 区间。
- 增益曲线存在 Monte Carlo 波动和局部回落，不应把它解释为严格单调理论曲线；大样本结果支持“更多 RIS 单元在当前模型下整体提升 ZF-SNR 可优化空间”这一工程结论。
- 当前正式实验已经满足“横坐标更密、样本更丰富、输出可追踪”的要求。

## 2026-05-21 Stage 4.6 N_RIS 与 ZF-SNR 增益扫描缩减验证

- 运行主函数：`main_stage4_snr_gain_vs_nris(opts)`
- 缩减验证参数：`NrisAxis = [4, 8, 12, 16]`，`numTrials = 3`
- 正式默认参数：`NrisAxis = 4:4:64`，`numTrials = 100`，`phaseGridSize = 16`，`numStarts = 3`，`maxSweeps = 4`
- 主优化器：`fixed_grid` 坐标相位搜索，`objectiveType = "zf_snr"`
- 比较对象：同一 trial 下的 random RIS 与 fixed-grid optimized RIS
- 输出图：`outputs/figures/stage4_snr_gain_vs_nris_validation_20260521.png`
- 输出数据：`outputs/data/stage4_snr_gain_vs_nris_validation_20260521.mat`
- 输出日志：`outputs/logs/stage4_snr_gain_vs_nris_validation_20260521.txt`

缩减验证统计：

| N_RIS | random SNR mean dB | optimized SNR mean dB | mean SNR gain dB | mean runtime s |
| ---: | ---: | ---: | ---: | ---: |
| `4` | `-109.3906` | `-99.9628` | `9.4278` | `0.1106` |
| `8` | `-92.2718` | `-80.0387` | `12.2330` | `0.2882` |
| `12` | `-89.5187` | `-76.0909` | `13.4279` | `0.4356` |
| `16` | `-82.7750` | `-73.3829` | `9.3921` | `0.6248` |

验收结论：

- 新脚本可保存 `.png`、`.mat` 和 `.txt`，命令行逐 trial 统计正常打印。
- 缩减验证中每个 `N_RIS` 点的优化相位平均 ZF-SNR 均高于随机相位。
- 在当前固定噪声定义下，`G_ZF` 增益与 ZF-SNR 增益一致。
- 3-trial 验证只用于检查链路，不用于判断正式曲线的平滑性和饱和趋势；正式趋势需要运行默认 `100` 次 Monte Carlo。

## 2026-05-21 Stage 4.5 CA-CFAR Pd-vs-SNR quick 验证

- 运行主函数：`main_stage4_pd_vs_snr("quick")`
- 模式：`quick`
- Monte Carlo 次数：`8`
- 信道设置：固定一组 Stage 4 信道和 RIS 相位，每个 trial 重采样回波噪声。
- 检测规则：全图 CA-CFAR + 真值目标邻域关联。
- SNR 轴：`[-40, -35, -30, -25, -20, -15, -10] dB`
- 输出图：`outputs/figures/stage4_pd_vs_snr_quick_20260521_134702.png`
- 输出数据：`outputs/data/stage4_pd_vs_snr_quick_20260521_134702.mat`
- 输出日志：`outputs/logs/stage4_pd_vs_snr_quick_20260521_134702.txt`

平均 Pd 结果：

| Echo SNR dB | No RIS | Random RIS | Fixed-grid optimized RIS |
| ---: | ---: | ---: | ---: |
| `-40` | `0` | `0` | `0` |
| `-35` | `0` | `0` | `0` |
| `-30` | `0` | `0` | `0.0625` |
| `-25` | `0` | `0` | `0.25` |
| `-20` | `0` | `0.0625` | `0.9375` |
| `-15` | `0` | `0.3125` | `1` |
| `-10` | `0` | `0.90625` | `1` |

验收结论：

- quick 脚本无报错，`.png`、`.mat` 和 `.txt` 输出成功。
- 命令行逐 trial 表格式打印已启用，可用于后续 full 模式本地观察趋势。
- optimized RIS 的平均 Pd 过渡区明显早于 random RIS。
- No RIS 在当前遮挡 NLOS 零目标回波模型下保持最低。
- quick 统计只用于功能验收和趋势预检查；正式结果应运行 full 模式或提高 Monte Carlo 次数。

## 2026-05-21 Stage 4.4 全图 CA-CFAR 检测验证

- 运行脚本：`main/main_stage4_rd_detection.m`、`scripts/plot_stage4_nature_figures.py`
- 检测逻辑：完整 RD 功率图 CA-CFAR 检测，随后对 CFAR 局部峰做四目标真值邻域关联。
- CFAR 参数：`trainingCells = [6,6]`、`guardCells = [2,2]`、`Pfa = 1e-4`、`localMaxRadiusCells = [1,1]`
- 输出日志：`outputs/logs/stage4_rd_detection_20260521_121642.txt`
- 输出数据：
  - `outputs/data/stage4_rd_detection_cfar_20260521_121642.mat`
  - `outputs/data/stage4_rd_four_targets_cfar_latest.mat`
  - `outputs/data/stage4_rd_four_targets_cfar_detection_latest.csv`
- 输出图件：
  - `outputs/figures/stage4_rd_detection_cfar_20260521_121642.png`
  - `outputs/figures/stage4_rd_detection_cfar_20260521_121642.fig`
  - `outputs/figures/stage4_rd_four_targets_cfar_nature_2d.png/.svg`
  - `outputs/figures/stage4_rd_four_targets_cfar_nature_3d_clean_surface_*.png/.svg`
  - `outputs/figures/stage4_rd_four_targets_cfar_nature_3d_wireframe.png/.svg`

CFAR 结果：

| 组别 | 全图 CFAR 峰数 | 四目标关联命中数 |
| --- | ---: | ---: |
| no RIS | `4` | `0` |
| random RIS | `7` | `4` |
| optimized RIS | `7` | `4` |

| 目标 | random CFAR peak dB | optimized CFAR peak dB | optimized vs random |
| --- | ---: | ---: | ---: |
| T1 | `-29.0675` | `-18.2766` | `10.7909 dB` |
| T2 | `-30.3685` | `-19.5840` | `10.7845 dB` |
| T3 | `-31.6786` | `-20.8907` | `10.7879 dB` |
| T4 | `-33.2032` | `-22.4240` | `10.7792 dB` |

验收结论：

- local-peak 检测输出保留，CFAR 输出使用独立 `cfar` 命名。
- random RIS 与 optimized RIS 的四个目标均被全图 CFAR 峰关联命中。
- optimized RIS 的四目标 CFAR 峰值均高于 random RIS，提升约 `10.78 dB`。
- `No RIS` 组在当前遮挡 NLOS 零目标回波模型下没有真值目标 CFAR 关联命中。
- MATLAB CFAR 测试、Stage 4 FMCW RD 测试和 Python 绘图检查均通过。

## 2026-05-21 Stage 4.3 三组 RD 图复核

- 运行脚本：`main/main_stage4_rd_detection.m`、`scripts/plot_stage4_nature_figures.py`。
- 对照组：无 RIS、随机相位 RIS、固定网格 ZF-SNR 优化 RIS。
- optimized 相比 random 的四目标 RD 峰值提升约为 `[10.7909, 10.7845, 10.7879, 10.7792] dB`。
- 图形输出：`stage4_rd_four_targets_nature_2d.*`、`stage4_rd_four_targets_nature_3d_clean_surface_no_ris.*`、`stage4_rd_four_targets_nature_3d_clean_surface_random_ris.*`、`stage4_rd_four_targets_nature_3d_clean_surface_optimized_ris.*`、`stage4_rd_four_targets_nature_3d_wireframe.*`。

## 2026-05-20 Stage 4 FMCW RD 检测验证

- 运行脚本：`main/main_stage4_rd_detection.m`
- 主优化器：`fixed_grid_zf_snr`
- 输出日志：`outputs/logs/stage4_rd_detection_20260520_204249.txt`
- 输出数据：`outputs/data/stage4_rd_detection_20260520_204249.mat`
- 输出图：
  - `outputs/figures/stage4_rd_detection_20260520_204249.png`
  - `outputs/figures/stage4_rd_detection_20260520_204249.fig`

目标参数：

- 距离：`25 m`
- 速度：`3 m/s`
- 散射系数：`alpha = 1`
- 回波噪声功率：`1e-12 W`

检测结果：

| 指标 | random RIS | fixed-grid ZF-SNR optimized RIS |
| --- | ---: | ---: |
| `G_ZF = ||Heff*B||_F^2` | `4.5576e-11` | `5.4626e-10` |
| Stage 3 风格 ZF-SNR | `-83.4126 dB` | `-72.6260 dB` |
| RD 局部目标峰值 | `-28.4769 dB` | `-17.6983 dB` |
| 峰值距离 | `24.9 m` | `24.9 m` |
| 峰值速度 | `3.0438 m/s` | `3.0438 m/s` |

验收：

- `G_ZF` 提升：`10.7866 dB`
- RD 峰值提升：`10.7786 dB`
- 距离检测：`PASS`
- 速度检测：`PASS`
- optimized 峰值高于 random：`PASS`
- 总体验收：`PASS`

回归验证：

- `tests/test_stage4_fmcw_rd.m`：`1 Passed, 0 Failed`
- `main/main_stage2_model_validation.m`：通过，输出 `stage2_model_validation_20260520_204059`
- `main/main_stage3_zf_snr_stability.m`：通过，8 个 trial failure count 为 `0`

## 2026-05-20 Stage 4.1 四目标 RD 验证与 Nature 风格绘图

- 运行 MATLAB 脚本：`main/main_stage4_rd_detection.m`
- 运行 Python 绘图脚本：`python scripts/plot_stage4_nature_figures.py`
- 主优化器：`fixed_grid_zf_snr`
- 输出日志：`outputs/logs/stage4_rd_detection_20260520_210255.txt`
- 输出数据：
  - `outputs/data/stage4_rd_detection_20260520_210255.mat`
  - `outputs/data/stage4_rd_four_targets_latest.mat`
  - `outputs/data/stage4_rd_four_targets_detection_latest.csv`
- Nature 风格二维图：
  - `outputs/figures/stage4_rd_four_targets_nature_2d.svg`
  - `outputs/figures/stage4_rd_four_targets_nature_2d.pdf`
  - `outputs/figures/stage4_rd_four_targets_nature_2d.png`
  - `outputs/figures/stage4_rd_four_targets_nature_2d.tiff`
- Nature 风格三维图：
  - `outputs/figures/stage4_rd_four_targets_nature_3d.svg`
  - `outputs/figures/stage4_rd_four_targets_nature_3d.pdf`
  - `outputs/figures/stage4_rd_four_targets_nature_3d.png`
  - `outputs/figures/stage4_rd_four_targets_nature_3d.tiff`

四目标设置：

| 目标 | 距离 m | 速度 m/s | alpha |
| --- | ---: | ---: | ---: |
| T1 | `25` | `-1` | `1.00` |
| T2 | `20` | `1` | `0.86` |
| T3 | `10` | `-1` | `0.74` |
| T4 | `5` | `1` | `0.62` |

检测结果：

| 目标 | random peak dB | optimized peak dB | 提升 dB | optimized 距离 m | optimized 速度 m/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | `-29.0675` | `-18.2766` | `10.7909` | `24.9` | `-1.0653` |
| T2 | `-30.3685` | `-19.5840` | `10.7845` | `20.1` | `1.0653` |
| T3 | `-31.6786` | `-20.8907` | `10.7879` | `9.9` | `-1.0653` |
| T4 | `-33.2032` | `-22.4240` | `10.7792` | `5.1` | `1.0653` |

验收结果：

- 四个目标的距离误差均约 `0.1 m`。
- 四个目标的速度误差均约 `0.0653 m/s`。
- 四个目标的 RD 峰值提升均约 `10.78 dB`。
- `G_ZF` 提升：`10.7866 dB`。
- MATLAB 验证状态：`PASS`。
- `tests/test_stage4_fmcw_rd.m`：`1 Passed, 0 Failed`。
- Python 绘图脚本通过 `py_compile`，并成功导出 `svg/pdf/png/tiff`。

## 2026-05-21 Stage 4.2 三维 RD 图论文化精简

- 修改脚本：`scripts/plot_stage4_nature_figures.py`
- 数据输入保持不变：`outputs/data/stage4_rd_four_targets_latest.mat`
- 二维主图保持输出：`outputs/figures/stage4_rd_four_targets_nature_2d.*`
- 新增三维 clean surface 输出：`outputs/figures/stage4_rd_four_targets_nature_3d_clean_surface.*`
- 新增三维 wireframe 输出：`outputs/figures/stage4_rd_four_targets_nature_3d_wireframe.*`

绘图调整：

- 对三维谱图使用低端动态范围软压缩：以 `vmax - 40 dB` 为参考下限，压窄更弱谱值的起伏而不删除噪声底。
- 降低三维采样密度并改用浅色 surface / wireframe，弱化噪声底的材质感和视觉干扰。
- clean surface 版本改为浅蓝单色曲面和细边线，不再使用彩色材质型 surface。
- wireframe 版本改为简洁蓝色线框。
- 三维图只保留红色峰值 `x` 标记，不再在三维图中重复 `T1` 至 `T4` 文字。
- 标题简化为 `Random RIS` 和 `Optimized RIS`。

执行验证：

- `python -m py_compile scripts/plot_stage4_nature_figures.py`：通过。
- `python scripts/plot_stage4_nature_figures.py`：通过，成功导出 `png/svg/pdf/tiff`。

当前建议：

- 论文主图仍优先使用二维 RD 复合图。
- 若正文需要三维谱图，优先采用 `clean_surface`。
- `wireframe` 更适合作为附图、补充材料或方法说明图。

## 2026-05-17 Stage 3.4 ZF-SNR 优化器公平稳定性验证

- 运行脚本：`main/main_stage3_optimizer_comparison.m`
- 最新输出日志：`outputs/logs/stage3_optimizer_comparison_20260517_194837.txt`
- 最新输出数据：`outputs/data/stage3_optimizer_comparison_20260517_194837.mat`
- 最新输出图：
  - `outputs/figures/stage3_optimizer_comparison.png`
  - `outputs/figures/stage3_optimizer_comparison.fig`
- trials：`30`
- numStarts：`3`
- 主要目标：`zf_snr`
- 公平性设置：每个 trial 中 `random_best_of_numStarts`、`fixed_grid_zf_snr`、`coarse_to_fine_zf_snr`、`coarse_to_fine_zf_snr_with_condition_penalty` 使用同一组 `startPhases`。

统计结果摘要：

| 方法 | mean SNR dB | median SNR dB | mean improvement vs random single | mean improvement vs random best | failure vs random single | failure vs random best | mean cond | mean ZF raw power | mean path gain | mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_single | `-85.640` | `-84.887` | `0` | `-4.187` | `30` | `30` | `19.191` | `1.164e10` | `2.184e-07` | `0` |
| random_best_of_numStarts | `-81.453` | `-81.401` | `4.187` | `0` | `11` | `30` | `8.604` | `6.557e08` | `2.292e-07` | `0` |
| fixed_grid_zf_snr | `-72.694` | `-72.811` | `12.945` | `8.758` | `0` | `0` | `2.181` | `7.747e07` | `3.114e-07` | `0.492 s` |
| coarse_to_fine_zf_snr | `-72.591` | `-72.579` | `13.049` | `8.862` | `0` | `0` | `2.146` | `7.501e07` | `3.139e-07` | `0.913 s` |
| coarse_to_fine_zf_snr_with_condition_penalty | `-72.568` | `-72.534` | `13.072` | `8.885` | `0` | `0` | `2.122` | `7.456e07` | `3.127e-07` | `0.956 s` |

验收结论：

- `coarse_to_fine_zf_snr` 平均 SNR 高于 `random_single`，且 failure count 为 `0`。
- `coarse_to_fine_zf_snr` 平均 SNR 高于同 numStarts 的 `random_best_of_numStarts`，且 failure count 为 `0`。
- `coarse_to_fine_zf_snr_with_condition_penalty` 在本轮统计中平均 SNR、平均条件数和平均 ZF raw power 略优，但优势很小，暂定为“候选主算法”，后续还需在不同 `Nr` 和功率扫描下复核。
- `fixed_grid_zf_snr` 运行更快，平均 SNR 略低，可作为快速诊断基线。
- `random_best_of_numStarts` 的 `failureCountVsRandomBest = 30` 是因为该列把“相对自身提升 <= 0”计为失败，对基线自身没有算法判别意义。

## 2026-05-17 Stage 3.3 ZF-SNR 稳定性测试

- 运行脚本：`main/main_stage3_zf_snr_stability.m`
- 输出日志：`outputs/logs/stage3_zf_snr_stability_20260517_180009.txt`
- 输出数据：`outputs/data/stage3_zf_snr_stability_20260517_180009.mat`
- 输出图：
  - `outputs/figures/stage3_zf_snr_stability.png`
  - `outputs/figures/stage3_zf_snr_stability.fig`

统计结果：

- trials: `8`
- failure count (`improvement <= 0 dB`): `0`
- mean improvement: `13.8389 dB`
- median improvement: `13.7255 dB`
- min improvement: `6.8751 dB`
- max improvement: `19.303 dB`
- best-so-far endpoint matches final best: `true`

结论：

- `objective_zf_snr` 在 8 个随机信道上均显著优于 random phase。
- 优化后 `cond(Heff)` 和 ZF raw power 整体明显下降。
- 部分 trial 中 path gain 不一定增加，但 SNR 增加，这进一步说明 path gain 不是当前主目标。

注意：日志表格中的 seed 使用 MATLAB 默认显示格式，显示为科学计数法，看起来相近；实际 seed 已保存在 `.mat` 文件的 `trialTable` 中。

## 2026-05-17 第三阶段目标统一诊断

- 运行脚本：`main/main_stage3_admm_validation.m`
- 输出日志：`outputs/logs/stage3_admm_validation_20260517_173118.txt`
- 输出数据：`outputs/data/stage3_admm_validation_20260517_173118.mat`
- 输出图：
  - `outputs/figures/stage3_objective_diagnostics.png`
  - `outputs/figures/stage3_objective_diagnostics.fig`

对比结果：

| 方法 | 优化目标 | path gain | SNR dB | cond(Heff) | ZF raw power |
| --- | --- | ---: | ---: | ---: | ---: |
| random | none | `8.9838e-08` | `-82.614` | `5.2381` | `7.3023e+08` |
| quadratic_admm | quadratic_trace_proxy | `4.1721e-07` | `-86.593` | `24.906` | `1.8253e+09` |
| objective_path_gain | path_gain | `5.4459e-07` | `-85.225` | `24.86` | `1.3323e+09` |
| objective_zf_snr | zf_snr | `1.2813e-07` | `-76.395` | `2.2125` | `1.7442e+08` |

诊断结论：

- quadratic ADMM 的代理目标改善，但真实 ZF-SNR 变差，说明代理目标与工程目标不一致。
- path_gain 优化器显著提高 path gain，但同样降低 ZF-SNR，说明单纯增大 `||Heff||_F^2` 不适合作为后续 SNR 曲线优化目标。
- zf_snr 目标驱动优化器显著提高 ZF-SNR，当前最适合后续图3/图4相关 SNR 实验。

## 2026-05-17 第三阶段 ADMM 整改验证

- 运行脚本：`main/main_stage3_admm_validation.m`
- 输出日志：`outputs/logs/stage3_admm_validation_20260517_170010.txt`
- 输出数据：`outputs/data/stage3_admm_validation_20260517_170010.mat`
- 收敛图：
  - `outputs/figures/stage3_admm_convergence.png`
  - `outputs/figures/stage3_admm_convergence.fig`

算法状态：

- `optimize_ris_admm.m` 方法：`quadratic_admm_approximation`
- 是否使用有限差分梯度：`false`
- surrogate 对照函数：`optimize_ris_surrogate.m`

三者对比：

| phase | path gain | SNR dB | 单位模误差 |
| --- | ---: | ---: | ---: |
| random | `8.9838e-08` | `-82.614` | 约 `1e-16` |
| quadratic ADMM approximation | `9.0555e-08` | `-82.5793` | `1.1102e-16` |
| finite-difference surrogate | `1.1582e-07` | `-82.3683` | `1.1102e-16` |

ADMM residual：

- final primal residual: `3.6459e-09`
- final dual residual: `9.6119e-05`
- iterations: `500`
- converged: `false`
- rho: `1`

结论：

- ADMM 后 path gain 不低于 random。
- ADMM 后 SNR 不低于 random。
- surrogate 结果优于当前二次型 ADMM 近似，但 surrogate 不是论文 ADMM，不能冒充正式 ADMM。
- 当前不建议直接进入 CD 或图3/图4，应继续修正 ADMM 目标与当前物理模型的一致性。

## 2026-05-17 Stage 3 ADMM Validation

- Script: `main/main_stage3_admm_validation.m`
- Random seed: `20251012`
- Dimensions:
  - `Hsr = [16 4]`
  - `Hrd = [16 16]`
  - `vRandom = [16 1]`
  - `vAdmm = [16 1]`
- Unit-modulus max error: `1.1102e-16`
- Path gain:
  - random: `8.9838e-08`
  - ADMM: `1.1582e-07`
  - improvement: `1.1034 dB`
- ZF-normalized SNR:
  - random: `-82.614 dB`
  - ADMM: `-82.3683 dB`
  - improvement: `0.24576 dB`
- ADMM settings:
  - method: `stage3_projected_phase_admm_surrogate`
  - iterations: `50`
  - converged: `false`
  - rho: `1`
  - gradientStep: `0.005`
- Output log: `outputs/logs/stage3_admm_validation_20260517_161538.txt`
- Output data: `outputs/data/stage3_admm_validation_20260517_161538.mat`
- Output figures:
  - `outputs/figures/stage3_admm_convergence.png`
  - `outputs/figures/stage3_admm_convergence.fig`
- Acceptance result: script ran successfully; unit-modulus constraint passed; ADMM path gain and ZF SNR were both not lower than the random phase baseline; convergence figures were saved.

## 2026-05-17 Stage 3 Final Verification Run

- Script: `main/main_stage3_admm_validation.m`
- Purpose: final verification after Code Analyzer cleanup.
- Random seed: `20251012`
- Unit-modulus max error: `1.1102e-16`
- Random path gain: `8.9838e-08`
- ADMM path gain: `1.1582e-07`
- Gain improvement: `1.1034 dB`
- Random SNR: `-82.614 dB`
- ADMM SNR: `-82.3683 dB`
- SNR improvement: `0.24576 dB`
- ADMM iterations: `50`
- ADMM converged: `false`
- Output log: `outputs/logs/stage3_admm_validation_20260517_161902.txt`
- Output data: `outputs/data/stage3_admm_validation_20260517_161902.mat`
- Output figures:
  - `outputs/figures/stage3_admm_convergence.png`
  - `outputs/figures/stage3_admm_convergence.fig`
- Acceptance result: passed.

本文件记录每次实验运行结果。每次运行 `main/` 下脚本后必须追加记录。

## 记录模板

```text
日期：
运行脚本：
Git/文件状态：
参数设置：
输出数据：
输出图表：
运行结果摘要：
是否符合论文趋势：
异常现象：
后续动作：
```

## 2026-05-17 工程初始化

- 运行脚本：无。
- 结果：仅创建项目结构和文档占位，未运行仿真实验。
- 输出：无实验图表或数据。

## 2026-05-17 Stage 2 Model Validation

- 运行脚本：`main/main_stage2_model_validation.m`
- 参数入口：`config/paper_params.m`
- 随机种子：`20251009`
- 采用维度：
  - `Hsr = [16 4]`
  - `Hrd = [16 16]`
  - `Phi = [16 16]`
  - `Heff = [4 4]`
  - `B = [4 4]`
- ZF 结果：
  - raw power: `7.302333e+08 W`
  - normalized power: `1.000000e-02 W`
  - relative error: `6.991189e-16`
- 名义 SNR：
  - linear: `5.477701e-09`
  - dB: `-82.614017 dB`
- 发射功率扫描：
  - power dBm: `[0 5 10 15 20]`
  - SNR dB: `[-92.614 -87.614 -82.614 -77.614 -72.614]`
  - 验收结果：SNR 随发射功率增大而增大。
- 噪声功率扫描：
  - noise dBm: `[-20 -10 0 10 20]`
  - SNR dB: `[-52.614 -62.614 -72.614 -82.614 -92.614]`
  - 验收结果：SNR 随噪声功率增大而减小。
- 输出日志：`outputs/logs/stage2_model_validation_20260517_153924.txt`
- 输出数据：`outputs/data/stage2_model_validation_20260517_153924.mat`
- 备注：当前绝对 SNR 很低，原因可能是阶段性路径损耗假设和 `noisePower_dBm = 10` 共同导致。第二阶段只将其作为模型一致性验证，不作为论文图3或图4数值复现结论。

## 2026-05-17 Stage 2 Final Verification Run

- 运行脚本：`main/main_stage2_model_validation.m`
- 运行目的：文档同步后进行最终验收验证。
- 结果摘要：
  - `Hsr = [16 4]`
  - `Hrd = [16 16]`
  - `Phi = [16 16]`
  - `Heff = [4 4]`
  - `B = [4 4]`
  - ZF normalized power: `0.01 W`
  - ZF relative error: `6.9912e-16`
  - nominal SNR: `-82.614 dB`
  - SNR vs power dB: `[-92.614 -87.614 -82.614 -77.614 -72.614]`
  - SNR vs noise dB: `[-52.614 -62.614 -72.614 -82.614 -92.614]`
- 输出日志：`outputs/logs/stage2_model_validation_20260517_154101.txt`
- 输出数据：`outputs/data/stage2_model_validation_20260517_154101.mat`
- 验收结论：维度、功率约束、发射功率单调性、噪声功率单调性均通过。
