# Project Architecture

## Stage 4.6：RIS 单元数与 ZF-SNR 增益扫描

本轮把 `N_RIS` 扫描从检测概率实验中拆出，改为直接观察当前主线算法在等效信道层的 ZF 输出能力。该实验不统计四目标 `Pd`，主指标为随机 RIS 与固定网格 ZF-SNR 优化 RIS 的输出 SNR 和 `G_ZF` 增益。

- `main/main_stage4_snr_gain_vs_nris.m`：新增正式扫描主函数。默认扫描 `N_RIS = 4:4:64`，每点 `100` 次 Monte Carlo，使用 `fixed_grid`、`objectiveType = "zf_snr"`、`phaseGridSize = 16`、`numStarts = 3`、`maxSweeps = 4`。
- `tests/test_stage4_snr_gain_vs_nris.m`：新增缩减轴 smoke test，检查结果结构、运行时统计字段和默认固定网格设置。
- `outputs/data/stage4_snr_gain_vs_nris_*.mat`：保存每个 `N_RIS` 与 trial 的随机/优化 ZF-SNR、`G_ZF`、条件数、ZF raw power、运行时间和汇总曲线。
- `outputs/figures/stage4_snr_gain_vs_nris_*.png`：保存 ZF 输出 SNR、SNR 增益、`G_ZF` 增益和优化器运行时间随 `N_RIS` 变化的四面板图。
- `outputs/logs/stage4_snr_gain_vs_nris_*.txt`：保存逐 trial 命令行进度和统计摘要，便于正式长跑时观察趋势。

该脚本仅复用已有 `generate_channels.m`、`evaluate_ris_objective.m` 和 `optimize_ris_objective_driven.m`，没有改动 Stage 1-4 的 RD 与 Pd-vs-SNR 主流程。

## Stage 4.5：CA-CFAR 检测概率 Pd-vs-SNR

本轮新增 Stage 4 Monte Carlo 检测概率脚本，用于比较 NLOS `No RIS`、`Random RIS` 和 `Fixed-grid ZF-SNR optimized RIS` 在不同回波 SNR 下的四目标 CA-CFAR 检测概率。

- `main/main_stage4_pd_vs_snr.m`：新增 quick/full 双模式主函数。它复用四目标 FMCW 回波、RD FFT、固定网格 ZF-SNR RIS 优化器和 `detect_rd_targets_cfar.m`，输出每目标 Pd、平均 Pd、进度打印、数据、图和日志。
- `tests/test_stage4_pd_vs_snr.m`：新增 quick smoke test，检查 Pd 输出结构、SNR 轴、目标维度和 CFAR 参数。
- `outputs/data/stage4_pd_vs_snr_*.mat`：保存 SNR 轴、噪声功率轴、命中计数、每目标 Pd、平均 Pd、RIS 增益和运行配置。
- `outputs/figures/stage4_pd_vs_snr_*.png`：保存 average Pd 曲线和 random/optimized 每目标 Pd 曲线。
- `outputs/logs/stage4_pd_vs_snr_*.txt`：保存命令行进度行和 quick/full 汇总结果。

当前主横轴定义为 optimized RIS 无噪声 beat signal 平均样本功率参考下的 `echo SNR dB`；日志和 `.mat` 同时保留每个点对应的 `echoNoisePower_W`。

## Stage 4.4：全图 CA-CFAR 检测与真值邻域关联

本轮在保留 Stage 4 局部峰值检测输出的基础上，增加全图 CA-CFAR 检测链路。CA-CFAR 先在完整 RD 功率图上产生检测单元和局部极大值，再把检测峰与四个真值目标邻域关联；关联失败时保留未命中状态，不用真值邻域局部峰替代 CFAR 结果。

新增或修改文件：

- `functions/ca_cfar_2d.m`：在线性 RD 功率域实现二维 CA-CFAR，输出检测掩膜、门限图、噪声估计图和训练单元元数据。
- `functions/detect_rd_targets_cfar.m`：封装“全图 CFAR 检测 + CFAR 峰提取 + 真值邻域关联”流程，输出每个目标的关联结果和全图候选峰。
- `tests/test_stage4_cfar_detection.m`：验证强目标单元可被 CA-CFAR 检出，并验证关联逻辑只使用全图 CFAR 峰。
- `main/main_stage4_rd_detection.m`：保留原有局部峰值检测，同时增加无 RIS、random RIS、optimized RIS 三组 CFAR 检测结果、CFAR PASS/FAIL 日志和 `cfar` 命名输出。
- `scripts/plot_stage4_nature_figures.py`：读取 `stage4_rd_four_targets_cfar_latest.mat`，导出与现有 Stage 4 风格同步的 CFAR Nature 2D/3D 图。
- `outputs/data/stage4_rd_four_targets_cfar_latest.mat` 与 `stage4_rd_four_targets_cfar_detection_latest.csv`：保存 CFAR 绘图源数据和目标关联表。
- `outputs/figures/stage4_rd_detection_cfar_*.png/.fig` 与 `stage4_rd_four_targets_cfar_nature_*`：保存 MATLAB CFAR 快速检查图和 Python 图件。

当前 CFAR 分支用于验证四目标 RD 检测性能，不改变 Stage 4 的 FMCW 回波模型、RIS 增益链路和原有局部峰值图件。

## Stage 4.3：三组 RD 图形链路更新

- `main/main_stage4_rd_detection.m` 同步输出无 RIS、随机相位 RIS、固定网格 ZF-SNR 优化 RIS 三组四目标 RD 数据。
- `scripts/plot_stage4_nature_figures.py` 同步导出三组 2D heatmap、单面板 3D `clean_surface` 和三联 3D `wireframe` 图。
- 3D 图采用低饱和度高度分层配色，保留噪声底起伏，同时用红色峰值标记突出 random/optimized 目标峰。
- 3D `clean_surface` 按 `no_ris`、`random_ris`、`optimized_ris` 分成三个命名清楚的单图导出，颜色条跟随单图放在右侧，避免三联 3D 轴外扩后压图。

## Stage 4：FMCW 回波模型与距离-多普勒检测验证

本阶段新增单目标 FMCW beat signal 生成、距离-多普勒 FFT 和 RD 图检测验证。当前主线 RIS 相位优化器只使用 `fixed_grid_zf_snr`，不继续推进 ADMM/CD，也不复现图3/图4。

新增或修改文件：

- `config/paper_params.m`：新增 `params.radar.numFastTimeSamples = round(sampleRate * chirpTime)`；修正阶段性速度分辨率字段为 `lambda / (2 * Nchirp * Tchirp)`。
- `functions/compute_effective_channel.m`：新增等效信道封装，统一计算 `Phi = diag(v)` 和 `Heff = Hsr' * Phi * Hrd * Phi' * Hsr`。
- `functions/compute_path_gain.m`：改为复用 `compute_effective_channel.m`，避免重复写等效信道公式。
- `functions/generate_fmcw_echo.m`：实现解调后 FMCW beat signal 模型，输出 `Nfast x Nchirp` 复数回波矩阵。
- `functions/range_doppler_fft.m`：实现 range FFT 和 Doppler FFT，Doppler 维使用 `fftshift`，输出 `RD_complex`、`RD_dB`、`rangeAxis`、`velocityAxis`。
- `main/main_stage4_rd_detection.m`：新增 Stage 4 主脚本，对比 random RIS 和 `fixed_grid_zf_snr` optimized RIS 的 RD 图与目标峰值。
- `tests/test_stage4_fmcw_rd.m`：新增轻量 MATLAB 单元测试，验证单目标 RD 峰值位置接近真实距离和速度。
- `outputs/figures/stage4_rd_detection_*.png/.fig`：保存 random RIS、optimized RIS 和峰值柱状图。
- `outputs/logs/stage4_rd_detection_*.txt`：保存峰值检测和 PASS/FAIL 日志。
- `outputs/data/stage4_rd_detection_*.mat`：保存回波、RD 谱、坐标轴、检测结果和参数。

Stage 4 基础链路已从单目标扩展到四目标，并增加局部峰值检测与 CA-CFAR 检测分支。当前仍无 DOA、无杂波、无真实近场几何；后续若扩展到图5/图6，需要继续明确检测门限统计、目标幅度设定和物理几何建模。

## Stage 4.1：四目标 RD 验证与 Nature 风格图

本轮将 Stage 4 主脚本从单目标扩展为四目标验证，并新增 Python 绘图脚本：

- `main/main_stage4_rd_detection.m`：目标改为四个点，`R = [25, 20, 10, 5] m`，`v = [-1, 1, -1, 1] m/s`，并为每个目标分别做局部峰值检测。
- `scripts/plot_stage4_nature_figures.py`：读取 `outputs/data/stage4_rd_four_targets_latest.mat`，用 Python/matplotlib 输出 Nature 风格二维复合图和三维 RD surface 图。
- `outputs/data/stage4_rd_four_targets_latest.mat`：Python 绘图 source data。
- `outputs/data/stage4_rd_four_targets_detection_latest.csv`：四目标检测结果表。
- `outputs/figures/stage4_rd_four_targets_nature_2d.*`：二维 RD map + 峰值恢复 + 提升柱状图，输出 `svg/pdf/png/tiff`。
- `outputs/figures/stage4_rd_four_targets_nature_3d.*`：三维 RD surface 对比图，输出 `svg/pdf/png/tiff`。

图形逻辑：

- 二维图是主证据：展示 random RIS 和 fixed-grid ZF-SNR RIS 的四目标 RD map，并量化每个目标的局部峰值提升。
- 三维图是辅助视觉证据：展示优化前后四个谱峰的表面形态，不作为定量结论的唯一依据。

## Stage 4.2：三维 RD 图版本约定

`scripts/plot_stage4_nature_figures.py` 当前保留二维 RD 主图，同时导出两类三维图：

- `stage4_rd_four_targets_nature_3d_clean_surface.*`：浅蓝单色峰区曲面，适合正文中需要 3D 辅助展示时使用。
- `stage4_rd_four_targets_nature_3d_wireframe.*`：简洁线框峰区图，适合补充材料、方法说明或比较不同渲染方式。

三维图统一使用 `vmax - 40 dB` 附近的低端动态范围压缩和降采样。低于该参考下限的谱值做软压缩，保留连续噪声底及其小起伏；渲染重点放在目标峰区，避免低幅噪声底被绘成高材质感地形表面。

## Stage 3.4：ZF-SNR 驱动优化器强化与公平稳定性验证

本阶段继续保持当前工程结构，不进入 RD 图、图3或图4复现。主线目标明确为 `objectiveType = "zf_snr"`，即直接优化 ZF 预编码归一化后的 SNR。`path_gain` 只作为辅助观察指标，`quadratic ADMM proxy` 只作为学习和诊断模块，不作为后续 SNR 曲线主算法。

本阶段新增或修改文件：

- `functions/optimize_ris_objective_driven.m`：增加 `searchMode` 参数，支持 `"fixed_grid"` 和 `"coarse_to_fine"`。`coarse_to_fine` 采用三层坐标相位搜索：全局粗搜、局部细搜、更小范围细搜；支持多 start、多 sweep、early stop，并记录 best-so-far 历史曲线。该函数现在允许 `options.initialV` 为 `Nr x K` 初值矩阵，用于公平复用同一组随机初值。
- `main/main_stage3_optimizer_comparison.m`：新增 Stage 3.4 算法对照脚本。每个 trial 使用同一组 `Hsr/Hrd` 和同一组 start phases，对比 `random_single`、`random_best_of_numStarts`、`fixed_grid_zf_snr`、`coarse_to_fine_zf_snr`、`coarse_to_fine_zf_snr_with_condition_penalty`。
- `outputs/logs/stage3_optimizer_comparison_*.txt`：保存 30 次随机信道对照统计日志。
- `outputs/data/stage3_optimizer_comparison_*.mat`：保存 `resultTable` 和 `summaryTable`。
- `outputs/figures/stage3_optimizer_comparison.png/.fig`：保存 SNR、improvement、condition number、runtime、SNR-vs-cond、ZF raw power 的综合诊断图。

当前主算法候选：

- 首选候选：`coarse_to_fine_zf_snr_with_condition_penalty`。在 30 次 trial 的最新公平对照中，它平均 SNR、平均条件数和平均 ZF raw power 略优，但相对 `coarse_to_fine_zf_snr` 的优势很小。
- 稳健候选：`coarse_to_fine_zf_snr`。不引入条件数惩罚，目标最直接，且相对 `random_single` 和 `random_best_of_numStarts` 的 failure count 均为 0。
- 快速基线：`fixed_grid_zf_snr`。运行时间约为 coarse-to-fine 的一半，结果略低但仍明显优于随机相位。

后续进入图3/图4前，必须继续保持同一统计口径：每个方法在同一 trial 中共享同一信道和同一组随机初值，避免把多随机初值带来的收益误判为算法收益。

## Stage 3.3：ZF-SNR 主线优化与稳定性诊断

本轮在不复现图3/图4的前提下，强化 RIS 相位优化诊断链路：

- `functions/optimize_ris_objective_driven.m`：修正多 start 历史记录。现在每个 start 的初始点和每轮 sweep 都会记录，同时维护 `bestObjectiveHistory`、`bestPathGainHistory`、`bestSnrDbHistory`、`bestCondHistory`、`bestZfRawPowerHistory`。图像应使用 best-so-far 曲线，保证曲线终点与最终 best 结果一致。
- `main/main_stage3_admm_validation.m`：诊断图改为使用 best-so-far 历史，不再把不同 start 的普通 history 串接后误读为单条连续优化曲线。
- `main/main_stage3_zf_snr_stability.m`：新增多随机种子稳定性测试。该脚本只验证 `objectiveType = "zf_snr"` 是否在多个随机信道下稳定优于 random phase。

当前主线约定：

- 主优化目标：`zf_snr`
- 辅助观察指标：`path_gain`、`cond(Heff)`、`ZF raw power`
- quadratic ADMM：仅作为代理目标学习和诊断模块，不作为后续 SNR 曲线主算法

## 第三阶段目标统一整改

本轮新增并调整以下文件：

- `functions/evaluate_ris_objective.m`：统一 RIS 相位目标评估，支持 `path_gain`、`zf_snr`、`zf_snr_with_condition_penalty`。后续优化器必须明确自己优化哪一个目标。
- `functions/optimize_ris_admm.m`：保留为 `quadratic_admm_approximation`，只优化自己的二次代理目标 `real(v^H Q v)`，不再声称优化真实 path gain 或 ZF-SNR。已加入 `Q/T` 归一化、`rho` 按 `norm(T,2)` 设置，并记录真实 path gain、ZF-SNR 和条件数曲线。
- `functions/optimize_ris_objective_driven.m`：新增工程目标驱动优化器，采用多初值坐标相位搜索，默认优化 `zf_snr`，保证单位模约束，并逐轮记录目标、path gain、SNR、条件数和更新数。
- `main/main_stage3_admm_validation.m`：改为算法诊断脚本，对比 random、quadratic ADMM approximation、objective-driven path_gain、objective-driven zf_snr，不再用单一 “passed” 掩盖结果。

当前诊断结论：后续若目标是复现 SNR 曲线，优先考虑 `optimize_ris_objective_driven(..., "zf_snr", ...)`，而不是 quadratic ADMM proxy。

## 第三阶段整改：ADMM 算法结构修正

本轮将 `functions/optimize_ris_admm.m` 从有限差分相位梯度 surrogate 改为闭式 ADMM 更新结构。当前实现不再使用 finite-difference phase-gradient 作为主体优化逻辑。

当前 `optimize_ris_admm.m` 的方法标识为 `quadratic_admm_approximation`。它使用论文形式的扩展变量：

- `x`: `(Nr+1) x 1`
- `u`: `(Nr+1) x 1`，满足单位模约束
- `mu`: `(Nr+1) x 1`
- `T`: `(Nr+1) x (Nr+1)`

更新公式为：

```text
u = exp(1j * angle(x - mu/rho))
x = (rho*I + T)^(-1) * (rho*u + mu)
mu = mu + rho*(u - x)
```

需要明确的是，当前工程模型采用 `Hrd = Nr x Nr`，而真实路径增益 `||Heff||_F^2` 对相位 `v` 是四次函数，不能直接写成论文中的二次 `x^H T x`。因此当前 `T` 来自 `trace(Heff)` 的 Hermitian 二次代理：

```text
Q = (Hsr*Hsr') .* transpose(Hrd)
Qh = (Q + Q')/2
T(1:Nr,1:Nr) = -Qh
T(Nr+1,Nr+1) = 0
```

新增 `functions/optimize_ris_surrogate.m`，只作为有限差分相位 surrogate 的显式对照，不再作为正式 ADMM。

## Stage 3 Update: RIS Phase ADMM Validation

This stage added or modified:

- `functions/compute_path_gain.m`: executable path-gain objective under the current matrix convention, `gain = ||Heff||_F^2`, with `Heff = Hsr^H * Phi * Hrd * Phi^H * Hsr`.
- `functions/optimize_ris_admm.m`: projected/proximal ADMM surrogate for RIS phase optimization. Because the current `Hrd: Nr x Nr` convention makes `||Heff||_F^2` quartic in `v`, the implementation keeps the ADMM-style `x/u/mu/rho` consensus projection structure but uses finite-difference phase-gradient backtracking for the surrogate `x` step.
- `main/main_stage3_admm_validation.m`: validation script for unit-modulus phases, random-vs-ADMM path gain, random-vs-ADMM ZF SNR, and convergence output.
- `outputs/figures/stage3_admm_convergence.png` and `.fig`: ADMM path-gain convergence curve.
- `outputs/logs/stage3_admm_validation_*.txt`: Stage 3 validation logs.
- `outputs/data/stage3_admm_validation_*.mat`: Stage 3 validation data.

Still not implemented in Stage 3: `optimize_ris_cd.m`, Fig. 3, Fig. 4, Fig. 5, and Fig. 6 reproduction.

## 项目定位

本项目不是单脚本复现，而是面向长期调试和扩展的 MATLAB 科研复现工程。目标是逐步复现论文《RIS辅助MIMO-FMCW雷达的非视距目标参数估计方法》中的 RIS 辅助 MIMO-FMCW 雷达非视距目标参数估计方法和仿真实验。

## 目录结构

```text
RIS_MIMO_FMCW_Reproduction/
├─ main/
├─ config/
├─ functions/
├─ docs/
├─ outputs/
│  ├─ figures/
│  ├─ data/
│  └─ logs/
└─ README.md
```

## 目录职责

`main/` 存放可直接运行的主脚本。每个主脚本对应一个明确实验或图表复现任务，不能把多个无关功能混写到同一个脚本中。

`config/` 集中保存论文参数和仿真参数。后续所有主脚本和函数应优先从 `config/paper_params.m` 读取参数，避免在多个脚本中重复硬编码。

`functions/` 存放可复用 MATLAB 函数。每个函数只负责一个明确功能，例如信道生成、ZF 预编码、RIS 相移优化、SNR 计算、FMCW 回波生成、距离-多普勒处理和绘图风格管理。

`docs/` 是项目管理核心目录。后续每轮代码修改、公式理解、debug、假设调整和实验结果都必须记录到对应文档。

`outputs/figures/` 保存复现图表，例如 `.png`、`.fig`、`.pdf`。

`outputs/data/` 保存中间数据，例如 `.mat` 文件、SNR 曲线数据、距离-多普勒谱矩阵。

`outputs/logs/` 保存脚本运行日志和必要的文本输出。

## 主脚本规划

- `main_reproduce_all.m`：一键运行全部复现实验。后期在各子实验稳定后再完善。
- `main_fig3_snr_vs_Nris.m`：复现图3，比较 ADMM 与 CD 算法的 SNR 随 RIS 反射单元数量变化曲线。
- `main_fig4_snr_vs_power.m`：复现图4，比较不同 RIS 反射单元数量下 ADMM 与 CD 算法的 SNR 随发射功率变化曲线。
- `main_fig5_range_time_3d.m`：复现图5，多目标距离-时间三维幅度谱。
- `main_fig6_range_doppler_maps.m`：复现图6，不同 RIS 反射单元数量下 ADMM 和 CD 算法的距离-多普勒图。

## 函数规划

- `generate_channels.m`：生成 RIS 辅助雷达系统中的 `H_sr` 和 `H_rd` 信道矩阵。
- `design_precoder_zf.m`：根据等效信道设计迫零预编码矩阵 `B`，并进行功率归一化。
- `optimize_ris_admm.m`：根据论文中的 ADMM 更新公式优化 RIS 相移向量 `v` 或相移矩阵 `Phi`。
- `optimize_ris_cd.m`：实现 CD 坐标下降算法，作为 ADMM 的对比基线。
- `compute_snr.m`：根据等效信道、预编码矩阵和噪声功率计算 SNR。
- `generate_fmcw_echo.m`：生成多目标 FMCW 差拍信号，用于距离和速度估计。
- `range_doppler_fft.m`：对 FMCW 回波进行距离 FFT 和多普勒 FFT，生成距离-多普勒谱。
- `plot_utils.m`：统一图表格式，例如坐标轴、字体、图例和保存路径。

## 文档更新规范

1. 每次新增、删除或修改文件后，必须更新 `project_architecture.md`。
2. 每次修改核心公式或变量定义后，必须更新 `paper_formula_notes.md`。
3. 每次新增合理假设后，必须更新 `reproduction_assumptions.md`。
4. 每次运行实验后，必须更新 `experiment_log.md`。
5. 每次遇到错误并修复后，必须更新 `debug_log.md`。
6. 每次完成一个阶段任务后，必须更新 `todo.md`。
7. 不允许在没有说明的情况下大范围重构项目。
8. 不允许把多个功能混写在一个 `main` 脚本里。
9. 不允许为了贴合论文图表而硬编码实验结果。
10. 如果论文细节不足，必须明确标注为“合理复现假设”，不能说成“论文原文如此”。

## 当前阶段

第一轮仅完成工程骨架和文档初版。核心算法、FMCW 回波生成和图3-图6复现均未实现。

## Stage 2 Update: Base Model and Validation

本阶段新增或修改以下文件：

- `config/paper_params.m`：补充线性单位字段和单位换算函数句柄，包括 `dbm2w`、`w2dbm`、`db2pow`、`pow2db`。保留论文参数，并新增阶段性信道几何假设。
- `functions/generate_channels.m`：实现 Rician 信道和路径损耗生成。当前采用 `Hsr: Nr x Nt`、`Hrd: Nr x Nr` 的 RIS 域等效回波信道约定。
- `functions/design_precoder_zf.m`：实现基于 `pinv` 的 ZF 预编码，并按线性发射功率 `txPower_W` 归一化，使 `||B||_F^2 <= P`。
- `functions/compute_snr.m`：实现论文目标形式 `SNR = ||Heff * B||_F^2 / sigma^2`，其中 `sigma^2` 必须为线性功率。
- `main/main_stage2_model_validation.m`：新增第二阶段验证脚本，检查矩阵维度、ZF 功率约束、SNR 随发射功率和噪声功率的单调性，并保存日志和 `.mat` 数据。
- `outputs/logs/stage2_model_validation_*.txt`：保存第二阶段验证日志。
- `outputs/data/stage2_model_validation_*.mat`：保存第二阶段验证数据。

第二阶段仍未实现 `optimize_ris_admm.m`、`optimize_ris_cd.m`、FMCW 回波生成和图3-图6复现。
