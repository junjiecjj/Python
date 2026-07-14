# TODO

## Stage 4.6 状态：RIS 单元数与 ZF-SNR 增益扫描

### 已完成

- [x] 将 `N_RIS` 扫描主指标改为 ZF 输出 SNR 和 `G_ZF` 增益，不再用四目标 `Pd` 作为该实验主目标。
- [x] 新增 `main/main_stage4_snr_gain_vs_nris.m`，支持密集默认横轴和长 Monte Carlo 统计。
- [x] 新增 `tests/test_stage4_snr_gain_vs_nris.m` 并通过缩减轴 smoke test。
- [x] 输出逐 trial 打印、`.mat` 数据、`.png` 图和 `.txt` 日志。
- [x] 完成 `[4, 8, 12, 16]`、每点 `3` 次的保存输出验证。

### 下一步建议

- [x] 已运行默认正式扫描 `main_stage4_snr_gain_vs_nris()`，使用 `N_RIS = 4:4:64`、每点 `100` 次统计正式曲线。
- [x] 已检查正式曲线的 SNR 增益均值、方差带和运行时间趋势；当前全部 `N_RIS` 点平均增益为正，局部回落保留为 Monte Carlo 波动。
- [ ] 若大 `N_RIS` 运行时间过高，再决定是否并行化 trial 或把固定网格搜索参数拆成“诊断”和“正式”两档。

## Stage 4.5 完成状态

### 已完成

- [x] 新增 `main/main_stage4_pd_vs_snr.m`，支持 `quick` 和 `full` 模式。
- [x] 用 CA-CFAR 统计 No RIS、Random RIS、Fixed-grid optimized RIS 的四目标 Pd。
- [x] 同时保存 `echo SNR dB` 轴和 `echoNoisePower_W` 轴。
- [x] 输出 average Pd 曲线和每目标 Pd 曲线。
- [x] 为本地 full 模式加入逐 trial `fprintf` 进度表。
- [x] 新增 `tests/test_stage4_pd_vs_snr.m` 并完成 quick 模式 smoke test。
- [x] 实际运行 quick 模式并验证 optimized RIS 检测概率过渡区优于 random RIS。

### 下一步建议

- [ ] 本地运行 `full` 模式，把 Monte Carlo 次数提高到 `100` 或更多后保存正式 Pd 曲线。
- [ ] 根据 full 结果决定是否打开 `resampleChannelPerTrial` 做跨信道统计。
- [ ] 若需要固定虚警率下的更完整检测性能，可后续增加 Pd/Pfa 或 ROC 统计。

## Stage 4.4 完成状态

### 已完成

- [x] 新增 `functions/ca_cfar_2d.m`，在完整 RD 功率图上执行二维 CA-CFAR。
- [x] 新增 `functions/detect_rd_targets_cfar.m`，实现全图 CFAR 峰提取和真值邻域关联。
- [x] 新增 `tests/test_stage4_cfar_detection.m`，覆盖强峰检测和关联逻辑。
- [x] 在 `main_stage4_rd_detection.m` 中保留局部峰检测，并新增独立 `cfar` 命名输出。
- [x] 输出 CFAR MATLAB 快速检查图、CFAR Nature 2D/3D 图、`.mat` 和 `.csv` 数据。
- [x] 验证 random RIS 与 optimized RIS 四目标均可被 CFAR 关联命中。
- [x] 验证 optimized RIS 的 CFAR 目标峰值高于 random RIS。

### 下一步建议

- [ ] 增加 CFAR 参数扫描和多随机噪声/信道统计，避免只依赖单次受控样例。
- [ ] 若后续加入 direct-path 或杂波模型，重新评估 `Pfa`、训练窗和保护窗。

## Stage 4.3 完成状态

- [x] 增加无 RIS 对照组并保持 Stage 4 三类图同步。
- [x] 为 3D RD 图增加高度分层配色，避免整面单色。
- [x] 将 3D `clean_surface` 拆成三张命名单图，避免颜色条压住图 c。
- [ ] 若需要无 RIS 直达探测对照，新增 direct-path 回波模型。

## Stage 4 完成状态

### 已完成

- [x] 实现 `functions/compute_effective_channel.m`，统一封装 `Heff` 计算。
- [x] 实现 `functions/generate_fmcw_echo.m`，生成解调后复数 FMCW beat signal。
- [x] 实现 `functions/range_doppler_fft.m`，输出 RD 复谱、dB 图、距离轴和速度轴。
- [x] 新增 `main/main_stage4_rd_detection.m`，对比 random RIS 和 `fixed_grid_zf_snr` optimized RIS。
- [x] 新增 `tests/test_stage4_fmcw_rd.m`，验证单目标 RD 峰值接近真实距离和速度。
- [x] 生成 Stage 4 RD 对比图、日志和 `.mat` 数据。
- [x] 验证 optimized RIS 的 RD 目标峰值高于 random RIS。
- [x] 回归运行 Stage 2 和 Stage 3.3 核心脚本，未发现破坏。

### 当前不做

- [ ] 不实现 ADMM/CD。
- [ ] 不复现图3/图4。
- [x] 已完成当前四目标 RD 图和 CA-CFAR 检测分支。
- [ ] 不做 DOA、杂波或真实近场几何。

### 下一步建议

- [ ] 在单目标模型稳定后，再扩展到多目标 `targets.range_m`、`targets.velocity_mps`、`targets.alpha` 向量输入。
- [ ] 在进入图5/图6前，明确 RD 图使用的噪声、归一化、动态范围和目标幅度设定。
- [ ] 增加 `fixed_grid_zf_snr` 对 RD 峰值的多随机信道统计，而不只看单次信道。
- [ ] 若要与论文图6接近，需要定义不同 `Nr` 下的 RD 图对比方案。

## Stage 4.1 完成状态

### 已完成

- [x] 将 Stage 4 主脚本扩展为四目标 RD 验证。
- [x] 四目标分别进行局部峰值搜索，并输出检测表。
- [x] 保存 Python 绘图用 source data：`stage4_rd_four_targets_latest.mat`。
- [x] 保存四目标检测 CSV：`stage4_rd_four_targets_detection_latest.csv`。
- [x] 新增 Python/matplotlib Nature 风格绘图脚本。
- [x] 输出二维 Nature 风格复合图：`stage4_rd_four_targets_nature_2d.*`。
- [x] 输出三维 RD surface 图：`stage4_rd_four_targets_nature_3d.*`。
- [x] 验证四个目标峰值位置接近真实距离和速度。
- [x] 验证四个目标 optimized 峰值均高于 random，平均提升约 `10.79 dB`。

### 下一步建议

- [ ] 如果继续推进图5，可在当前四目标回波上增加距离-慢时间幅度谱。
- [ ] 如果继续推进图6，可在不同 `Nr` 下重复本四目标 RD 流程。
- [ ] 后续图件可继续使用 `scripts/plot_stage4_nature_figures.py` 的 Python 出图风格，但需要根据新数据拆分脚本或增加参数入口。

## Stage 4.2 完成状态

### 已完成

- [x] 保留二维 RD heatmap 主图不变。
- [x] 三维 RD 图加入 `vmax - 40 dB` 附近的低端软压缩，保留噪声底连续性和小起伏并弱化过度渲染。
- [x] 新增浅色 `clean_surface` 版本。
- [x] 新增 `wireframe` 版本。
- [x] 将三维标题简化为 `Random RIS` 和 `Optimized RIS`。
- [x] 导出两版三维图的 `png/svg/pdf/tiff`。

### 当前建议

- [ ] 正文若只保留一张主图，优先保留二维 RD 复合图。
- [ ] 若正文需要三维辅助图，优先选 `clean_surface`。
- [ ] 若附录需要更克制的三维谱形说明，使用 `wireframe`。

## Stage 3.4 完成状态

### 已完成

- [x] 在 `optimize_ris_objective_driven.m` 中增加 `searchMode = "fixed_grid"`。
- [x] 在 `optimize_ris_objective_driven.m` 中增加 `searchMode = "coarse_to_fine"`。
- [x] 支持 `objectiveType = "zf_snr"`。
- [x] 支持 `objectiveType = "zf_snr_with_condition_penalty"`。
- [x] 记录 best-so-far objective、SNR、path gain、condition number、ZF raw power。
- [x] 支持 `options.initialV` 传入多初值矩阵，保证 random-best 和优化器使用同一组初值。
- [x] 新增 `main/main_stage3_optimizer_comparison.m`。
- [x] 完成 30 个随机信道 trial 的公平稳定性验证。
- [x] 保存 `.mat` 数据、`.txt` 日志和综合统计图。
- [x] 明确记录 failure count、runtime、condition number 和 ZF raw power。

### 当前结论

- [x] `coarse_to_fine_zf_snr` 相比 `random_single` 和 `random_best_of_numStarts` 平均 SNR 均有明显提升。
- [x] `coarse_to_fine_zf_snr` 在 30 个 trial 中相对两个随机基线的 failure count 均为 `0`。
- [x] `coarse_to_fine_zf_snr_with_condition_penalty` 本轮平均 SNR 和平均条件数略优，可作为后续候选主算法，但需要在不同 `Nr` 和功率扫描下继续验证。

### 暂不进入

- [ ] 暂不复现图3。
- [ ] 暂不复现图4。
- [ ] 暂不做 RD 图。
- [ ] 暂不把 quadratic ADMM proxy 作为主结果算法。

### 下一步建议

- [ ] 在进入图3前，固定主算法选择：`coarse_to_fine_zf_snr` 或 `coarse_to_fine_zf_snr_with_condition_penalty`。
- [ ] 增加 `Nr` 扫描前的小规模稳定性测试，检查 `Nr` 改变后条件数和 ZF raw power 是否失控。
- [ ] 增加发射功率扫描前的小规模测试，确认优化后的 SNR 随功率保持单调。
- [ ] 后续正式曲线应提高 Monte Carlo 次数，并保存每个 trial 的失败样本用于 debug。

## Stage 3.3 完成状态

### 已完成

- [x] 修正多 start 历史记录。
- [x] 增加 best-so-far 曲线字段。
- [x] 让 best-so-far 曲线终点与最终 best result 一致。
- [x] 新增多随机种子稳定性测试脚本 `main_stage3_zf_snr_stability.m`。
- [x] 验证 `objective_zf_snr` 在 8 个随机信道下平均 SNR 提升为正。
- [x] 明确 failure count。
- [x] 输出 random 和 optimized 的 SNR、path gain、`cond(Heff)`、ZF raw power。

### 当前不做

- [ ] 不进入图3复现。
- [ ] 不进入图4复现。
- [ ] 不继续把 quadratic ADMM proxy 作为主结果算法。

### 下一步建议

- [ ] 将 seed 字段在日志中改为字符串，避免 MATLAB 表格科学计数法显示造成误读。
- [ ] 对比 `zf_snr` 与 `zf_snr_with_condition_penalty` 的多随机种子稳定性。
- [ ] 评估 coarse-to-fine phase search 或 Adam phase optimizer 是否能进一步提高 SNR 并降低运行时间。

## 第三阶段目标统一整改状态

### 已完成

- [x] 新增 `evaluate_ris_objective.m`，统一计算 `path_gain`、`zf_snr` 和带条件数惩罚的 ZF-SNR。
- [x] 修正 `optimize_ris_admm.m`，明确其只优化 `quadratic_trace_proxy`。
- [x] 对 `Q/T` 做尺度归一化，并按 `norm(T,2)` 设置 `rho`。
- [x] 新增 `optimize_ris_objective_driven.m`，直接优化工程目标。
- [x] 将 `main_stage3_admm_validation.m` 改为算法诊断脚本。
- [x] 输出并保存四类曲线：ADMM 代理目标、真实 path gain、ZF-SNR、条件数。
- [x] 证明 `objective_zf_snr` 相比 random 有明显 SNR 提升。

### 下一步建议

- [ ] 不建议继续围绕 quadratic ADMM proxy 做图3/图4。
- [ ] 若用户接受工程目标优先，应使用 `objective_zf_snr` 作为后续 SNR 曲线主算法。
- [ ] 若仍要论文 ADMM，需要重新定义物理模型，使论文 `T` 矩阵目标和工程目标一致。

## 当前 ADMM 整改状态

### 已完成

- [x] 正视并移除 `optimize_ris_admm.m` 中的 finite-difference phase-gradient 主体逻辑。
- [x] 按论文形式实现 `x/u/mu/rho` 闭式 ADMM 更新。
- [x] 构造 `(Nr+1) x (Nr+1)` 的二次型近似 `T` 矩阵。
- [x] 从 `x(1:Nr)/x(Nr+1)` 恢复 RIS 相位 `v`。
- [x] 新增 `optimize_ris_surrogate.m` 作为有限差分 surrogate 对照。
- [x] 在 `main_stage3_admm_validation.m` 中对比 random、ADMM、surrogate。
- [x] 验证 ADMM 单位模约束、path gain、SNR、primal residual 和 dual residual。

### 暂不进入

- [ ] 暂不实现 CD。
- [ ] 暂不复现图3。
- [ ] 暂不复现图4。

### 需要继续修正

- [ ] 当前 ADMM 是 `quadratic_admm_approximation`，还不是严格论文 ADMM。
- [ ] 需要重新审查论文中 `Hrd`、目标散射矩阵和 `T` 的推导，决定是否调整当前 `Hrd = Nr x Nr` 工程模型。
- [ ] 需要研究如何让 ADMM 优化目标同时服务于 `||Heff||_F^2` 和 ZF 后 SNR，而不是只优化 `trace(Heff)` 的二次代理。

## Stage 3 Status

### Completed

- [x] Confirmed and reused current dimensions: `Hsr: Nr x Nt`, `Hrd: Nr x Nr`, `Phi: Nr x Nr`, `Heff: Nt x Nt`, `B: Nt x Nt`, `v: Nr x 1`.
- [x] Added `functions/compute_path_gain.m`.
- [x] Implemented `functions/optimize_ris_admm.m` as a projected/proximal ADMM surrogate.
- [x] Added and ran `main/main_stage3_admm_validation.m`.
- [x] Verified unit-modulus constraint for `v_admm`.
- [x] Verified ADMM path gain is not lower than random phase.
- [x] Verified ADMM ZF-normalized SNR is not lower than random phase.
- [x] Saved ADMM convergence curve as `.png` and `.fig`.
- [x] Saved Stage 3 validation log and data.
- [x] Updated project documents.

### Not Done

- [ ] `optimize_ris_cd.m` is still not implemented.
- [ ] Fig. 3, Fig. 4, Fig. 5, and Fig. 6 are still not reproduced.
- [ ] The implementation does not yet claim exact reproduction of the paper's closed-form `T`-matrix ADMM.

### Next Suggestions

- [ ] Add MATLAB unittest or small fixed-matrix tests for `compute_path_gain.m` and `optimize_ris_admm.m`.
- [ ] Derive whether the current `Hrd: Nr x Nr` model can produce a valid quadratic ADMM target, or whether the model should be changed before Fig. 3 reproduction.
- [ ] Before Fig. 3, define `N_r` sweep values, Monte Carlo count, and whether to optimize path gain, ZF SNR, or a safeguarded objective.

## 已完成

- [x] 阅读论文 PDF 并提取第一轮复现所需的核心信息。
- [x] 创建 MATLAB 项目目录结构。
- [x] 创建 `main/`、`config/`、`functions/`、`docs/`、`outputs/`。
- [x] 创建项目管理文档初版。
- [x] 创建 MATLAB 占位脚本和函数。

## 进行中

- [ ] 继续核对论文公式与 MATLAB 矩阵维度。

## 待完成

- [ ] 完善 `paper_params.m` 的参数字段和单位换算。
- [ ] 实现并验证 `generate_channels.m`。
- [ ] 实现并验证 `compute_snr.m`。
- [ ] 实现并验证 `design_precoder_zf.m`。
- [ ] 推导并实现 `optimize_ris_admm.m`。
- [ ] 明确 CD 算法细节并实现 `optimize_ris_cd.m`。
- [ ] 复现图3的 SNR vs RIS 单元数量曲线。
- [ ] 复现图4的 SNR vs 发射功率曲线。
- [ ] 实现 FMCW 多目标回波生成。
- [ ] 实现距离-多普勒 FFT。
- [ ] 复现图5多目标距离-时间三维幅度谱。
- [ ] 复现图6不同 RIS 单元数量下的距离-多普勒图。

## 需要人工确认

- [ ] 图3横轴 `N_r` 取值范围。
- [ ] 图4发射功率扫描范围。
- [ ] 是否严格采用三角扫频上下拍频，还是使用标准 chirp 序列 2D FFT 生成距离-多普勒谱。
- [ ] CD 算法是否需要严格复现引用文献[38]的 accelerated coordinate descent。
- [ ] 是否需要复现论文图中的绝对数值，还是先以趋势一致为阶段目标。

## Stage 2 状态

### 已完成

- [x] 完善 `config/paper_params.m`，加入线性功率字段和单位换算函数句柄。
- [x] 实现 `functions/generate_channels.m`，生成 `Hsr` 和 `Hrd` 并返回 `meta`。
- [x] 实现 `functions/compute_snr.m`。
- [x] 实现 `functions/design_precoder_zf.m`。
- [x] 新增并运行 `main/main_stage2_model_validation.m`。
- [x] 验证 `Hsr`、`Hrd`、`Phi`、`Heff`、`B` 维度。
- [x] 验证 ZF 预编码功率约束。
- [x] 验证 SNR 随发射功率增大而增大。
- [x] 验证 SNR 随噪声功率增大而减小。
- [x] 保存第二阶段验证日志和数据。
- [x] 同步更新项目文档。

### 下一阶段建议

- [ ] 增加小型单元测试脚本或 MATLAB unittest，覆盖单位换算、SNR 维度错误、ZF 功率约束。
- [ ] 审查 `Hrd` 的物理建模，决定是否引入目标散射向量和双程标量 RCS。
- [ ] 在实现 ADMM 前，先推导并验证路径增益目标函数与 `T` 矩阵的小尺寸等价性。
- [ ] 明确 Monte Carlo 设置，避免 `generate_channels.m` 在循环中固定生成同一个信道。
