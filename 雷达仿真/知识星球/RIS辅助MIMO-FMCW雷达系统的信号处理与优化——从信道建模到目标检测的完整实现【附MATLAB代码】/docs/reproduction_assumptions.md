# Reproduction Assumptions

## Stage 4.6：RIS 单元数增益扫描假设

1. `N_RIS` 扫描当前用于观察 ZF 归一化输出增益，而不是直接观察四目标 RD 检测概率。原因是四目标 `Pd` 在有限样本下容易被 CFAR 阈值、目标个数和噪声 trial 离散性主导，不能清楚表达 RIS 单元数对当前主线相位优化器的信道收益。
2. 每个 `N_RIS` 点都重新生成维度自洽的 `Hsr: Nr x Nt`、`Hrd: Nr x Nr` 和 RIS 相位向量，不把小尺寸信道补零或截断到大尺寸。
3. 每个 trial 内，随机相位和 fixed-grid 优化相位复用同一组 `Hsr/Hrd`；优化器初始相位集合包含被比较的随机相位，避免把不同信道和不同起点混入同一增益定义。
4. 默认正式扫描使用 `N_RIS = 4:4:64` 和每点 `100` 次 Monte Carlo。该取值是当前工程的统计与可运行性折中，不声称来自论文原文。
5. 当前仍保留带路径损耗的 Rician 等效信道模型，因此绝对 SNR 数值很低时，主要结论应看随机与优化相位在同一 trial 下的增益统计，而不把绝对 dB 值直接解释为论文图中物理链路预算。

## Stage 4.5：Pd-vs-SNR 合理复现假设

1. `Pd-vs-SNR` 主横轴使用 optimized RIS 无噪声 beat signal 平均样本功率作为参考回波功率。该定义便于在同一噪声环境下比较 random 与 optimized RIS 的系统检测收益。
2. quick 模式默认固定一组信道和 RIS 相位，仅在每个 Monte Carlo trial 中重采样回波噪声；它用于趋势验收，不替代正式统计。
3. full 模式默认 `100` 次 Monte Carlo，并预留 `resampleChannelPerTrial` 开关。若开启该开关，每个 trial 会重新生成信道、随机相位和固定网格 optimized RIS，相应噪声功率也按该 trial 的 optimized 参考回波重新换算。
4. 当前 `No RIS` 组仍是遮挡 NLOS 零目标回波基线，因此 Pd 理应最低，通常接近真值邻域内噪声 CFAR 峰关联概率。
5. 当前 Pd 统计采用 CA-CFAR 命中，不采用 Stage 4 旧局部峰值检测结果。

## Stage 4.4：CA-CFAR 检测合理复现假设

1. 当前检测分支采用二维 CA-CFAR，输入为 RD 复谱的线性功率 `abs(RD_complex).^2`，而不是 dB 图。
2. CFAR 训练窗、保护窗和 `Pfa` 先按当前四目标受控仿真调到可稳定工作：`trainingCells = [6,6]`、`guardCells = [2,2]`、`Pfa = 1e-4`。这些值属于当前工程检测假设，不是论文给定参数。
3. 真值距离和速度邻域只用于检测后关联、统计命中数和误差验收；CFAR 本身始终在完整 RD 图上运行。
4. 如果某个目标邻域内没有 CFAR 候选峰，代码保留 `hit = false` 和缺失峰值，不使用局部最大值回填为 CFAR 命中。
5. 当前 `No RIS` 组仍是遮挡 NLOS 零目标回波基线，因此其全图 CFAR 候选峰可能来自噪声，四个真值目标关联未命中属于当前模型下的合理结果。

## Stage 4.3：无 RIS 对照组合理复现假设

1. 当前 Stage 4 主场景仍是 RIS 辅助非视距探测，尚未加入独立 direct-path 回波链路。
2. 因此 `No RIS` 组暂按遮挡 NLOS 基线处理：目标回波幅度为零，只保留与 RIS 组匹配的 echo-domain 噪声实现。
3. 该组用于说明当前 NLOS RIS 链路是否让目标从噪声底中显现，不等价于“直达链路普通雷达”的物理对照。

## Stage 4：FMCW RD 检测合理复现假设

1. Stage 4 第一版从单目标 smoke validation 起步；当前后续分支已增加四目标图和 CA-CFAR 检测，但仍不引入 DOA、杂波、真实近场几何或 ADMM/CD。

2. 当前回波模型是解调后的复数 beat signal，不模拟完整发射 chirp、传播延迟、混频和低通滤波链路。该简化用于先验证距离-多普勒轴、目标峰值位置和 RIS/ZF 增益对 RD 峰值的影响。

3. `range_doppler_fft.m` 对复数 beat signal 保留完整 range FFT 频谱。若以后改成实采样或真实 ADC 模型，需要重新检查 range 轴和单边/双边谱处理。

4. Stage 4 主脚本使用受控 echo-domain 噪声功率 `echoNoisePower_W = 1e-12`。这是为了让单目标 RD 图在当前保守路径损耗和 Stage 2/3 噪声设定下可见；它不等同于论文绝对链路预算，也不用于声称论文图5/图6的数值复现。

5. 当前 optimized RIS 回波幅度使用 `A = sqrt(G_ZF) * alpha`，其中 `G_ZF = ||Heff * B||_F^2`。这与 Stage 3 主线 `fixed_grid_zf_snr` 的优化目标一致，避免用 path gain 替代 ZF 后工程目标。

6. 当前目标设置为 `R = 25 m`、`v = 3 m/s`、`alpha = 1`，用于单目标 smoke validation。后续多目标图需要另行定义目标列表、幅度归一化和检测规则。

## Stage 4.1：四目标绘图假设

1. 四目标验证沿用论文参数中的距离和速度组合：`R = [25, 20, 10, 5] m`，`v = [-1, 1, -1, 1] m/s`。

2. 为避免四个谱峰完全等高，当前使用经验散射系数 `alpha = [1.00, 0.86, 0.74, 0.62]`。这不是论文给定 RCS 参数，只是用于四目标图的可视化区分。

3. Nature 风格图使用 Python/matplotlib 生成，MATLAB 只负责仿真数据和检测结果输出。Python 图不重新计算雷达模型。

4. 二维复合图是当前主要结论图；三维 surface 图用于直观展示谱峰形态。三维图容易受视角、裁剪和透视影响，不建议作为定量比较唯一依据。

## Stage 3.4：ZF-SNR 主线优化假设

1. 当前工程把 `zf_snr` 作为 RIS 相位优化主目标，而不是把 `path_gain = ||Heff||_F^2` 作为主目标。原因是 Stage 3 诊断已经显示：path gain 增大可能导致 `Heff` 条件数变差，进而使 ZF 归一化后的 SNR 下降。

2. `quadratic ADMM approximation` 仅保留为论文 ADMM 思想的学习模块。由于当前工程维度和真实目标使 `||Heff||_F^2` 对 `v` 呈四次形式，现有二次代理目标不能代表真实 ZF-SNR 目标。

3. Stage 3.4 的主优化器采用坐标相位搜索，而不是严格论文 ADMM。该选择属于“合理工程复现假设”：先保证当前模型下真实工程目标能稳定改善，再考虑是否重新推导更贴近论文的闭式 ADMM。

4. 多随机种子稳定性验证中，每个 trial 使用不同信道种子。`generate_channels` 当前仍可根据 `params.repro.rngSeed` 重置随机数，但脚本会在每个 trial 设置不同 `rngSeed`，因此不会重复生成同一信道。

5. 为公平比较，Stage 3.4 中 `random_best_of_numStarts` 和所有优化器使用同一个 trial 内的同一组随机初值 `startPhases`。这样可以区分“多初值筛选带来的收益”和“相位优化本身带来的收益”。

6. `zf_snr_with_condition_penalty` 中的 `alpha = 0.05` 是当前阶段的经验参数，不是论文给定参数。最新 30 次 trial 显示它略优于纯 `zf_snr`，但优势较小，后续仍需在不同 `Nr`、发射功率和噪声设定下复核。

## Stage 3.3：后续算法主线假设

1. 后续 RIS 相位优化主线采用 `objectiveType = "zf_snr"`。
   - 这是当前工程模型下的工程目标。
   - `path_gain` 只作为辅助观察指标，不再作为 SNR 图表主结论。

2. 多 start 优化结果必须使用 best-so-far 曲线解释。
   - 不同 start 的普通 history 不能直接串接后当作单条连续优化轨迹。
   - 当前已记录 `bestSnrDbHistory` 等字段，终点与最终 best 结果一致。

3. 稳定性判断必须跨多个随机信道。
   - 单次随机结果不再作为算法有效性的充分证据。
   - 当前 Stage 3.3 使用 8 个随机信道种子测试 `zf_snr` 目标驱动优化器。

## 第三阶段目标统一后的结论

1. `quadratic_admm_approximation` 不是后续 SNR 曲线的首选算法。
   - 它能优化自己的二次代理目标。
   - 但本轮诊断显示，它会把真实 ZF-SNR 从 `-82.614 dB` 降到 `-86.593 dB`。

2. `path_gain` 和 `zf_snr` 是不同工程目标。
   - `objective_path_gain` 将 path gain 提高到 `5.4459e-07`，但 SNR 降到 `-85.225 dB`。
   - 原因是 `Heff` 条件数恶化，ZF 归一化后反而不利。

3. 当前最适合后续 SNR 曲线的算法是 `objective_zf_snr`。
   - 它直接优化 ZF 归一化后的线性 SNR。
   - 本轮诊断中 SNR 从 `-82.614 dB` 提升到 `-76.395 dB`。
   - 条件数从 `5.2381` 降到 `2.2125`，ZF raw power 从 `7.3023e+08` 降到 `1.7442e+08`。

4. 本轮不再把“ADMM 不低于 random”作为充分验收。
   - 必须同时说明优化目标、真实 path gain、ZF-SNR 和条件数。

## 第三阶段整改后的合理假设

1. 当前未能严格实现论文原始 `T` 矩阵。
   - 原因不是代码偷懒，而是当前工程采用 `Hrd = Nr x Nr` 后，真实目标 `||Heff||_F^2` 是四次目标。
   - 论文中的 `T` 矩阵推导更接近二次型路径增益表达，和当前 `||Heff||_F^2` 目标不完全一致。

2. 当前正式 ADMM 是 `quadratic_admm_approximation`。
   - 它使用论文形式的 `x/u/mu/rho` 和闭式 `x` 更新。
   - `T` 的维度是 `(Nr+1) x (Nr+1)`。
   - `T` 来自 `trace(Heff)` 的 Hermitian 二次代理，而不是来自 `||Heff||_F^2` 的精确等价变换。

3. 当前 `optimize_ris_admm.m` 不使用有限差分相位梯度。
   - `info.usesFiniteDifferenceGradient = false`。
   - 原有限差分方法保留为 `optimize_ris_surrogate.m`，只用于对照。

4. ADMM 的 residual 有意义但尚未收敛。
   - 最终验证中 primal residual 为 `3.6459e-09`。
   - dual residual 为 `9.6119e-05`。
   - `converged = false`，说明当前二次代理 ADMM 结构可运行，但收敛准则和目标一致性仍需继续修正。

## Stage 3 Assumptions Added

1. `compute_path_gain.m` uses `gain = ||Heff||_F^2`.
   - This is the executable objective under `Hsr: Nr x Nt` and `Hrd: Nr x Nr`.
   - It is not presented as the exact paper `T`-matrix quadratic objective.

2. `optimize_ris_admm.m` is a projected/proximal ADMM surrogate.
   - It keeps the `x/u/mu/rho` consensus-projection structure.
   - The `x` step uses finite-difference phase-gradient backtracking.
   - The closed-form paper update `(rho I + T)^(-1)(rho u + mu)` is not forced, because the current objective is quartic under the Stage-2 `Hrd` convention.

3. Stage 3 validation checks both path gain and ZF-normalized SNR.
   - Actual debugging showed that aggressive path-gain ascent can reduce ZF SNR by making `Heff` more ill-conditioned.
   - The validation script therefore uses `gradientStep = 0.005` and `maxIter = 50`.

4. `info.converged = false` is acceptable for Stage 3.
   - The stage acceptance criteria are successful script execution, unit-modulus phases, saved convergence curve, path gain not lower than random, and SNR not lower than random.
   - Formal convergence tuning is deferred until the figure-reproduction stage.

本文档只记录论文未明确给出、但代码复现必须补充的内容。凡是这里的内容都属于“合理复现假设”，不能写成论文原文。

## 当前已识别的合理复现假设

1. 信道几何距离未完整给出。
   - 合理复现假设：先采用可控的 Rician 随机信道模型，并加入路径损耗指数 `alpha = 2`。
   - 后续如果需要几何建模，再明确基站、RIS、目标坐标。

2. Rician 信道生成细节未完整给出。
   - 合理复现假设：使用 `K = 10 dB` 的 LoS 加 NLoS 复高斯模型。
   - LoS 阵列响应、阵元间距、角度分布需要后续人工确认或从相关文献补充。

3. 噪声功率设置不够明确。
   - 论文图3文字称发射功率和噪声功率基准都定为 10 dBm。
   - 合理复现假设：先把噪声功率作为显式参数 `noisePower_dBm`，并在图3中默认设置为 10 dBm；后续检查这是否符合 SNR 量级。

4. CD 算法细节未在论文正文中展开。
   - 合理复现假设：参考坐标下降相位优化基线，每次更新一个 RIS 单元相位，并以路径增益或 SNR 为目标。
   - 具体更新式需要单独推导或参考论文引用文献[38]。

5. ADMM 中 `T` 矩阵维度和增广变量维度需要核对。
   - 论文从 `N_r` 维相移向量扩展到 `N_r + 1` 维变量，但 PDF 公式排版存在压缩。
   - 合理复现假设：实现前先写小尺寸矩阵测试，验证目标函数等价性。

6. FMCW 调制方式需要确认。
   - 论文估计模型中讨论三角波上下扫频，实验图5和图6可能可用等效 chirp 序列距离-多普勒处理实现。
   - 合理复现假设：先实现标准 FMCW 数据立方体和 2D FFT，再决定是否补充上下扫频拍频显式估计。

7. 目标 RCS 细节未区分四个目标。
   - 论文表1给出 `RCS = 1`，图5文字提到强/弱目标。
   - 合理复现假设：第一版所有目标 RCS 相同；如需复现强弱差异，再记录每个目标 RCS 设置。

8. 随机性需要可复现。
   - 合理复现假设：默认固定 `rngSeed = 20251009`，与论文网络首发日期对应，便于复现实验结果。

## 后续待确认

- `H_rd` 的物理维度是否代表 RIS 到目标、目标到 RIS、还是目标散射后的等效矩阵。
- 接收信号中的双程 RIS 结构是否应写成 `H_sr^H * Phi * H_rd * Phi^H * H_sr`，还是需要转置/共轭修正。
- 图3和图4横轴具体取值范围。
- 图6 距离轴和多普勒轴的 FFT 点数、窗函数、归一化和动态范围。

## Stage 2 Assumptions Added

1. 本阶段采用 `Hsr: Nr x Nt`，`Hrd: Nr x Nr`。
   - 这是为了使 `Heff = Hsr^H * Phi * Hrd * Phi^H * Hsr` 得到 `Nt x Nt` 的等效信道。
   - `Hrd` 在代码中解释为 RIS 域目标散射/回波等效矩阵，而不是已完全物理展开的单程 RIS-target 信道。

2. 默认几何距离采用：
   - source-to-RIS distance: `15 m`
   - RIS-to-target effective distance: `20 m`
   - reference distance: `1 m`
   - path loss exponent: `2`
   这些距离并非论文表1给出，属于合理复现假设。

3. Rician 信道生成采用：
   - `K = 10 dB`
   - deterministic unit-norm ULA steering outer product as LoS component
   - circular complex Gaussian random matrix as NLoS component
   - path loss applied as amplitude scale `sqrt((d0/d)^alpha)`

4. `generate_channels.m` 默认在函数内根据 `params.repro.rngSeed` 重置随机种子。
   - 优点：基础验证可重复。
   - 后续做 Monte Carlo 仿真时，需要增加选项避免每次循环生成同一个信道。

5. 当前名义 SNR 绝对值不作为论文数值复现结论。
   - 第二阶段只验收维度、功率约束和单调性。
   - 当前 `noisePower_dBm = 10` 来自论文图3文字描述，但与双路径损耗结合后会得到很低的 SNR。后续如需匹配论文曲线，需要重新审查噪声功率定义、带宽噪声、接收增益和归一化方式。

6. 论文和人工建议都作为待验证输入。
   - 若公式、维度或趋势与实际 MATLAB 验证不一致，后续以可复现实验和维度自洽为准，并在本文档中记录偏差原因。
