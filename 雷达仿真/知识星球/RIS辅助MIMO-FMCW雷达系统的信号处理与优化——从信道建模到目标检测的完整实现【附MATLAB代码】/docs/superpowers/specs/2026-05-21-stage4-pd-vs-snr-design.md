# Stage 4 Pd-vs-SNR Design

## 目标

新增 Stage 4 的 `Pd-vs-SNR` Monte Carlo 实验，验证 NLOS 场景下 `No RIS`、`Random RIS`、`Fixed-grid ZF-SNR optimized RIS` 三组在不同回波 SNR 下的四目标 CA-CFAR 检测概率。

## 实验口径

- 主横轴采用 `echo SNR dB`。
- 每个 SNR 点同时保存对应的 `echoNoisePower_W`，便于按噪声功率复核。
- SNR 参考回波使用 optimized RIS 组的四目标平均无噪声 beat-signal 功率。
- 三组在同一 SNR 点使用同一噪声功率，RIS 方法差异保留在各自的回波幅度增益中。
- `No RIS` 继续沿用当前遮挡 NLOS 基线：目标回波幅度为零，只保留噪声。

## 数据流

1. 读取 `paper_params` 并构造当前四目标参数。
2. 生成一个 Stage 4 信道和相同的 start phases。
3. 复用 `fixed_grid_zf_snr` 优化器得到 optimized RIS，相同信道下保留 random RIS。
4. 根据 optimized RIS 的无噪声回波功率把 `echoSNRdB` 换算为每个点的 `echoNoisePower_W`。
5. 对每个 SNR 点执行 Monte Carlo：
   - 重采样回波噪声；
   - 生成 no-RIS、random、optimized 三组 beat signal；
   - 计算 RD 谱；
   - 使用现有 `detect_rd_targets_cfar` 做全图 CA-CFAR 和真值邻域关联；
   - 累计四目标命中数。
6. 输出每目标 Pd、平均 Pd、数据、图件和文本日志。

## 模式

- `quick`
  - 默认 `numTrials = 8`
  - 固定信道
  - 用于快速验收脚本、日志打印和趋势
- `full`
  - 默认 `numTrials = 100`
  - 支持通过配置选择是否每 trial 重新生成信道
  - 用于本地正式实验

## 输出与验收

- 输出 `.mat` 数据、`.png` 图和 `.txt` 日志。
- 命令行每个 trial 打印一行表格式进度，含 SNR 点、trial、噪声功率、三组命中数和当前平均 Pd。
- 图件至少包含：
  - 三组 average Pd vs SNR
  - 三组 per-target Pd vs SNR
- quick 验收关注：
  - 脚本无报错；
  - optimized RIS 平均 Pd 通常高于 random RIS；
  - No RIS 最低；
  - average Pd 随 SNR 整体上升。
