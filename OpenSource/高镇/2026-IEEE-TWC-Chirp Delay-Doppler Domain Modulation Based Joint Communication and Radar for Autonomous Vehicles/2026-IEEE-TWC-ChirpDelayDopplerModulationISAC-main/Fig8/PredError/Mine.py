#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte‑Carlo simulation for a single‑tone + QPSK system (1024‑sample symbol)
────────────────────────────────────────────────────────────────────────────
• 发送端：
  ‑ 单音频率 f_tone = n * 10 MHz / 512, n ∈ {0,…,511}  → 9 bit
  ‑ QPSK 符号 s ∈ {±1±j}/√2 → 2 bit
  ‑ 基带长度 N = 1024；Fs = 20 MHz ⇒ Δf_bin = Fs/N = 19.53125 kHz，
    与 10 MHz/512 完全一致 ⇒ bin index = n。
• 信道：
  ‑ 连续频偏 Δf ∼ U(0,10 MHz) ⇒ 归一化偏移 δ = Δf/Fs ∈ [0,0.5]。
  ‑ 加性独立复高斯噪声（时域，每采样单位信号功率=1）。
• 接收端：
  ‑ 已知 Δf，先乘 e^(−j2πδn) 完全抵消频偏；
  ‑ 1024‑点 FFT → 取幅度最大 bin → ML 估计 n；
  ‑ 对应 bin 上做 QPSK 判决。
• 仿真 3 个 case（Tone+QPSK / Tone‑only / QPSK‑only），
  结果输出 JSON&曲线，含 tqdm 进度条。
"""

import os, math, json, torch, importlib.util
import matplotlib.pyplot as plt

# ─── OMP 冲突 ─────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── tqdm ─────────────────────────────────────────────────────────────────
if importlib.util.find_spec("tqdm"):
    from tqdm.auto import tqdm
    progbar = True
else:                                  # 无 tqdm 时降级
    def tqdm(x, **k):
        return x
    progbar = False
    print("[Info] tqdm 未安装，使用简易进度。pip install tqdm 可获得更好体验")

# ─── 系统参数 ────────────────────────────────────────────────────────────
dev           = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES     = 1024                    # 每帧采样数
N_BINS        = N_SAMPLES               # FFT bin 数
TONE_SPACING_BIN = 1        # 频率信息调制间距（单位=bin=频率分辨率）；可设为 1/2/3/…
FREQ_ERR_BIN     = 0.5     # 接收端对“真实频偏”的估计误差（单位=bin）。例：-0.1=估计比真实低0.1个bin

# 由间距派生：可用音调个数与比特数
N_TONES         = 512 // TONE_SPACING_BIN
TONE_BITS       = int(math.floor(math.log2(N_TONES)))   # 若要继续用9bit，见文末“可选方案”
TONE_VALID_IDXS = torch.arange(0, 512, TONE_SPACING_BIN, device=dev)  # 合法bin集合
SNRdB_vec     = torch.arange(-30, 0, 5, device=dev)
ITER_ALL      = 1_000
BATCH_PER_IT  = 1_000
TOTAL_SYMS    = ITER_ALL * BATCH_PER_IT
CHUNK         = 4_096                   # 并行块，≈ 4k*1024 ≈ 4 Mi samples
TWO_PI_BY_N   = 2 * math.pi / N_SAMPLES

time_idx = torch.arange(N_SAMPLES, device=dev)               # 1×1024 行向量

# ─── QPSK 工具 ────────────────────────────────────────────────────────────

def bits_to_qpsk(b0: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """Gray 映射，输出复数 float32 幅度=1"""
    real = 1 - 2 * b0
    imag = 1 - 2 * b1
    return (real + 1j * imag).to(torch.complex64) / math.sqrt(2)

def qpsk_demod(sym: torch.Tensor):
    """硬判决"""
    return (sym.real < 0).long(), (sym.imag < 0).long()

# ─── 存储结果 ────────────────────────────────────────────────────────────
results = {
    "case1_both":      {"BER": [], "SER": [], "SER_tone": [], "SER_qpsk": []},
    "case2_tone_only": {"BER": [], "SER_tone": []},
    "case3_qpsk_only": {"BER": [], "SER_qpsk": []},
}

# ─── 主仿真 ──────────────────────────────────────────────────────────────
with torch.no_grad():
    for snr_db in tqdm(SNRdB_vec.tolist(), desc="SNR sweep"):
        snr_lin = 10 ** (snr_db / 10)
        noise_var = 1 / snr_lin                    # 每复采样功率
        noise_std = math.sqrt(noise_var / 2)       # 每实部/虚部 std

        b_err1 = b_err2 = b_err3 = 0
        s_err_t1 = s_err_t2 = 0
        s_err_q1 = s_err_q3 = 0
        s_err_all1 = 0

        processed = 0
        inner = tqdm(total=TOTAL_SYMS, leave=False,
                      desc=f"{snr_db:>3} dB") if progbar else None

        while processed < TOTAL_SYMS:
            bs = min(CHUNK, TOTAL_SYMS - processed)        # 当前批大小
            rng = torch.arange(bs, device=dev)

            # ── 通用随机变量 ───────────────────────────────────────
            tone_sel = torch.randint(N_TONES, (bs,), device=dev)  # 0..N_TONES-1
            tone_idx = tone_sel * TONE_SPACING_BIN                # 实际使用的bin（按间距）
            phase_tone = tone_idx.unsqueeze(1) * time_idx * TWO_PI_BY_N
            delta = torch.rand(bs, device=dev) * 0.5          # 连续偏移 0‑0.5 (归一化)
            phase_ch = delta.unsqueeze(1) * time_idx * 2 * math.pi
            phase_est = (delta + FREQ_ERR_BIN / N_SAMPLES).unsqueeze(1) * time_idx * 2 * math.pi

            # ─────────── CASE 1: Tone+QPSK ───────────
            q_bits1 = torch.randint(0, 2, (bs, 2), device=dev)
            q_sym1 = bits_to_qpsk(q_bits1[:, 0], q_bits1[:, 1])
            tx1 = q_sym1.unsqueeze(1) * torch.exp(1j * phase_tone)
            ch1 = tx1 * torch.exp(1j * phase_ch)
            noise1 = (torch.randn(bs, N_SAMPLES, device=dev) +
                      1j * torch.randn(bs, N_SAMPLES, device=dev)) * noise_std
            rx1 = (ch1 + noise1) * torch.exp(-1j * phase_est)  # 不完美频偏均衡
            fft1 = torch.fft.fft(rx1, dim=1)
            mag1 = torch.abs(fft1[:, :512]).index_select(1, TONE_VALID_IDXS)  # 只在合法bin上搜索
            sel_hat1 = torch.argmax(mag1, 1)
            idx_hat1 = TONE_VALID_IDXS[sel_hat1]                               # 还原为真实bin索引
            sym_hat1 = fft1[rng, idx_hat1] / N_SAMPLES
            b0h1, b1h1 = qpsk_demod(sym_hat1)

            tone_bits = ((tone_idx.unsqueeze(1) >> torch.arange(TONE_BITS, device=dev)) & 1)
            tone_bits_hat = ((idx_hat1.unsqueeze(1) >> torch.arange(TONE_BITS, device=dev)) & 1)

            b_err1 += (tone_bits != tone_bits_hat).sum().item()
            b_err1 += (q_bits1[:, 0] != b0h1).sum().item()
            b_err1 += (q_bits1[:, 1] != b1h1).sum().item()
            s_err_t1 += (idx_hat1 != tone_idx).sum().item()
            s_err_q1 += ((q_bits1[:, 0] != b0h1) | (q_bits1[:, 1] != b1h1)).sum().item()
            err_comb1 = (idx_hat1 != tone_idx) | ((q_bits1[:, 0] != b0h1) | (q_bits1[:, 1] != b1h1))
            s_err_all1 += err_comb1.sum().item()

            # ─────────── CASE 2: Tone‑only ───────────
            tx2 = torch.exp(1j * phase_tone)                 # 常数幅度 1
            ch2 = tx2 * torch.exp(1j * phase_ch)
            noise2 = (torch.randn(bs, N_SAMPLES, device=dev) +
                      1j * torch.randn(bs, N_SAMPLES, device=dev)) * noise_std
            rx2 = (ch2 + noise2) * torch.exp(-1j * phase_est)
            fft2 = torch.fft.fft(rx2, dim=1)
            mag2 = torch.abs(fft2[:, :512]).index_select(1, TONE_VALID_IDXS)
            sel_hat2 = torch.argmax(mag2, 1)
            idx_hat2 = TONE_VALID_IDXS[sel_hat2]
            tone_bits_hat2 = ((idx_hat2.unsqueeze(1) >> torch.arange(TONE_BITS, device=dev)) & 1)
            b_err2 += (tone_bits != tone_bits_hat2).sum().item()
            s_err_t2 += (idx_hat2 != tone_idx).sum().item()

            # ─────────── CASE 3: QPSK‑only ───────────
            q_bits3 = torch.randint(0, 2, (bs, 2), device=dev)
            q_sym3 = bits_to_qpsk(q_bits3[:, 0], q_bits3[:, 1])
            phase_zero = torch.zeros_like(phase_tone)
            tx3 = q_sym3.unsqueeze(1) * torch.exp(1j * phase_zero)  # bin 0
            ch3 = tx3 * torch.exp(1j * phase_ch)
            noise3 = (torch.randn(bs, N_SAMPLES, device=dev) +
                      1j * torch.randn(bs, N_SAMPLES, device=dev)) * noise_std
            rx3 = (ch3 + noise3) * torch.exp(-1j * phase_est)
            fft3 = torch.fft.fft(rx3, dim=1)
            mag3 = torch.abs(fft3[:, :512]).index_select(1, TONE_VALID_IDXS)
            sel_hat3 = torch.argmax(mag3, 1)
            idx_hat3 = TONE_VALID_IDXS[sel_hat3]
            sym_hat3 = fft3[rng, idx_hat3] / N_SAMPLES
            b0h3, b1h3 = qpsk_demod(sym_hat3)
            b_err3 += (q_bits3[:, 0] != b0h3).sum().item() + (q_bits3[:, 1] != b1h3).sum().item()
            s_err_q3 += ((q_bits3[:, 0] != b0h3) | (q_bits3[:, 1] != b1h3)).sum().item()

            processed += bs
            if progbar:
                inner.update(bs)
        if progbar:
            inner.close()

        # ─── 归一化 ────────────────────────────────────────────────────
        total_b1 = TOTAL_SYMS * (TONE_BITS + 2)
        total_b2 = TOTAL_SYMS * TONE_BITS
        total_b3 = TOTAL_SYMS * 2

        results["case1_both"]["BER"].append(b_err1 / total_b1)
        results["case1_both"]["SER"].append(s_err_all1 / TOTAL_SYMS) 
        results["case1_both"]["SER_tone"].append(s_err_t1 / TOTAL_SYMS)
        results["case1_both"]["SER_qpsk"].append(s_err_q1 / TOTAL_SYMS)

        results["case2_tone_only"]["BER"].append(b_err2 / total_b2)
        results["case2_tone_only"]["SER_tone"].append(s_err_t2 / TOTAL_SYMS)

        results["case3_qpsk_only"]["BER"].append(b_err3 / total_b3)
        results["case3_qpsk_only"]["SER_qpsk"].append(s_err_q3 / TOTAL_SYMS)

# ─── 保存结果 ───────────────────────────────────────────────────────────
# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本所在的文件夹路径
script_dir = os.path.dirname(script_path)
# 构造最终要保存的文件的完整路径
output_path = os.path.join(script_dir, 'Mine.json')
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("[✓] JSON 写入 Mine.json")

# ─── 绘图 ────────────────────────────────────────────────────────────────

def plot_log(x, ylist, labels, ylabel, markers):
    plt.figure()
    for y, lbl, m in zip(ylist, labels, markers):
        plt.semilogy(x, y, m, label=lbl)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both")
    plt.legend()
    plt.title(f"{ylabel} vs SNR")

plot_log(SNRdB_vec.cpu(),
         [results["case1_both"]["BER"], results["case2_tone_only"]["BER"], results["case3_qpsk_only"]["BER"]],
         ["all", "delay", "qpsk"],
         "Bit Error Rate", ["o-","x-","s-"])

plot_log(SNRdB_vec.cpu(),
         [results["case1_both"]["SER_tone"], results["case1_both"]["SER_qpsk"],
          results["case2_tone_only"]["SER_tone"], results["case3_qpsk_only"]["SER_qpsk"]],
         ["delay in all", "qpsk in all", "delay", "qpsk"],
         "Symbol Error Rate", ["o-","^-","x-","s-"])

plt.show()
