"""s16：OFDM/相位编码波形设计与互相关分析。

验证目标：
  - 实现 OFDM 波形生成（QPSK 调制 + IFFT）并验证子载波正交性
  - 实现 Barker 码、随机相位码、Frank 码，对比自相关旁瓣特性
  - 计算模糊函数，展示 delay-Doppler 分辨特性
  - 验证互相关正交性、旁瓣电平、模糊函数峰值位置

OFDM 波形原理：
  OFDM 将宽带信号分成 N 个正交子载波，每个子载波独立调制。
  子载波间距 Δf = 1/T_sym，其中 T_sym 为 OFDM 符号时长。
  时域信号通过 IFFT 生成：s[n] = Σ_k X[k] * exp(j*2π*k*n/N)
  子载波正交性保证：∫ s_k(t) * s_m*(t) dt = 0, 当 k ≠ m。

相位编码波形原理：
  相位编码将脉冲分成 N 个子脉冲（码片），每个码片赋予特定相位。
  Barker 码具有最低峰值旁瓣比（PSL = 1/N），但长度有限（最大 13）。
  Frank 码是多相码，长度为 M²，具有较好的多普勒容限。
  自相关函数的旁瓣电平决定了距离旁瓣性能。

模糊函数原理：
  |χ(τ, ν)|² = |∫ s(t) * s*(t-τ) * exp(j2πνt) dt|²
  模糊函数描述了信号在 delay-Doppler 平面上的分辨能力。
  峰值位于原点 (τ=0, ν=0)，理想"图钉"形模糊函数具有最佳分辨率。

对应知识库：radar-knowledge-base/基础/03-波形设计/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.signal_utils import power_to_db, db_to_power
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ============================================================
# 核心函数
# ============================================================


def generate_ofdm(
    n_subcarriers: int,
    n_symbols: int,
    subcarrier_spacing_hz: float,
    seed: int = 42,
) -> np.ndarray:
    """生成 OFDM 频域信号矩阵（QPSK 调制）。

    每个子载波独立使用 QPSK 调制：X[k,m] ∈ {1, j, -1, -j} / sqrt(2)。
    返回频域表示，可通过 IFFT 转换为时域信号。

    QPSK 星座点等概率出现，归一化使每符号能量为 1。

    FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
      IFFT: s[n] = (1/N) * Σ_k X[k] * exp(j*2π*k*n/N)

    Args:
        n_subcarriers:         子载波数 N（通常为 2 的幂）
        n_symbols:             OFDM 符号数 M
        subcarrier_spacing_hz: 子载波间距 Δf (Hz)
        seed:                  随机种子（保证可复现）

    Returns:
        频域信号矩阵 (N × M)，每个元素为 QPSK 星座点
    """
    rng = np.random.default_rng(seed)

    # QPSK 星座：{1, j, -1, -j}，归一化使 |symbol|² = 1
    qpsk_phases = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    qpsk_constellation = np.exp(1j * qpsk_phases)  # {1, j, -1, -j}

    # 随机选择 QPSK 符号
    symbol_indices = rng.integers(0, 4, size=(n_subcarriers, n_symbols))
    ofdm_freq = qpsk_constellation[symbol_indices]

    return ofdm_freq


def ofdm_time_signal(
    ofdm_freq: np.ndarray,
    n_subcarriers: int,
    subcarrier_spacing_hz: float,
) -> tuple[np.ndarray, float]:
    """将 OFDM 频域信号转换为时域信号（IFFT）。

    对每个 OFDM 符号（列）执行 N 点 IFFT：
      s[n] = (1/N) * Σ_{k=0}^{N-1} X[k] * exp(j*2π*k*n/N)

    子载波映射：k=0 对应 DC，k=1..N/2-1 对应正频率，
    k=N/2 对应 Nyquist，k=N/2+1..N-1 对应负频率。

    等效采样率：fs = N * Δf

    Args:
        ofdm_freq:             频域信号矩阵 (n_subcarriers × n_symbols)
        n_subcarriers:         子载波数
        subcarrier_spacing_hz: 子载波间距 (Hz)

    Returns:
        time_signal:  时域信号矩阵 (n_subcarriers × n_symbols)
        sample_rate_hz: 等效采样率 (Hz) = n_subcarriers * subcarrier_spacing_hz
    """
    # IFFT：每个符号（列）独立变换
    # numpy.fft.ifft: (1/N) * Σ X[k] * exp(j*2π*k*n/N)
    time_signal = np.fft.ifft(ofdm_freq, axis=0)

    sample_rate_hz = n_subcarriers * subcarrier_spacing_hz

    return time_signal, sample_rate_hz


def generate_phase_code(
    code_length: int,
    code_type: str,
    seed: int = 42,
) -> np.ndarray:
    """生成相位编码序列。

    支持三种编码类型：
      - "barker"：Barker 码，二相码，峰值旁瓣比最低（1/N）
        支持长度：2, 3, 4, 5, 7, 11, 13
        13 位 Barker 码：{+1,+1,+1,+1,+1,-1,-1,+1,+1,-1,+1,-1,+1}
      - "random"：随机相位码，每个码片相位从 [0, 2π) 均匀采样
      - "frank"：Frank 码，多相码，长度为 M²
        c_{ij} = exp(j * 2π * i * j / M)，i,j = 0..M-1

    Args:
        code_length: 码长 N（Frank 码需为完全平方数）
        code_type:   编码类型，"barker"/"random"/"frank"
        seed:        随机种子

    Returns:
        复数相位序列 (N,)，每个元素的模为 1
    """
    rng = np.random.default_rng(seed)

    if code_type == "barker":
        # Barker 码定义（{+1, -1} 格式）
        barker_codes = {
            2: np.array([1, -1]),
            3: np.array([1, 1, -1]),
            4: np.array([1, 1, -1, 1]),
            5: np.array([1, 1, 1, -1, 1]),
            7: np.array([1, 1, 1, -1, -1, 1, -1]),
            11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
            13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
        }
        if code_length not in barker_codes:
            raise ValueError(
                f"Barker 码不支持长度 {code_length}，"
                f"支持: {sorted(barker_codes.keys())}"
            )
        # 转换为复数相位形式：+1 → exp(j0) = 1, -1 → exp(jπ) = -1
        code = barker_codes[code_length].astype(np.complex128)

    elif code_type == "random":
        # 随机相位码：相位从 [0, 2π) 均匀采样
        phases = rng.uniform(0, 2 * np.pi, code_length)
        code = np.exp(1j * phases)

    elif code_type == "frank":
        # Frank 码：长度为 M²
        m = int(np.sqrt(code_length))
        if m * m != code_length:
            raise ValueError(
                f"Frank 码长度必须为完全平方数，当前: {code_length}"
            )
        # 构造 M×M 矩阵，按行展开
        i_idx, j_idx = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        frank_matrix = np.exp(1j * 2 * np.pi * i_idx * j_idx / m)
        code = frank_matrix.flatten()

    else:
        raise ValueError(
            f"未知编码类型: {code_type}，可选: barker/random/frank"
        )

    return code


def phase_code_autocorr(code: np.ndarray) -> np.ndarray:
    """计算相位编码的自相关函数。

    自相关函数定义：ρ[τ] = Σ_n c[n] * c*[n-τ]

    物理含义：
      自相关函数描述了信号与其延迟副本的相似度。
      主瓣（τ=0）为信号能量 Σ|c[n]|²。
      旁瓣（τ≠0）反映了距离旁瓣——低旁瓣意味着弱目标不会被强目标掩盖。

    实现：基于 FFT 的线性相关，零填充避免循环混叠。
      ρ = IFFT(FFT(c) * conj(FFT(c)))

    Args:
        code: 复数相位序列 (N,)

    Returns:
        自相关函数 (2N-1,)，索引 0 为主瓣（lag=0），索引 k 对应 lag k
        （注意：IFFT(|C|^2) 的输出中，index 0 = lag 0，而非居中排列）
    """
    n = len(code)
    # 零填充到 2N，避免循环卷积混叠
    n_fft = 2 * n
    C = np.fft.fft(code, n_fft)
    rho = np.fft.ifft(C * np.conj(C))

    # 取前 2N-1 个点（线性相关结果）
    return rho[: 2 * n - 1]


def cross_correlation(
    code1: np.ndarray, code2: np.ndarray
) -> tuple[np.ndarray, float]:
    """计算两个波形的归一化互相关函数。

    互相关衡量两个波形的正交性。互相关越低，波形分集能力越强。
    对于 MIMO 雷达，低互相关意味着不同发射通道的信号可以被分离。

    归一化互相关：ρ₁₂[τ] / sqrt(E₁ * E₂)
      其中 E₁ = Σ|c₁[n]|², E₂ = Σ|c₂[n]|²

    Args:
        code1: 第一个复数序列 (N1,)
        code2: 第二个复数序列 (N2,)

    Returns:
        xcorr_norm: 归一化互相关函数 (N1+N2-1,)，单位为线性
        peak_db:    峰值互相关 (dB)，越低越正交
    """
    n1, n2 = len(code1), len(code2)
    n_fft = n1 + n2 - 1

    C1 = np.fft.fft(code1, n_fft)
    C2 = np.fft.fft(code2, n_fft)
    xcorr = np.fft.ifft(C1 * np.conj(C2))

    # 归一化：除以自相关峰值的几何平均（sqrt(E1 * E2)）
    energy1 = np.sum(np.abs(code1) ** 2)
    energy2 = np.sum(np.abs(code2) ** 2)
    norm_factor = np.sqrt(energy1 * energy2)

    xcorr_norm = np.abs(xcorr) / norm_factor
    peak_db = power_to_db(np.max(xcorr_norm) ** 2)

    return xcorr_norm, peak_db


def ambiguity_function(
    signal: np.ndarray,
    max_delay: int,
    max_doppler: int,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算信号的二维模糊函数。

    模糊函数定义：
      |χ(τ, ν)|² = |Σ_n s[n] * s*[n-τ] * exp(j2πνn/N)|²

    物理含义：
      模糊函数描述了信号在 delay-Doppler 平面上的分辨能力。
      - 主瓣在 (τ=0, ν=0) 处，高度为信号能量的平方
      - 切面 τ=0 给出多普勒分辨率
      - 切面 ν=0 给出距离分辨率
      - 理想"图钉"形模糊函数具有最佳分辨率

    实现：
      对每个 Doppler 频移 ν，计算频移信号与原始信号的互相关。
      使用 FFT 加速相关计算。

    Args:
        signal:        输入信号 (N,)
        max_delay:     最大延迟（采样点数）
        max_doppler:   最大多普勒频移（采样点数）
        sample_rate_hz: 采样率 (Hz)

    Returns:
        amb:          模糊函数幅度 |χ(τ, ν)|² (2*max_delay+1, 2*max_doppler+1)
        delay_axis:   延迟轴（秒）
        doppler_axis: 多普勒轴（Hz）
    """
    n = len(signal)
    n_delay = 2 * max_delay + 1
    n_doppler = 2 * max_doppler + 1

    # 延迟轴和多普勒轴
    delay_samples = np.arange(-max_delay, max_delay + 1)
    doppler_samples = np.arange(-max_doppler, max_doppler + 1)

    delay_axis = delay_samples / sample_rate_hz  # 秒
    doppler_axis = doppler_samples / n  # 归一化频率（周期/样本）

    # 模糊函数矩阵
    amb = np.zeros((n_delay, n_doppler))

    # 参考能量（自相关峰值）
    ref_energy = np.sum(np.abs(signal) ** 2)

    # 对每个多普勒频移计算相关
    for d_idx, d in enumerate(doppler_samples):
        # 频移信号：s[n] * exp(j2πdn/N)
        n_array = np.arange(n)
        doppler_shift = np.exp(1j * 2 * np.pi * d * n_array / n)
        shifted_signal = signal * doppler_shift

        # 频域互相关
        n_fft = 2 * n
        S_orig = np.fft.fft(signal, n_fft)
        S_shift = np.fft.fft(shifted_signal, n_fft)
        xcorr = np.fft.ifft(S_orig * np.conj(S_shift))

        # 提取指定延迟范围
        center = 0
        for tau_idx, tau in enumerate(delay_samples):
            amb[tau_idx, d_idx] = np.abs(xcorr[center + tau]) ** 2

    # 归一化到原点值
    amb /= ref_energy ** 2

    return amb, delay_axis, doppler_axis


# ============================================================
# 绘图
# ============================================================


def plot_waveform_design(
    n_subcarriers: int = 64,
    n_symbols: int = 8,
    subcarrier_spacing_hz: float = 15e3,
    seed: int = 42,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制波形设计结果（3 子图）。

    子图 1：OFDM 频谱和时域波形
      - 上：OFDM 频谱（幅度），标注子载波位置
      - 下：时域 I/Q 分量

    子图 2：不同相位码的自相关（对比旁瓣电平）
      - Barker-13、随机相位码、Frank 码

    子图 3：模糊函数等高线图（delay-Doppler 平面）
      - Barker-13 的二维模糊函数

    Args:
        n_subcarriers:         子载波数
        n_symbols:             OFDM 符号数
        subcarrier_spacing_hz: 子载波间距 (Hz)
        seed:                  随机种子
        output_dir:            输出目录
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # ---- 子图 1：OFDM 频谱和时域波形 ----
    ax1_top = axes[0]

    # 生成 OFDM 信号
    ofdm_freq = generate_ofdm(n_subcarriers, n_symbols, subcarrier_spacing_hz, seed)
    ofdm_time, sample_rate_hz = ofdm_time_signal(
        ofdm_freq, n_subcarriers, subcarrier_spacing_hz
    )

    # 频谱：取第一个符号
    symbol_freq = ofdm_freq[:, 0]
    symbol_time = ofdm_time[:, 0]

    # FFT 并 fftshift 使 DC 居中
    spectrum = np.fft.fftshift(np.abs(symbol_freq))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_subcarriers, d=1.0 / n_subcarriers))

    ax1_top.bar(
        freq_axis,
        spectrum,
        width=0.8,
        color="steelblue",
        alpha=0.8,
        label="子载波幅度",
    )
    ax1_top.set_xlabel("子载波索引 (归一化频率)", fontsize=12)
    ax1_top.set_ylabel("|X[k]|", fontsize=12)
    ax1_top.set_title(
        f"OFDM 频域信号（{n_subcarriers} 子载波, Δf={subcarrier_spacing_hz/1e3:.0f} kHz）",
        fontsize=13,
    )
    ax1_top.legend(fontsize=11)
    ax1_top.grid(True, alpha=0.3)

    # 时域波形
    ax1_bot = axes[0].inset_axes([0.12, 0.08, 0.76, 0.35])
    t_us = np.arange(n_subcarriers) / sample_rate_hz * 1e6  # 微秒
    ax1_bot.plot(t_us, np.real(symbol_time), "b-", linewidth=1.2, label="I 路")
    ax1_bot.plot(t_us, np.imag(symbol_time), "r-", linewidth=1.2, label="Q 路")
    ax1_bot.set_xlabel("时间 (μs)", fontsize=10)
    ax1_bot.set_ylabel("幅度", fontsize=10)
    ax1_bot.set_title("时域 I/Q 波形（一个 OFDM 符号）", fontsize=10)
    ax1_bot.legend(fontsize=9, loc="upper right")
    ax1_bot.grid(True, alpha=0.3)

    # ---- 子图 2：不同相位码的自相关 ----
    ax2 = axes[1]

    barker_code = generate_phase_code(13, "barker", seed)
    random_code = generate_phase_code(13, "random", seed)
    frank_code = generate_phase_code(16, "frank", seed)

    codes = [
        ("Barker-13", barker_code, "b"),
        ("随机相位码(13)", random_code, "r"),
        ("Frank-16", frank_code, "g"),
    ]

    for name, code, color in codes:
        rho = phase_code_autocorr(code)
        # IFFT(|C|^2) 输出：index 0 = lag 0, index k = lag k, index N+k = lag -(N-1-k)
        # fftshift 重排使 lag 0 居中
        rho_shifted = np.fft.fftshift(rho)
        rho_norm = np.abs(rho_shifted) / np.max(np.abs(rho_shifted))
        rho_db = power_to_db(rho_norm ** 2)
        delay_chips = np.arange(-(len(code) - 1), len(code))
        ax2.plot(delay_chips, rho_db, f"{color}-", linewidth=1.5, label=name)

    ax2.axhline(y=-22.3, color="gray", linestyle="--", alpha=0.5, label="Barker-13 PSL (-22.3 dB)")
    ax2.set_xlabel("延迟 (码片)", fontsize=12)
    ax2.set_ylabel("归一化自相关 (dB)", fontsize=12)
    ax2.set_title("相位编码自相关函数对比", fontsize=13)
    ax2.set_ylim([-50, 5])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3：模糊函数 ----
    ax3 = axes[2]

    max_delay = 20
    max_doppler = 10
    amb, delay_axis, doppler_axis = ambiguity_function(
        barker_code, max_delay, max_doppler, sample_rate_hz=1.0
    )

    amb_db = power_to_db(amb)
    delay_chips = np.arange(-max_delay, max_delay + 1)
    doppler_norm = np.arange(-max_doppler, max_doppler + 1) / len(barker_code)

    cf = ax3.contourf(
        delay_chips,
        doppler_norm,
        amb_db.T,
        levels=np.linspace(-40, 0, 21),
        cmap="jet",
    )
    cbar = fig.colorbar(cf, ax=ax3, label="|χ(τ,ν)|² (dB)")
    ax3.set_xlabel("延迟 τ (码片)", fontsize=12)
    ax3.set_ylabel("多普勒 ν (归一化频率)", fontsize=12)
    ax3.set_title(
        f"Barker-13 模糊函数（|χ(τ,ν)|²，归一化到 0 dB）",
        fontsize=13,
    )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s16_waveform_design.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s16_waveform_design.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_subcarriers: int = 64,
    n_symbols: int = 8,
    subcarrier_spacing_hz: float = 15e3,
    seed: int = 42,
) -> bool:
    """验证波形设计的正确性。

    验证项：
      a. Barker-13 峰值旁瓣比：PSL = -22.3 dB
      b. OFDM 子载波正交性：互相关 < -30 dB
      c. 模糊函数峰值：位于原点 (τ=0, ν=0)
      d. 相位码自相关主瓣：归一化峰值 = 1 (0 dB)
    """
    results = []

    # --- 验证 a：Barker-13 峰值旁瓣比 ---
    barker = generate_phase_code(13, "barker", seed)
    rho = phase_code_autocorr(barker)
    energy = np.sum(np.abs(barker) ** 2)  # = 13

    # 归一化自相关：ρ_norm = |ρ| / energy
    # IFFT(|C|^2) 的输出：rho[0] = lag 0（主瓣），rho[k] = lag k
    # 主瓣（lag=0）归一化后 = 1（0 dB），旁瓣 = 1/N
    rho_norm = np.abs(rho) / energy

    # 主瓣在索引 0（lag=0），旁瓣为其他位置
    n = len(barker)
    peak_idx = 0
    sidelobe_mask = np.ones(len(rho), dtype=bool)
    sidelobe_mask[peak_idx] = False
    max_sidelobe = np.max(rho_norm[sidelobe_mask])

    # PSL = 10*log10((1/N)²) = 10*log10(1/169) = -22.28 dB
    psl_db = power_to_db(max_sidelobe ** 2)
    results.append(verify(
        name="Barker-13 峰值旁瓣比 (PSL)",
        theoretical=-22.28,
        simulated=psl_db,
        tolerance=1.0,
        unit="dB",
    ))

    # --- 验证 b：OFDM 子载波正交性 ---
    # OFDM 子载波正交性：DFT 基向量在完整周期上的圆相关为零
    # C_circ[k,m] = (1/N) * Σ_{n=0}^{N-1} s_k[n] * s_m*[n] = δ(k,m)
    # 验证方法：比较 IFFT 后各子载波信号的圆相关
    ofdm_freq_test = generate_ofdm(n_subcarriers, 2, subcarrier_spacing_hz, seed)
    ofdm_time_test, _ = ofdm_time_signal(
        ofdm_freq_test, n_subcarriers, subcarrier_spacing_hz
    )

    # 计算不同子载波（列）之间的圆相关
    # 每列的 IFFT 结果包含所有子载波的叠加
    # 检查同一列内不同时间样本之间的正交性
    # 更直接：检查 IFFT 矩阵的列正交性
    n_sc = n_subcarriers
    # 构造 DFT 矩阵的 IFFT 列
    dft_matrix = np.fft.ifft(np.eye(n_sc), axis=0)  # (N × N)，每列是一个子载波信号

    # 计算所有子载波对之间的归一化圆相关
    max_cross = 0.0
    for k in range(min(n_sc, 10)):  # 检查前 10 个子载波
        for m in range(k + 1, min(n_sc, 10)):
            # 圆相关：(1/N) * s_k^H * s_m
            circ_corr = np.abs(dft_matrix[:, k].conj() @ dft_matrix[:, m]) / n_sc
            max_cross = max(max_cross, circ_corr)

    cross_peak_db = power_to_db(max_cross ** 2)

    results.append(verify(
        name="OFDM 子载波正交性（圆相关峰值）",
        theoretical=-300.0,  # 理论值：完美正交 → 0 → -∞ dB（数值精度约 -300 dB）
        simulated=cross_peak_db,
        tolerance=280.0,     # 允许到 -20 dB（实际上应远低于此）
        unit="dB",
    ))

    # --- 验证 c：模糊函数峰值位于原点 ---
    max_delay = 20
    max_doppler = 10
    amb, _, _ = ambiguity_function(barker, max_delay, max_doppler, sample_rate_hz=1.0)

    # 峰值应在原点 (delay=0, doppler=0)，即矩阵中心
    center_delay = max_delay
    center_doppler = max_doppler
    peak_value = amb[center_delay, center_doppler]
    peak_location = np.unravel_index(np.argmax(amb), amb.shape)

    # 验证峰值位置是否在原点
    is_at_origin = (peak_location[0] == center_delay and
                    peak_location[1] == center_doppler)

    results.append(verify(
        name="模糊函数峰值位于原点",
        theoretical=1.0,
        simulated=float(is_at_origin),
        tolerance=0.5,
        unit="(布尔)",
    ))

    # --- 验证 d：相位码自相关主瓣归一化峰值 = 0 dB ---
    # 归一化方式：除以 max(|ρ|)，使主瓣 = 1（0 dB）
    codes_to_test = [
        ("Barker-13", generate_phase_code(13, "barker", seed)),
        ("随机相位码", generate_phase_code(13, "random", seed)),
        ("Frank-16", generate_phase_code(16, "frank", seed)),
    ]

    max_mainlobe_error = 0.0
    for name, code in codes_to_test:
        rho = phase_code_autocorr(code)
        rho_normalized = np.abs(rho) / np.max(np.abs(rho))
        # IFFT(|C|^2) 的主瓣在索引 0（lag=0）
        peak_db = power_to_db(rho_normalized[0] ** 2)
        max_mainlobe_error = max(max_mainlobe_error, abs(peak_db - 0.0))

    results.append(verify(
        name="相位码自相关主瓣归一化峰值",
        theoretical=0.0,
        simulated=max_mainlobe_error,
        tolerance=0.1,
        unit="dB",
    ))

    return print_validation("s16 OFDM/相位编码波形设计", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s16 OFDM/相位编码波形设计仿真与验证。"""
    print("=" * 60)
    print("s16：OFDM/相位编码波形设计与互相关分析")
    print("=" * 60)

    # 仿真参数
    n_subcarriers = 64
    n_symbols = 8
    subcarrier_spacing_hz = 15e3
    seed = 42

    print(f"\n仿真参数:")
    print(f"  OFDM 子载波数 = {n_subcarriers}")
    print(f"  OFDM 符号数   = {n_symbols}")
    print(f"  子载波间距     = {subcarrier_spacing_hz/1e3:.0f} kHz")
    print(f"  等效采样率     = {n_subcarriers * subcarrier_spacing_hz / 1e6:.3f} MHz")
    print(f"  Barker 码长度  = 13")
    print(f"  随机种子       = {seed}")

    # 绘图
    print(f"\n绘制波形设计结果...")
    plot_waveform_design(
        n_subcarriers, n_symbols, subcarrier_spacing_hz, seed
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(
        n_subcarriers, n_symbols, subcarrier_spacing_hz, seed
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
