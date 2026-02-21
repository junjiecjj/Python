%% OFDM-ISAC Beyond CP Limit - Iterative SIC-DFT
clear; clc; close all; rng(1);

%% ================== 系统参数（参考 Table I） ==================
c0  = 3e8;        % 光速 [m/s]
fc  = 28e9;       % 载波频率 [Hz]
N   = 128;        % 子载波数
M   = 64;         % OFDM 符号数
df  = 120e3;      % 子载波间隔 [Hz]
T   = 1/df;       % 有效符号时长 [s]
B   = N*df;       % 信号带宽 [Hz]

% 仅保留标准 CP
Tcp_std  = 0.59e-6;   % 标准 CP (3GPP NR)

% 目标参数（Fig. 6：300m, 600m, 800m）
R_targets = [300 600 800];      % [m]
v_targets = [-30 30 100];       % [m/s]
Q = numel(R_targets);           % 目标数

% 仿真 SNR
SNR_dB = 10;

% SIC 迭代次数（可自行调整）
maxIter = 5;

%% ====== 基本雷达性能参数 & CP 限制下的最大可感知距离 / 速度 ======
rangeRes      = c0 / (2*B);            % 距离分辨率
rangeMax_RDM  = rangeRes*(N-1);        % 由 FFT 轴决定的最大距离（RDM x 轴范围）

Ts_std  = T + Tcp_std;                 % 标准 CP 下 OFDM 符号总时长

% CP 限制下的“无 ISI 最大距离”：2R/c <= Tcp
Rmax_CP_std  = c0*Tcp_std / 2;

% 多普勒分辨率 & 无模糊最大速度（由 PRF = 1/Ts 决定）
velRes_std = c0 / (2*fc*M*Ts_std);     % 速度分辨率
velMax_std = c0 / (4*fc*Ts_std);       % 速度无模糊范围 |v| <= velMax_std

% ===== 打印系统 & 目标信息 =====
fprintf('==== System parameters ====\n');
fprintf('fc = %.2f GHz,  B = %.2f MHz\n', fc/1e9, B/1e6);
fprintf('N = %d subcarriers,  M = %d OFDM symbols\n', N, M);
fprintf('Range resolution      ΔR = %.2f m\n', rangeRes);
fprintf('RDM axis max range  R_max_RDM ≈ %.2f m\n\n', rangeMax_RDM);

fprintf('Standard CP:\n');
fprintf('  Tcp_std = %.3f us, Ts = %.3f us\n', Tcp_std*1e6, Ts_std*1e6);
fprintf('  ISI-free max range (CP) R_max_CP_std ≈ %.2f m\n', Rmax_CP_std);
fprintf('  Velocity resolution      Δv_std ≈ %.2f m/s\n', velRes_std);
fprintf('  Unambiguous |v|_max_std ≈ %.2f m/s\n\n', velMax_std);

fprintf('==== True target list ====\n');
for q = 1:Q
    tau_q = 2*R_targets(q)/c0;          % 往返时延
    fd_q  = 2*v_targets(q)*fc/c0;       % 多普勒频移
    fprintf('Target %d: R = %.1f m, v = %.1f m/s,  tau = %.3f us,  f_d = %.2f kHz\n', ...
        q, R_targets(q), v_targets(q), tau_q*1e6, fd_q/1e3);
end
fprintf('\n');

fprintf('Will run iterative SIC-DFT with maxIter = %d\n\n', maxIter);

%% ================== 生成 1024-QAM 数据矩阵 S (N×M) ==================
Mqam = 32;                                     % 32×32 = 1024-QAM
levels = -(Mqam-1):2:(Mqam-1);                 % -31, -29, ..., 31
[Ix, Qx] = meshgrid(levels, levels);
const = Ix(:) + 1j*Qx(:);                      % 1024 星座点
const = const / sqrt(mean(abs(const).^2));     % 归一化到单位平均功率

S = const(randi(numel(const), N, M));          % N×M 独立 1024-QAM 符号

%% ================== 标准 CP 场景：含 ISI/ICI ==================
simStd  = simulate_OFDM_echo(N,M,df,fc,Tcp_std,R_targets,v_targets,S,SNR_dB);

%% ================== 1) 直接 DFT w. standard-CP ==================
% 作为“无 SIC”基线
[RDM_std_dB,  rangeAxis, velAxis_std]  = compute_RDM(simStd.Y, simStd.S, simStd.Ts, fc, simStd.B);

%% ================== 2) 迭代式 SIC-DFT w. standard-CP ==================
% 利用仿真里分解出来的 ISI/ICI：Y = Y_free + Y_ISI - Y_ICI + Z
% 此处做一个“理想/Oracle SIC”：每次迭代消除 1/maxIter 份 ISI/ICI
Y_intf = simStd.Y_ISI - simStd.Y_ICI;      % 总干扰项
Y_sic  = simStd.Y;                         % 初始为含 ISI/ICI 的接收信号

RDM_sic_iter = zeros(size(RDM_std_dB,1), size(RDM_std_dB,2), maxIter);

for it = 1:maxIter
    % 每次迭代去掉 1/maxIter 的 ISI/ICI
    Y_sic = Y_sic - (1/maxIter)*Y_intf;

    % 当前迭代的 RDM
    [RDM_tmp_dB, ~, ~] = compute_RDM(Y_sic, simStd.S, simStd.Ts, fc, simStd.B);
    RDM_sic_iter(:,:,it) = RDM_tmp_dB;

    fprintf('SIC iteration %d / %d done.\n', it, maxIter);
end

% 最后一轮迭代的 RDM 作为“迭代 SIC-DFT”的结果
RDM_sic_dB = RDM_sic_iter(:,:,maxIter);

%% ================== 作图：每种情况 2D + 3D ==================

% 图 (a) Iterative SIC-DFT w. standard-CP
figure;

% --- 左：2D 图 ---
subplot(1,2,1);
imagesc(rangeAxis, velAxis_std, RDM_sic_dB.');   % 注意转置，使 x 轴为 range，y 轴为 velocity
set(gca, 'YDir','normal');
colormap(jet);
colorbar;
clim([-80 0]);                                  % dB 动态范围，可按需要调整
xlabel('Range (m)');
ylabel('Velocity (m/s)');
title(sprintf('Iterative SIC-DFT (2D) - standard CP, iter = %d', maxIter));
xlim([0 1000]);
ylim([-200 200]);
axis xy;

% --- 右：3D 曲面图 ---
subplot(1,2,2);
surf(rangeAxis, velAxis_std, RDM_sic_dB.');
shading interp;            % 平滑一下
xlabel('Range (m)');
ylabel('Velocity (m/s)');
zlabel('Magnitude (dB)');
title(sprintf('Iterative SIC-DFT (3D) - standard CP, iter = %d', maxIter));
xlim([0 1000]);
ylim([-200 200]);
zlim([-80 0]);
view(45,30);               % 视角可自己调
grid on;

% 图 (b) DFT w. standard-CP（无 SIC）
figure;

% --- 左：2D 图 ---
subplot(1,2,1);
imagesc(rangeAxis, velAxis_std, RDM_std_dB.');
set(gca, 'YDir','normal');
colormap(jet);
colorbar;
clim([-80 0]);
xlabel('Range (m)');
ylabel('Velocity (m/s)');
title('DFT (2D) - standard CP (no SIC)');
xlim([0 1000]);
ylim([-200 200]);

% --- 右：3D 曲面图 ---
subplot(1,2,2);
surf(rangeAxis, velAxis_std, RDM_std_dB.');
shading interp;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
zlabel('Magnitude (dB)');
title('DFT (3D) - standard CP (no SIC)');
xlim([0 1000]);
ylim([-200 200]);
zlim([-80 0]);
view(45,30);
grid on;


%% ================== 函数定义 ==================
function sim = simulate_OFDM_echo(N,M,df,fc,Tcp,R_targets,v_targets,S,SNR_dB)
% 依据文中 (15)、(21) 的频域模型，直接在 Y 维度构造：
% Y = Y_free + Y_ISI - Y_ICI + Z
% 其中 Y_free, Y_ISI, Y_ICI 都写成 b(τ_q), c(f_dq), Φ_q, J1, S 的显式形式

    c0 = 3e8;
    B  = N*df;
    T  = 1/df;
    Ts = T + Tcp;

    Q = numel(R_targets);
    tau = 2*R_targets / c0;         % 往返时延 τ_q
    fd  = 2*v_targets*fc / c0;      % 多普勒频移 f_d,q

    % 反射系数：按 1/R^2 归一化，主要体现距离衰减趋势
    alphas = 1./(R_targets.^2);
    alphas = alphas / norm(alphas);

    % CP 长度（采样点）
    B_eff = B;                      % 这里 B = N*df
    Ncp = round(Tcp * B_eff);

    % 时间移矩阵 J1：右乘实现“向后平移一列” => 使用上一符号的数据
    J1 = diag(ones(M-1,1), 1);

    % 初始化
    Y_free = zeros(N,M);
    Y_ISI  = zeros(N,M);
    Y_ICI  = zeros(N,M);

    % 预存 steering 向量和 Φ_q 矩阵
    b_list   = cell(Q,1);
    c_list   = cell(Q,1);
    Phi_list = cell(Q,1);
    l_list   = zeros(Q,1);

    n  = (0:N-1).';
    m  = 0:M-1;

    % --------- 计算 Y_free 及 Φ_q ----------
    for q = 1:Q
        tau_q = tau(q);
        fd_q  = fd(q);

        b_q = exp(-1j*2*pi*df*tau_q*n);         % N×1
        c_q = exp(-1j*2*pi*fd_q*Ts*m);          % 1×M

        b_list{q} = b_q;
        c_list{q} = c_q;

        % Y_free 的该目标分量：b(τ_q) c^H(f_dq) ⊙ S
        A_q   = (b_q * conj(c_q)) .* S;         % N×M
        Y_free = Y_free + alphas(q)*A_q;

        % 对应的离散延迟 tap
        l_q = round(tau_q*B_eff);
        l_list(q) = l_q;

        % Φ_q 矩阵 (N×N)
        Phi_list{q} = computePhi(N, Ncp, l_q);
    end

    % --------- 计算 ISI / ICI ----------
    for q = 1:Q
        l_q = l_list(q);
        if l_q > Ncp
            tau_q = tau(q);
            fd_q  = fd(q);
            b_q   = b_list{q};
            c_q   = c_list{q};
            Phi_q = Phi_list{q};

            % 过 CP 的多出来部分 τ_q - Tcp
            tau_excess = tau_q - Tcp;
            b_ex = exp(-1j*2*pi*df*tau_excess*n);  % N×1

            % ISI: 使用上一 OFDM 符号的数据 => 右乘 J1
            A_isi      = (b_ex * conj(c_q)) .* S;  % N×M
            A_isi_shift = A_isi * J1;              % 与 s_{m-1} 对应
            Y_ISI = Y_ISI + alphas(q) * (Phi_q * A_isi_shift);

            % ICI: 同一符号内频率泄漏
            A_ici = (b_q * conj(c_q)) .* S;
            Y_ICI = Y_ICI + alphas(q) * (Phi_q * A_ici);
        end
    end

    % --------- 合成回波并加噪声 ----------
    Y_clean = Y_free + Y_ISI - Y_ICI;   % (17) 式（不含噪声部分）

    % 噪声功率按 Y_free 的平均功率与给定 SNR 设置
    sigPower   = mean(abs(Y_free(:)).^2);
    noisePower = sigPower / (10^(SNR_dB/10));
    sigma      = sqrt(noisePower/2);
    Z          = sigma*(randn(N,M) + 1j*randn(N,M));

    Y = Y_clean + Z;

    % 输出结构体
    sim.Y      = Y;
    sim.Y_free = Y_free;
    sim.Y_ISI  = Y_ISI;
    sim.Y_ICI  = Y_ICI;
    sim.S      = S;
    sim.B      = B_eff;
    sim.Ts     = Ts;
    sim.Ncp    = Ncp;
end

function Phi = computePhi(N, Ncp, lq)
% 计算 Φ_q (N×N)，对应文中 φ_{q,n,n'}：
% φ_q(n,n') = (1/N) * sum_{i=0}^{l_q-N_cp-1} exp(j*2π*(n'-n)*i/N)
% 若 l_q <= Ncp，则没有 ISI/ICI，Φ_q 为零矩阵

    if lq <= Ncp
        Phi = zeros(N,N);
        return;
    end

    K = lq - Ncp;                   % 求和项数
    n      = (0:N-1).';             % N×1
    nprime = 0:N-1;                 % 1×N
    delta  = nprime - n;            % N×N

    Phi = zeros(N,N);

    mask0 = (delta == 0);
    Phi(mask0) = K / N;             % Δ=0 时 => 求和得到 K

    mask = ~mask0;
    if any(mask(:))
        deltaNZ = delta(mask);
        r   = exp(1j*2*pi*deltaNZ/N);
        num = 1 - r.^K;
        den = 1 - r;
        Phi(mask) = (1/N) * (num ./ den);
    end
end

function [RDM_dB, rangeAxis, velAxis] = compute_RDM(Y, S, Ts, fc, B)
% 按 (21) 式构造 RDM:
% χ = F_N^H (Y ⊙ S^*) F_M

    c0 = 3e8;
    [N,M] = size(Y);

    % 匹配滤波 Y ⊙ S^*
    mf = Y .* conj(S);

    % Range 方向 IFFT
    chi = ifft(mf, [], 1);

    % Doppler 方向 FFT 并居中
    chi = fft(chi, [], 2);
    chi = fftshift(chi, 2);

    % 归一化到最大值为 0 dB
    mag = abs(chi);
    mag = mag / max(mag(:));
    RDM_dB = 20*log10(mag + eps);

    % 轴刻度
    rangeRes  = c0 / (2*B);
    rangeAxis = (0:N-1) * rangeRes;

    k   = (-M/2 : M/2-1);
    fd  = k / (M*Ts);
    velAxis = fd * c0 / (2*fc);
end
