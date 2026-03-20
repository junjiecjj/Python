%% OFDM-ISAC Beyond CP Limit - SIC_ESPRIT (iterative)
clear; clc; close all; rng(1)

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

% 目标参数
R_targets = [80 120 200];       % [m]
v_targets = [-30 30 100];       % [m/s]
Q = numel(R_targets);           % 目标数

% 仿真 SNR
SNR_dB = 0;

% SIC-ESPRIT 迭代次数（可自行修改）
maxIter = 5;

% ====== 基本性能参数 & CP 限制下最大可感知距离/速度 ======
rangeRes      = c0 / (2*B);              % 距离分辨率
rangeMax_RDM  = rangeRes*(N-1);          % 由 FFT 轴决定的最大距离

Ts_std  = T + Tcp_std;                   % 标准 CP 下符号时长

% CP 限制下的“无 ISI 最大距离”：2R/c <= Tcp
Rmax_CP_std  = c0*Tcp_std  / 2;

% 多普勒分辨率 & 无模糊最大速度（PRF = 1/Ts）
velRes_std = c0 / (2*fc*M*Ts_std);
velMax_std = c0 / (4*fc*Ts_std);

fprintf('==== System parameters (ESPRIT) ====\n');
fprintf('fc = %.2f GHz,  B = %.2f MHz\n', fc/1e9, B/1e6);
fprintf('N = %d subcarriers,  M = %d OFDM symbols\n', N, M);
fprintf('Range resolution      ΔR = %.2f m\n', rangeRes);
fprintf('RDM axis max range  R_max_RDM ≈ %.2f m\n\n', rangeMax_RDM);

fprintf('Standard CP:\n');
fprintf('  Tcp_std = %.3f us, Ts = %.3f us\n', Tcp_std*1e6, Ts_std*1e6);
fprintf('  ISI-free max range (CP) R_max_CP_std  ≈ %.2f m\n', Rmax_CP_std);
fprintf('  Velocity resolution      Δv_std       ≈ %.2f m/s\n', velRes_std);
fprintf('  Unambiguous |v|_max_std  ≈ %.2f m/s\n\n', velMax_std);

fprintf('==== True target list ====\n');
for q = 1:Q
    tau_q = 2*R_targets(q)/c0;        % 往返时延
    fd_q  = 2*v_targets(q)*fc/c0;     % 多普勒频移
    fprintf('Target %d: R = %.1f m, v = %.1f m/s,  tau = %.3f us,  f_d = %.2f kHz\n', ...
        q, R_targets(q), v_targets(q), tau_q*1e6, fd_q/1e3);
end
fprintf('\n');

fprintf('Will run iterative SIC-ESPRIT with maxIter = %d\n\n', maxIter);

% 再次设置随机种子确保可复现
rng(1);

%% ================== 生成 1024-QAM 数据矩阵 S (N×M) ==================
Mqam = 32;                                     % 32×32 = 1024-QAM
levels = -(Mqam-1):2:(Mqam-1);                 % -31, -29, ..., 31
[Ix, Qx] = meshgrid(levels, levels);
const = Ix(:) + 1j*Qx(:);                      % 1024 星座点
const = const / sqrt(mean(abs(const).^2));     % 归一化到单位平均功率

S = const(randi(numel(const), N, M));          % N×M 独立 1024-QAM 符号

%% ================== 场景：标准 CP，含 ISI/ICI ==================
simStd  = simulate_OFDM_echo(N,M,df,fc,Tcp_std,R_targets,v_targets,S,SNR_dB);

%% ================== 1) 迭代 SIC-ESPRIT w. standard-CP ==================
% 利用仿真中分解出的 ISI/ICI：Y = Y_free + Y_ISI - Y_ICI + Z
Y_intf = simStd.Y_ISI - simStd.Y_ICI;      % 总干扰
Y_sic  = simStd.Y;                         % 迭代起点（含干扰）

R_hat_sic_all   = zeros(Q, maxIter);
V_hat_sic_all   = zeros(Q, maxIter);
alpha_sic_all   = zeros(Q, maxIter);

for it = 1:maxIter
    % 每次迭代消掉 1/maxIter 份 ISI/ICI
    Y_sic = Y_sic - (1/maxIter) * Y_intf;

    % 使用当前 Y_sic 做 ESPRIT
    [R_tmp, V_tmp, alpha_tmp] = ...
        esprit_delay_doppler(Y_sic, simStd.S, Q, simStd.Ts, df, fc);

    R_hat_sic_all(:,it) = R_tmp;
    V_hat_sic_all(:,it) = V_tmp;
    alpha_sic_all(:,it) = alpha_tmp;

    fprintf('SIC-ESPRIT iteration %d / %d done.\n', it, maxIter);
end

% 取最后一轮迭代结果作为“收敛”后的 SIC-ESPRIT 估计
R_hat_sic = R_hat_sic_all(:, end);
V_hat_sic = V_hat_sic_all(:, end);
alpha_sic = alpha_sic_all(:, end);

%% ================== 2) 普通 ESPRIT w. standard-CP（不做 SIC） ==================
[R_hat_std, V_hat_std, alpha_std] = ...
    esprit_delay_doppler(simStd.Y, simStd.S, Q, simStd.Ts, df, fc);

%% ================== 作图 ==================
range_xlim = [0 1000];
vel_ylim   = [-200 200];

% ---------- 图 (a) Iterative SIC-ESPRIT w. standard-CP ----------
figure;

% 2D: Range-Velocity 散点
subplot(1,2,1);
plot(R_targets, v_targets, 'ko', 'MarkerSize',8, 'LineWidth',1.5); hold on;
plot(R_hat_sic, V_hat_sic, 'rx', 'MarkerSize',8, 'LineWidth',1.5);
grid on;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
legend('True targets', sprintf('SIC-ESPRIT est. (iter=%d)', maxIter), 'Location','best');
title('Iterative SIC-ESPRIT (2D) - standard CP');
xlim(range_xlim);
ylim(vel_ylim);

% 3D: 加上振幅
subplot(1,2,2);
amp_sic = abs(alpha_sic);
amp_sic = amp_sic / max(amp_sic);  % 归一化
stem3(R_hat_sic, V_hat_sic, amp_sic, 'filled'); hold on;
stem3(R_targets, v_targets, ones(size(R_targets)), 'k--');
grid on;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
zlabel('|\alpha| (norm.)');
title(sprintf('Iterative SIC-ESPRIT (3D) - standard CP, iter=%d', maxIter));
xlim(range_xlim);
ylim(vel_ylim);

% ---------- 图 (b) ESPRIT w. standard-CP（无 SIC） ----------
figure;

% 2D
subplot(1,2,1);
plot(R_targets, v_targets, 'ko', 'MarkerSize',8, 'LineWidth',1.5); hold on;
plot(R_hat_std, V_hat_std, 'rx', 'MarkerSize',8, 'LineWidth',1.5);
grid on;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
legend('True targets','ESPRIT est.','Location','best');
title('ESPRIT (2D) - standard CP (no SIC)');
xlim(range_xlim);
ylim(vel_ylim);

% 3D
subplot(1,2,2);
amp_std = abs(alpha_std);
amp_std = amp_std / max(amp_std);
stem3(R_hat_std, V_hat_std, amp_std, 'filled'); hold on;
stem3(R_targets, v_targets, ones(size(R_targets)), 'k--');
grid on;
xlabel('Range (m)');
ylabel('Velocity (m/s)');
zlabel('|\alpha| (norm.)');
title('ESPRIT (3D) - standard CP (no SIC)');
xlim(range_xlim);
ylim(vel_ylim);


%% ================== 仿真函数定义 ==================
function sim = simulate_OFDM_echo(N,M,df,fc,Tcp,R_targets,v_targets,S,SNR_dB)
% 按论文频域模型在 Y 维度构造：
% Y = Y_free + Y_ISI - Y_ICI + Z

    c0 = 3e8;
    B  = N*df;
    T  = 1/df;
    Ts = T + Tcp;

    Q = numel(R_targets);
    tau = 2*R_targets / c0;         % 往返时延 τ_q
    fd  = 2*v_targets*fc / c0;      % 多普勒频移 f_d,q

    % 反射系数：按 1/R^2 归一化（如需修改目标强度分布，在这里改）
    alphas = 1./(R_targets.^2);
    alphas = alphas / norm(alphas);

    % CP 长度（采样点）
    Ncp = round(Tcp * B);

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
        l_q = round(tau_q*B);
        l_list(q) = l_q;

        % Φ_q 矩阵 (N×N)
        Phi_list{q} = computePhi(N, Ncp, l_q);
    end

    % --------- 计算 ISI / ICI ----------
    for q = 1:Q
        l_q = l_list(q);
        if l_q > Ncp  % 超出 CP 才有 ISI/ICI
            tau_q = tau(q);
            fd_q  = fd(q);
            b_q   = b_list{q};
            c_q   = c_list{q};
            Phi_q = Phi_list{q};

            % 过 CP 的多出来部分 τ_q - Tcp
            tau_excess = tau_q - Tcp;
            b_ex = exp(-1j*2*pi*df*tau_excess*n);  % N×1

            % ISI: 使用上一 OFDM 符号的数据 => 右乘 J1
            A_isi       = (b_ex * conj(c_q)) .* S; % N×M
            A_isi_shift = A_isi * J1;              % 与 s_{m-1} 对应
            Y_ISI = Y_ISI + alphas(q) * (Phi_q * A_isi_shift);

            % ICI: 同一符号内频率泄漏
            A_ici = (b_q * conj(c_q)) .* S;
            Y_ICI = Y_ICI + alphas(q) * (Phi_q * A_ici);
        end
    end

    % --------- 合成回波并加噪声 ----------
    Y_clean = Y_free + Y_ISI - Y_ICI;

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
    sim.B      = B;
    sim.Ts     = Ts;
    sim.Ncp    = Ncp;
end

function Phi = computePhi(N, Ncp, lq)
% 计算 Φ_q (N×N)：
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

%% ================== ESPRIT：延迟-多普勒估计 ==================
function [R_hat, V_hat, alpha_sel] = esprit_delay_doppler(Y, S, Q, Ts, df, fc)
% 简化版 2D ESPRIT

    c0 = 3e8;
    [N,M] = size(Y);

    % 匹配滤波后的“通道”矩阵
    Hhat = Y .* conj(S);   % N×M
    hvec = Hhat(:);
    NM = N*M;

    % ===== 1D ESPRIT in range dimension =====
    lambda_tau = esprit_1d_eig(Hhat, Q);
    tau_hat = -angle(lambda_tau) / (2*pi*df);    % tau_q

    % ===== 1D ESPRIT in Doppler dimension =====
    Hhat_d = Hhat.';                             % M×N
    lambda_fd = esprit_1d_eig(Hhat_d, Q);
    fd_hat = angle(lambda_fd) / (2*pi*Ts);       % f_d,q

    % ===== LS 相关度配对 (tau_i, f_d,j) =====
    n = (0:N-1).';
    m = 0:M-1;

    alpha_ij = zeros(Q,Q);
    for i = 1:Q
        b_i = exp(-1j*2*pi*df * tau_hat(i) * n);     % N×1
        for j = 1:Q
            c_j = exp(-1j*2*pi*fd_hat(j)*Ts * m);    % 1×M
            A_ij = (b_i * conj(c_j));                % N×M
            a_vec = A_ij(:);
            alpha_ij(i,j) = (a_vec' * hvec) / NM;    % LS 估计
        end
    end

    used_i = false(Q,1);
    used_j = false(Q,1);
    tau_sel  = zeros(Q,1);
    fd_sel   = zeros(Q,1);
    alpha_sel = zeros(Q,1);

    for k = 1:Q
        scores = abs(alpha_ij);
        scores(used_i,:) = -inf;
        scores(:,used_j) = -inf;

        [~, idx_max] = max(scores(:));
        [ii, jj] = ind2sub([Q,Q], idx_max);

        used_i(ii) = true;
        used_j(jj) = true;

        tau_sel(k)   = tau_hat(ii);
        fd_sel(k)    = fd_hat(jj);
        alpha_sel(k) = alpha_ij(ii,jj);
    end

    % 转成距离和速度
    R_hat = c0 * tau_sel / 2;
    V_hat = fd_sel * c0 / (2*fc);
end

function lambda = esprit_1d_eig(X, Q)
% 1D ESPRIT 核心：给出阵列观测 X (K×L)，返回 Q 个特征值 lambda

    [K,L] = size(X);

    % 样本协方差
    R = zeros(K,K);
    for l = 1:L
        x = X(:,l);
        R = R + x * x';
    end
    R = R / L;

    % 特征分解，取最大 Q 个特征值对应的特征向量
    [U,D] = eig(R);
    [~, idx] = sort(real(diag(D)), 'descend');
    Us = U(:, idx(1:Q));

    % 选择矩阵
    J1 = [eye(K-1), zeros(K-1,1)];
    J2 = [zeros(K-1,1), eye(K-1)];

    % 旋转矩阵 & 特征值
    Psi = pinv(J1*Us) * (J2*Us);
    lambda = eig(Psi);
end

