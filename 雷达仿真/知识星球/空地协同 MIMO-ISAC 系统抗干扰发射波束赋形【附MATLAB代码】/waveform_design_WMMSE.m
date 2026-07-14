function [ R, r, W ] = waveform_design_WMMSE( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power )
% WMMSE 波束赋形 — 2025-2026 ISAC 标准迭代基线 [Shi et al. 2011 + ISAC扩展]
%
% 与 SDR 的本质区别：
%   - 无需求解半正定规划 (SDP)，每步迭代有闭式解
%   - 直接优化波束赋形向量 v_k，而非协方差矩阵
%   - 最终 R = V*V^H + R_radar，（V*V^H 天然秩 ≤ K，无需 rank relaxation）
%
% 算法：BCD 三步迭代
%   Step 1: MMSE 接收滤波器 u_k  (闭式)
%   Step 2: MSE 权重 w_k           (闭式)
%   Step 3: 发射波束赋形 v_k       (正则化迫零 + 功率 bisection)

max_iter  = 80;
tol       = 1e-4;
radar_frac = 0.65;             % 雷达功率占比（65% 雷达 / 35% 通信）

%% Step 0: 纯雷达协方差（固定不变）
R_radar = waveform_design_radar_only_covmat( ...
    d_theta, M, P, a, a_theta_bar, theta, power * radar_frac);
P_comm = power * (1 - radar_frac);

%% Step 0: 初始化通信波束赋形（正则化迫零）
H_V_init = H * H' + noise_variance * eye(K);
V = H' / H_V_init;             % 正则化 ZF 初值: M×K
V = V / sqrt(trace(V*V')) * sqrt(P_comm);

%% ---- WMMSE 迭代 ----
H_full = H;                    % K×M
prev_V = zeros(size(V));

for t = 1:max_iter
    HV = H_full * V;           % K×K: HV(k,j) = h_k * v_j

    u_vec = zeros(1, K);
    w_vec = zeros(1, K);

    % ---- Step 1 & 2: 更新接收滤波器和权重 ----
    for k = 1:K
        h_k = H_full(k, :);    % 1×M
        sig = HV(k, k);        % 有用信号 h_k * v_k

        % 总干扰: 多用户 + 雷达 + 噪声
        I_mui = sum(abs(HV(k, :)).^2) - abs(sig)^2;   % 多用户干扰
        I_rad = real(h_k * R_radar * h_k');             % 雷达干扰
        I_tot = I_mui + I_rad + noise_variance;

        % MMSE 接收滤波器 (标量)
        u_vec(k) = conj(sig) / max(abs(sig)^2 + I_tot, eps);

        % MSE
        mse_k = 1 - abs(sig)^2 / max(abs(sig)^2 + I_tot, eps);

        % WMMSE 权重 (确保 SINR 达标: w_k 大 → 优先级高)
        target_mse = 1 / (1 + SINR_threshold);
        w_vec(k) = max(1.0 / max(mse_k, 1e-8), target_mse);
    end

    % ---- Step 3: 更新发射波束赋形 ----
    % 构建: A = Σ w_j |u_j|² h_j^H h_j + μI,  B = [w_1 u_1^* h_1^H, ...]
    A = zeros(M, M);
    B = zeros(M, K);
    for k = 1:K
        h_k = H_full(k, :);
        coef = w_vec(k) * abs(u_vec(k))^2;
        A = A + coef * (h_k' * h_k);
        B(:, k) = w_vec(k) * conj(u_vec(k)) * h_k';
    end

    % Bisection 求解 Lagrange 乘子 μ 满足功率约束 ||V||²_F = P_comm
    mu_low  = 0;
    mu_high = 1e4;
    for bb = 1:25
        mu = (mu_low + mu_high) / 2;
        V_new = (A + mu * eye(M)) \ B;
        pwr = real(trace(V_new' * V_new));
        if pwr > P_comm
            mu_low = mu;
        else
            mu_high = mu;
        end
    end
    V = (A + mu_high * eye(M)) \ B;

    % 收敛检查
    change = norm(V - prev_V, 'fro') / max(norm(prev_V, 'fro'), eps);
    prev_V = V;
    if change < tol
        break;
    end
end

%% ---- 输出 ----
R = V * V' + R_radar;
for k = 1:K
    r(:, :, k) = V(:, k) * V(:, k)';   % 秩1通信分量
end

% 波束赋形矩阵: [通信 | 雷达]
[U_rd, S_rd] = eig(R_radar);
V_radar = U_rd * sqrt(max(S_rd, 0));
W = [V, V_radar];

fprintf('  WMMSE: converged in %d iterations\n', t);
end
