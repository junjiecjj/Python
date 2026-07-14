function [ R, r, W ] = waveform_design_WMMSE_anti_jamming( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power, theta_jammer, null_depth_dB )
% WMMSE 抗干扰波束赋形 — 含零陷惩罚项
%
% 与 SDR 抗干扰的本质区别：
%   - 零陷不通过硬约束 (a'Ra ≤ ε) 实现，而通过正则化惩罚项
%   - WMMSE 的 V 更新矩阵中注入 λ * a_jammer * a_jammer^H，
%     等价于在干扰方向"加大正则化强度"，自然压低该方向的发射功率
%
% 参考文献：
%   [1] Shi et al., "An Iteratively Weighted MMSE Approach to Distributed
%       Sum-Utility Maximization," IEEE TSP, 2011.
%   [2] Robust Beamforming for Active STAR-RIS-Aided ISAC, IEEE WCL, 2026.
%   [3] Joint Beamforming for CPS-STAR-RIS Aided MIMO-ISAC, IEEE, 2025.

max_iter   = 80;
tol        = 1e-4;
radar_frac = 0.60;                  % 雷达功率占比（抗干扰版略低，给零陷留余量）
P_comm     = power * (1 - radar_frac);

% 零陷惩罚权重: λ ∝ 10^(null/20)，目标越深惩罚越重
a_jammer = ULA_steering_vector(M, theta_jammer);
null_penalty = 10^(null_depth_dB / 15);   % scale factor

%% Step 0: 纯雷达协方差
R_radar = waveform_design_radar_only_covmat( ...
    d_theta, M, P, a, a_theta_bar, theta, power * radar_frac);

%% Step 0: 初始化（正则化迫零）
H_full = H;
V = H_full' / (H_full * H_full' + noise_variance * eye(K));
V = V / sqrt(trace(V*V')) * sqrt(P_comm);
prev_V = zeros(size(V));

%% ---- WMMSE 迭代 ----
for t = 1:max_iter
    HV = H_full * V;

    u_vec = zeros(1, K);
    w_vec = zeros(1, K);

    % Step 1 & 2: 接收滤波器 + 权重
    for k = 1:K
        h_k = H_full(k, :);
        sig = HV(k, k);
        I_mui = sum(abs(HV(k, :)).^2) - abs(sig)^2;
        I_rad = real(h_k * R_radar * h_k');
        I_tot = I_mui + I_rad + noise_variance;

        u_vec(k) = conj(sig) / max(abs(sig)^2 + I_tot, eps);
        mse_k = 1 - abs(sig)^2 / max(abs(sig)^2 + I_tot, eps);
        target_mse = 1 / (1 + SINR_threshold);
        w_vec(k) = max(1.0 / max(mse_k, 1e-8), target_mse);
    end

    % Step 3: 更新波束赋形 ★ 含零陷惩罚 ★
    A = zeros(M, M);
    B = zeros(M, K);
    for k = 1:K
        h_k = H_full(k, :);
        coef = w_vec(k) * abs(u_vec(k))^2;
        A = A + coef * (h_k' * h_k);
        B(:, k) = w_vec(k) * conj(u_vec(k)) * h_k';
    end

    % ★★★ 零陷惩罚项（WMMSE 独有机制）★★★
    % 在干扰方向增加等效"虚拟噪声"，迫使波束赋形自动规避
    A_null = A + null_penalty * (a_jammer * a_jammer');
    %
    % 解释: a_jammer * a_jammer^H 是干扰方向的秩1投影矩阵。
    % 当 null_penalty 很大时, (A_null)^{-1} 在 a_jammer 方向的分量被强烈衰减，
    % 因此 V = A_null^{-1} * B 的列在干扰方向自然形成零陷。
    % 这与 SDR 的硬约束 a'Ra ≤ ε 等价，但实现方式不同——WMMSE 用软惩罚,
    % SDR 用硬约束。

    % Bisection for power
    mu_low  = 0;
    mu_high = 1e4;
    for bb = 1:25
        mu = (mu_low + mu_high) / 2;
        V_new = (A_null + mu * eye(M)) \ B;
        pwr = real(trace(V_new' * V_new));
        if pwr > P_comm
            mu_low = mu;
        else
            mu_high = mu;
        end
    end
    V = (A_null + mu_high * eye(M)) \ B;

    change = norm(V - prev_V, 'fro') / max(norm(prev_V, 'fro'), eps);
    prev_V = V;
    if change < tol
        break;
    end
end

%% ---- 输出 ----
R = V * V' + R_radar;
for k = 1:K
    r(:, :, k) = V(:, k) * V(:, k)';
end
[U_rd, S_rd] = eig(R_radar);
V_radar = U_rd * sqrt(max(S_rd, 0));
W = [V, V_radar];

fprintf('  WMMSE-AntiJam: converged in %d iterations, null penalty = %.1f\n', t, null_penalty);
end
