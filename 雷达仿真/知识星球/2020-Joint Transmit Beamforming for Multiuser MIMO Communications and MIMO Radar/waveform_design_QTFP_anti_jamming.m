function [ R, r, W ] = waveform_design_QTFP_anti_jamming( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power, theta_jammer, null_depth_dB )
% QT-FP 抗干扰波束赋形 — 含零陷约束 [Shen et al. JSAC 2024 + 本文扩展]
%
% 与 SDR 抗干扰的本质区别:
%   - SDR-AJ: 零陷作为硬约束 a'Ra ≤ ε 加入 SDP
%   - QTFP-AJ: 零陷作为软惩罚项加入目标函数，在迭代中自然收敛
%   - 零陷惩罚项: + w_null · (a_jammer^H R a_jammer)
%     等价于在干扰方向"加大等效噪声"，迫使波束自动规避

max_iter   = 30;
tol        = 1e-3;
SINR_linear = 10^(SINR_threshold/10);
lambda_rate = 0.25;          % 速率权重（抗干扰版稍低）
lambda_null = 5.0;            % ★ 零陷惩罚权重
null_target_power = 10^(-null_depth_dB / 10) * power / M;
a_jammer = ULA_steering_vector(M, theta_jammer);

%% Step 0: 初始化
R = waveform_design_radar_only_covmat( ...
    d_theta, M, P, a, a_theta_bar, theta, power);
for k = 1:K
    r(:, :, k) = zeros(M, M);
    r(k, k, k) = power / (M * K);
end

%% QT-FP 迭代
prev_obj = Inf;

for t = 1:max_iter
    %% Step 1: 更新辅助变量
    y_k = zeros(K, 1);
    for k = 1:K
        sig = real(H(k, :) * r(:, :, k) * H(k, :)');
        R_others = R - r(:, :, k);
        intf = real(H(k, :) * R_others * H(k, :)') + noise_variance;
        y_k(k) = sqrt(max(sig, 1e-12)) / max(intf, 1e-12);
    end

    %% Step 2: 凸子问题 (CVX) ★ 含零陷惩罚 ★
    L = length(theta);
    cvx_solver sedumi
    cvx_begin quiet

    variable bt
    variable r_new(M, M, K) hermitian semidefinite
    variable R_new(M, M) hermitian semidefinite
    expressions u1(L) u2((P^2 - P) / 2)

    for ii = 1:L
        u1(ii) = (bt * d_theta(ii) - a(:, ii)' * R_new * a(:, ii));
    end
    for ii = 1:(P - 1)
        for jj = (ii + 1):P
            u2(ii + jj - 2) = a_theta_bar(:, jj)' * R_new * a_theta_bar(:, ii);
        end
    end

    %% QT-FP 速率项
    qt_term = 0;
    for k = 1:K
        hk = H(k, :);
        sig_term = real(hk * r_new(:, :, k) * hk');
        intf_term = real(hk * (R_new - r_new(:, :, k)) * hk') + noise_variance;
        qt_term = qt_term + (2 * y_k(k) * sqrt(sig_term) ...
                             - y_k(k)^2 * intf_term);
    end

    %% ★★★ 零陷惩罚项（QT-FP 独有机制）★★★
    % 在目标函数中增加: λ_null · (a_jammer^H R a_jammer)
    % 与 SDR 的硬约束等价，但收敛更平滑
    null_penalty_term = lambda_null * real(a_jammer' * R_new * a_jammer) / (power / M);

    minimize( square_pos(norm(u1, 2)) / L ...
            + square_pos(norm(u2, 2)) * (2 / (P^2 - P)) ...
            - lambda_rate * qt_term ...
            + null_penalty_term )

    subject to
        R_new - sum(r_new, 3) == hermitian_semidefinite(M);
        diag(R_new) == ones(M, 1) * power / M;
        for k = 1:K
            real((1 + 1/SINR_linear) * H(k, :) * r_new(:, :, k) * H(k, :)') ...
                >= real(H(k, :) * R_new * H(k, :)' + noise_variance);
        end

    cvx_end

    %% Step 3: 收敛
    obj = cvx_optval;
    if ~isnan(obj)
        change = abs(prev_obj - obj) / max(abs(prev_obj), eps);
        prev_obj = obj;
        R = R_new;
        r = r_new;
        if change < tol
            break;
        end
    else
        break;
    end
end

%% 输出
for k = 1:K
    wk = (H(k, :) * r(:, :, k) * H(k, :)')^(-1/2) * r(:, :, k) * H(k, :)';
    r(:, :, k) = wk * wk';
    Wc(:, k) = wk;
end
Wr_WrH = R - sum(r, 3);
[Wr, p_flag] = chol(Wr_WrH);
if size(Wr, 1) ~= M
    Wr_WrH = nearestSPD(Wr_WrH);
    [Wr, ~] = chol(Wr_WrH);
end
W = [Wc, Wr'];

fprintf('  QT-FP-AntiJam: converged in %d iterations, lambda_null=%.1f\n', t, lambda_null);
end
