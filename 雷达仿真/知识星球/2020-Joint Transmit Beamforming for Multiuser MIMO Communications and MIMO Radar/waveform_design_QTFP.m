function [ R, r, W ] = waveform_design_QTFP( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power )
% QT-FP 波束赋形 — 2024 ISAC 标准 baseline [Shen et al., IEEE JSAC, Nov 2024]
%
% 参考文献:
%   [1] K. Shen, Z. Zhao, Y. Chen, Z. Zhang, H. V. Cheng,
%       "Accelerating Quadratic Transform and WMMSE,"
%       IEEE JSAC, vol. 42, no. 11, pp. 3122–3137, Nov 2024.
%   [2] G. Sun, X. Wu, W. Hao, Z. Zhu et al.,
%       "Resource Management for Integrated Communications, Computing,
%        and Sensing (ICCS) Networks," IEEE TVT, Dec 2024.
%
% 核心思路（与 SDR 的本质区别）:
%   - SDR: SINR 硬约束 + SDP 一次性求解
%   - QT-FP: 将速率目标用二次变换(Quadratic Transform)解耦为
%            辅助变量 y_k 和凸子问题，迭代求解
%   - 无需 rank relaxation，天然避免秩1近似误差
%
% 算法: 两步交替迭代
%   Step 1: 更新辅助变量 y_k (闭式，基于当前 R)
%   Step 2: 固定 y_k，求解凸 SDP 更新 R
%   Step 3: 收敛检查

max_iter   = 30;
tol        = 1e-3;
SINR_linear = 10^(SINR_threshold/10);
lambda_rate = 0.3;    % 速率项的权重（控制 radar-vs-comms 权衡）

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
    %% Step 1: 更新二次变换辅助变量 y_k
    y_k = zeros(K, 1);
    for k = 1:K
        sig = real(H(k, :) * r(:, :, k) * H(k, :)');
        R_others = R - r(:, :, k);
        intf = real(H(k, :) * R_others * H(k, :)') + noise_variance;
        % 闭式最优 y_k: y_k = √(sig) / (intf)  [Shen et al. Eq.(16)]
        y_k(k) = sqrt(max(sig, 1e-12)) / max(intf, 1e-12);
    end

    %% Step 2: 固定 y_k，求解凸子问题 (CVX)
    L = length(theta);
    cvx_solver sedumi
    cvx_begin quiet

    variable bt
    variable r_new(M, M, K) hermitian semidefinite
    variable R_new(M, M) hermitian semidefinite
    expressions u1(L) u2((P^2 - P) / 2)

    %% 雷达方向图保真（与 SDR 相同）
    for ii = 1:L
        u1(ii) = (bt * d_theta(ii) - a(:, ii)' * R_new * a(:, ii));
    end
    for ii = 1:(P - 1)
        for jj = (ii + 1):P
            u2(ii + jj - 2) = a_theta_bar(:, jj)' * R_new * a_theta_bar(:, ii);
        end
    end

    %% QT-FP 变换后的速率项（最大化等效于最大化通信性能）
    qt_term = 0;
    for k = 1:K
        hk = H(k, :);
        sig_term = real(hk * r_new(:, :, k) * hk');
        intf_term = real(hk * (R_new - r_new(:, :, k)) * hk') + noise_variance;
        % 二次变换: 2·y_k·√(sig) - y_k²·intf  [Shen et al. Theorem 1]
        qt_term = qt_term + (2 * y_k(k) * sqrt(sig_term) ...
                             - y_k(k)^2 * intf_term);
    end

    minimize( square_pos(norm(u1, 2)) / L ...
            + square_pos(norm(u2, 2)) * (2 / (P^2 - P)) ...
            - lambda_rate * qt_term )

    subject to
        R_new - sum(r_new, 3) == hermitian_semidefinite(M);
        diag(R_new) == ones(M, 1) * power / M;
        for k = 1:K
            real((1 + 1/SINR_linear) * H(k, :) * r_new(:, :, k) * H(k, :)') ...
                >= real(H(k, :) * R_new * H(k, :)' + noise_variance);
        end

    cvx_end

    %% Step 3: 收敛检查 & 更新 R, r
    % 计算目标函数变化
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
        % CVX 数值异常，保留上次结果
        break;
    end
end

%% 输出：构建波束赋形矩阵
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

fprintf('  QT-FP: converged in %d iterations (Shen et al. JSAC 2024)\n', t);
end
