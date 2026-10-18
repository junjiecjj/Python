function [ R, r, W ] = waveform_design_SDR_anti_jamming( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power, theta_jammer, null_depth_dB )
% 抗干扰 SDR 联合波束赋形 — 含零陷约束
%
% 输入新增：
%   theta_jammer  — 干扰机方向角 (rad)
%   null_depth_dB — 期望零陷深度 (dB)，如 25 表示 -25dB
%
% 输出：
%   R  — 发射协方差矩阵 (通信 + 雷达 + 零陷)
%   r  — 各用户通信分量
%   W  — 波束赋形矩阵

L = length(theta);

% 零陷上限：a_jammer' * R * a_jammer <= 10^(-null_depth_dB/10) * Pt/M
null_ceiling = 10^(-null_depth_dB / 10) * power / M;
a_jammer = ULA_steering_vector(M, theta_jammer);

cvx_solver sedumi
cvx_begin quiet

variable bt
variable r(M, M, K) hermitian semidefinite
variable R(M, M) hermitian semidefinite
expressions u1(L) u2((P^2 - P) / 2)

%% ---- 目标1: 方向图保真 ----
for ii = 1:L
    u1(ii) = (bt * d_theta(ii) - a(:, ii)' * R * a(:, ii));
end

%% ---- 目标2: 互相关抑制 ----
for ii = 1:(P - 1)
    for jj = (ii + 1):P
        u2(ii + jj - 2) = a_theta_bar(:, jj)' * R * a_theta_bar(:, ii);
    end
end

%% ---- 目标函数 ----
minimize square_pos(norm(u1, 2)) / L ...
       + square_pos(norm(u2, 2)) * (2 / (P^2 - P))

subject to
    %% 雷达分量正定
    R - sum(r, 3) == hermitian_semidefinite(M);

    %% 等功率约束
    diag(R) == ones(M, 1) * power / M;

    %% 通信 SINR 约束
    for k = 1:K
        real((1 + 1 / SINR_threshold) * H(k, :) * r(:, :, k) * H(k, :)') ...
            >= real(H(k, :) * R * H(k, :)' + noise_variance);
    end

    %% ★★★ 抗干扰零陷约束（新增！）★★★
    real(a_jammer' * R * a_jammer) <= null_ceiling;

cvx_end

%% ---- 构建波束赋形矩阵 ----
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

end
