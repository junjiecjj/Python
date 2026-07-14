function [ R, r, W ] = waveform_design_SDR_anti_jamming_multi( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power, theta_jammers, null_depth_dB_vec )
% SDR 抗干扰 — 复数干扰机版本
%
% 输入:
%   theta_jammers    — 干扰机方向角数组 (rad)，长度 J
%   null_depth_dB_vec — 各干扰机期望零陷深度 (dB)，长度 J 或标量
%
% 原理:
%   每个干扰方向一条线性约束: a(θ_j)^H R a(θ_j) ≤ 10^(-null_dB/10) · P_t/M
%   多条约束共享空间自由度 (M = 10 根天线)

J = length(theta_jammers);
if length(null_depth_dB_vec) == 1
    null_depth_dB_vec = null_depth_dB_vec * ones(1, J);
end

L = length(theta);

% 预计算各干扰方向的零陷上限 + 导向矢量
null_ceilings = zeros(1, J);
a_jammers     = zeros(M, J);
for j = 1:J
    null_ceilings(j) = 10^(-null_depth_dB_vec(j) / 10) * power / M;
    a_jammers(:, j)  = ULA_steering_vector(M, theta_jammers(j));
end

cvx_solver sedumi
cvx_begin quiet

variable bt
variable r(M, M, K) hermitian semidefinite
variable R(M, M) hermitian semidefinite
expressions u1(L) u2((P^2 - P) / 2)

%% 方向图保真
for ii = 1:L
    u1(ii) = (bt * d_theta(ii) - a(:, ii)' * R * a(:, ii));
end

%% 互相关抑制
for ii = 1:(P - 1)
    for jj = (ii + 1):P
        u2(ii + jj - 2) = a_theta_bar(:, jj)' * R * a_theta_bar(:, ii);
    end
end

minimize square_pos(norm(u1, 2)) / L ...
       + square_pos(norm(u2, 2)) * (2 / (P^2 - P))

subject to
    R - sum(r, 3) == hermitian_semidefinite(M);
    diag(R) == ones(M, 1) * power / M;

    for k = 1:K
        real((1 + 1 / SINR_threshold) * H(k, :) * r(:, :, k) * H(k, :)') ...
            >= real(H(k, :) * R * H(k, :)' + noise_variance);
    end

    %% ★ 每个干扰方向一条零陷约束 ★
    for j = 1:J
        real(a_jammers(:, j)' * R * a_jammers(:, j)) <= null_ceilings(j);
    end

cvx_end

%% 构建波束赋形矩阵
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
