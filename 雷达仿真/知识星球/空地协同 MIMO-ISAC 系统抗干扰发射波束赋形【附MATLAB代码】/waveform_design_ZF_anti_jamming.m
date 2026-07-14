function [ R, p, W ] = waveform_design_ZF_anti_jamming( d_theta, H, M, K, P, a, a_theta_bar, theta, SINR_threshold, noise_variance, power, theta_jammer, null_depth_dB )
% 抗干扰 ZF 联合波束赋形 — 含零陷约束
%
% 在原始 ZF 逼零约束基础上，新增干扰方向的零陷

L = length(theta);
null_ceiling = 10^(-null_depth_dB / 10) * power / M;
a_jammer = ULA_steering_vector(M, theta_jammer);

cvx_solver sedumi
cvx_begin quiet

variables bt p(K)
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
    %% ZF 约束: 消除多用户干扰
    H * R * H' == diag(p);

    %% 等功率
    diag(R) == ones(M, 1) * power / M;

    %% SINR 达标
    for k = 1:K
        p(k) >= SINR_threshold * noise_variance;
    end

    %% ★ 抗干扰零陷约束 ★
    real(a_jammer' * R * a_jammer) <= null_ceiling;

cvx_end

%% 构建波束赋形矩阵
[Lr, p_flag] = chol(R);
if size(Lr, 1) ~= M
    R = nearestSPD(R);
    [Lr, ~] = chol(R);
end
[~, Qh] = qr(H * Lr', 0);
W = [Lr * Qh', zeros(M, M)];

end
