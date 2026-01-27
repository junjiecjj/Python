% =============================================================
%  Fig.4(b) - Shared Deployment (3 dB Beampattern, Monte Carlo)
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear total power
N      = 20;                 % total antennas (shared)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);                      % base seed

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Full N-antenna ULA steering
A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

% Precompute S matrices for beampatterns: S(m) = a(theta_m) a(theta_m)^H
Sfull = cell(M,1);      % N x N
for m = 1:M
    afm = A_full(:,m);
    Sfull{m} = afm * afm';
end

% =============================================================
%  Part 1: Radar-Only (Shared) via Problem (10) - 3 dB beam
%          —— 与信道无关，只求一次
% =============================================================

% ---- 1.1 Define main-beam and sidelobe regions ----
theta0   = 0;     % main-beam center (deg)
bw_3dB   = 10;    % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2;   % -5°
theta2   = theta0 + bw_3dB/2;   % +5°

% Indices on the angle grid (θ0, θ1, θ2)
[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

% Sidelobe region 𝕌 = 所有在 [θ1, θ2] 之外的角度
idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

fprintf('Solving radar-only 3dB (shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t_sh

    % Objective: maximize t_sh  <=> minimize -t_sh
    minimize( -t_sh )

    subject to
        % (i) Per-antenna power on full array: diag(R2) = P0/N
        diag(R2) == (P0/N) * ones(N,1);

        % (ii) 3dB constraints: P(theta1) = P(theta2) = 0.5 * P(theta0)
        main0 = real( trace(R2 * Sfull{idx0}) );
        left  = real( trace(R2 * Sfull{idx1}) );
        right = real( trace(R2 * Sfull{idx2}) );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % (iii) Sidelobe suppression: P(theta0) - P(thetam) >= t_sh
        for mm = idx_sidelobe
            sidel = real( trace(R2 * Sfull{mm}) );
            main0 - sidel >= t_sh;
        end
cvx_end

% 雷达-only 3dB 方向图：P_radar_3dB(theta) = a^H R2 a
P_radar_3dB = zeros(M,1);
for m = 1:M
    P_radar_3dB(m) = real( trace(R2 * Sfull{m}) );
end

% =============================================================
%  Part 2: RadCom (Shared, Problem 20) - Monte Carlo over channels
%          Wi = ti ti^H (SDR)，对多个信道求平均方向图
% =============================================================

%MC = 10;   % Monte Carlo 次数，可先设 5 或 10 试跑，再加大
MC=1;

P_radcom_3dB_MC = zeros(M, MC);   % 每次仿真的 RadCom 方向图

for mc = 1:MC
    fprintf('Monte Carlo run %d / %d ...\n', mc, MC);

    % ----- 随机生成信道 H (N x K) -----
    H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

    % Precompute outer-products for SINR constraints (shared deployment)
    HiHiH = cell(K,1);   % N  x N
    for i = 1:K
        hi = H(:,i);
        HiHiH{i} = hi * hi';
    end

    % ----- CVX 解 RadCom (shared, Problem 20, 3dB) -----
    cvx_begin sdp quiet
        % Wi: 用户 i 的下行协方差矩阵（N×N），对应 ti ti^H
        variable W1(N,N) hermitian semidefinite
        variable W2(N,N) hermitian semidefinite
        variable W3(N,N) hermitian semidefinite
        variable W4(N,N) hermitian semidefinite
        % σ：缩放系数
        variable sig_sh

        % Σ Wi
        expression Wsum(N,N)
        Wsum = W1 + W2 + W3 + W4;

        % ===== 目标函数：beampattern matching =====
        % e(m) = trace(Wsum * Sfull{m}) - σ * trace(R2 * Sfull{m})
        expression diff_vec(M,1)
        for m_idx = 1:M
            diff_vec(m_idx) = real( trace(Wsum * Sfull{m_idx}) ...
                               - sig_sh * trace(R2 * Sfull{m_idx}) );
        end
        minimize( sum_square(diff_vec) )

        subject to
            % 每天线功率约束：diag(Wsum) = P0/N * 1_N
            diag(Wsum) == (P0/N) * ones(N,1);

            % PSD 约束：Wi ⪰ 0
            W1 >= 0;
            W2 >= 0;
            W3 >= 0;
            W4 >= 0;

            % SINR 约束：β_i >= γ
            % β_i = tr(HiHiH{i} * Wi) / (sum_{k≠i} tr(HiHiH{i} * Wk) + N0)

            % 用户 1：
            trace(HiHiH{1} * W1) >= gamma * ( ...
                trace(HiHiH{1} * W2) + ...
                trace(HiHiH{1} * W3) + ...
                trace(HiHiH{1} * W4) + N0 );

            % 用户 2：
            trace(HiHiH{2} * W2) >= gamma * ( ...
                trace(HiHiH{2} * W1) + ...
                trace(HiHiH{2} * W3) + ...
                trace(HiHiH{2} * W4) + N0 );

            % 用户 3：
            trace(HiHiH{3} * W3) >= gamma * ( ...
                trace(HiHiH{3} * W1) + ...
                trace(HiHiH{3} * W2) + ...
                trace(HiHiH{3} * W4) + N0 );

            % 用户 4：
            trace(HiHiH{4} * W4) >= gamma * ( ...
                trace(HiHiH{4} * W1) + ...
                trace(HiHiH{4} * W2) + ...
                trace(HiHiH{4} * W3) + N0 );

            % σ >= 0
            sig_sh >= 0;
    cvx_end

    % ----- 本次 Monte Carlo 的 RadCom 方向图 -----
    P_radcom_now = zeros(M,1);
    for m_idx = 1:M
        P_radcom_now(m_idx) = real( trace(Wsum * Sfull{m_idx}) );
    end

    P_radcom_3dB_MC(:, mc) = P_radcom_now;
end

% Monte Carlo 平均（在线性功率域平均）
P_radcom_3dB_avg = mean(P_radcom_3dB_MC, 2);

%% ============= 归一化并绘图 ===================

% Radar-Only（Shared, 3 dB）：用自身最大值归一化(这不对吧）
%norm_radar = max(P_radar_3dB);
norm_radar = 5;
P_radar_n  = P_radar_3dB / (norm_radar + eps);

% RadCom（Shared, 3 dB，MC 平均）：用自身最大值归一化
%norm_radc  = max(P_radcom_3dB_avg);
norm_radc  =5;
P_radc_n   = P_radcom_3dB_avg / (norm_radc + eps);

% 线性刻度
figure;
plot(theta_deg, P_radar_n,'b--','LineWidth',1.5); hold on;
plot(theta_deg, P_radc_n, 'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
legend('Radar-Only (Shared, 3dB)','RadCom (Shared, 3dB, MC-avg)','Location','Best');
title(sprintf('Fig.4(b) Shared Deployment: 3 dB Beampatterns (MC = %d)', MC));

% dB 刻度
P_radar_dB = 10*log10(P_radar_n + eps);
P_radc_dB  = 10*log10(P_radc_n  + eps);

figure;
plot(theta_deg, P_radar_dB,'b--','LineWidth',1.5); hold on;
plot(theta_deg, P_radc_dB, 'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern (dB)');
legend('Radar-Only (Shared, 3dB)','RadCom (Shared, 3dB, MC-avg)','Location','Best');
title(sprintf('Fig.4(b) Shared Deployment: 3 dB Beampatterns (MC = %d, dB)', MC));
