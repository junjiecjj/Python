% =============================================================
%  Fig.4(b) - Shared Deployment (3 dB Beampattern)
%  MU-MIMO Communications with MIMO Radar: From Co-existence
%  to Joint Transmission (Liu et al., TWC 2018)
%
%  - Radar-Only: 3 dB main-beam centered at 0 deg (width 10 deg)
%                via Problem (10) on N-element shared array
%  - RadCom:     Shared deployment via Problem (20),
%                matching the radar 3 dB beampattern + SINR constraints
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear total power
N      = 20;                 % total antennas (shared)
NR     = 14;                 % radar antennas if separated (unused here)
NC     = N - NR;             % comm antennas if separated (unused here)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);                      % for reproducibility

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

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% Precompute outer-products for SINR constraints (shared deployment)
HiHiH = cell(K,1);   % N  x N
for i = 1:K
    hi = H(:,i);
    HiHiH{i} = hi * hi';
end

% =============================================================
%  Part 1: Radar-Only (Shared) via Problem (10) - 3 dB beam
% =============================================================

% ---- 1.1 Define main-beam and sidelobe regions ----
theta0   = 0;     % main-beam center (deg)
bw_3dB   = 10;    % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2;   % left  3 dB point (-5)
theta2   = theta0 + bw_3dB/2;   % right 3 dB point (+5)

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

        % (ii) 3dB constraints:
        %      P(θ1) = P(θ2) = 0.5 * P(θ0)，P(θ) = a^H R2 a = trace(R2 * Sfull{m})
        main0 = real( trace(R2 * Sfull{idx0}) );
        left  = real( trace(R2 * Sfull{idx1}) );
        right = real( trace(R2 * Sfull{idx2}) );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % (iii) Sidelobe suppression: P(θ0) - P(θm) >= t_sh, ∀θm ∈ 𝕌
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
%  Part 2: RadCom (Shared, Problem 20) - 3 dB beampattern matching
%          —— SDR 形式：Wi = ti ti^H
% =============================================================

fprintf('Solving RadCom (shared, Problem 20, 3dB) via CVX (SDP)...\n');

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

    % ===== (20a, 3 dB 版本) 目标函数：beampattern matching =====
    % e(m) = [beamp_RadCom(theta_m)] - σ * [beamp_Radar(theta_m)]
    %      = trace(Wsum * Sfull{m}) - σ * trace(R2 * Sfull{m})
    expression diff_vec(M,1)
    for m = 1:M
        diff_vec(m) = real( trace(Wsum * Sfull{m}) ...
                         - sig_sh * trace(R2 * Sfull{m}) );
    end
    minimize( sum_square(diff_vec) )

    subject to
        % ===== 总功率 / 每天线功率约束（与 Radar-only 一致） =====
        % 这里我们用 per-antenna：diag(Wsum) = P0/N * 1_N
        diag(Wsum) == (P0/N) * ones(N,1);

        % ===== PSD 约束：Wi ⪰ 0 =====
        W1 >= 0;
        W2 >= 0;
        W3 >= 0;
        W4 >= 0;

        % ===== SINR 约束：β_i >= γ =====
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

% RadCom 情况下的方向图：P_radcom_3dB(theta) = a^H Wsum a
P_radcom_3dB = zeros(M,1);
for m = 1:M
    P_radcom_3dB(m) = real( trace(Wsum * Sfull{m}) );
end

%% ============= 归一化并绘图（与 Fig.3(b) 风格一致） ===================

% Radar-Only（Shared, 3 dB）：用自身最大值归一化(这不对吧）
%norm_radar = max(P_radar_3dB);
norm_radar = 5;
P_radar_n  = P_radar_3dB / (norm_radar + eps);

% RadCom（Shared, 3 dB）：用自身最大值归一化
%norm_radc  = max(P_radcom_3dB);
norm_radc  = 5;
P_radc_n   = P_radcom_3dB / (norm_radc + eps);

% 线性刻度
figure;
plot(theta_deg, P_radar_n,'b--','LineWidth',1.5); hold on;
plot(theta_deg, P_radc_n, 'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
legend('Radar-Only (Shared, 3dB)','RadCom (Shared, 3dB)','Location','Best');
title('Fig.4(b) Shared Deployment: 3 dB Beampatterns (self-normalized)');

% dB 刻度
P_radar_dB = 10*log10(P_radar_n + eps);
P_radc_dB  = 10*log10(P_radc_n  + eps);

figure;
plot(theta_deg, P_radar_dB,'b--','LineWidth',1.5); hold on;
plot(theta_deg, P_radc_dB, 'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern (dB)');
legend('Radar-Only (Shared, 3dB)','RadCom (Shared, 3dB)','Location','Best');
title('Fig.4(b) Shared Deployment: 3 dB Beampatterns (self-normalized, dB)');
