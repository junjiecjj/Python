% =============================================================
%  Fig.4(b) - Shared Deployment (3 dB Beampattern, Array Gain)
%  MU-MIMO Communications with MIMO Radar:
%  From Co-existence to Joint Transmission (Liu et al., TWC 2018)
%
%  - Radar-only: 3 dB main-beam centered at 0 deg (width 10 deg)
%                via Problem (10) on full N antennas
%  - RadCom:     Shared deployment via Problem (20),
%                matching radar 3 dB beampattern
%  - 绘图使用“阵列增益 G(theta)”并转为 dBi，主瓣平移到 ~5 dBi
%  - 计算 RadCom (Shared) 的 PSLR，用于检查 ≈ 15 dB
% =============================================================
clear; clc; close all;

%% ============= System parameters (same as Fig.3) =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear
N      = 20;                 % total antennas at BS
NR     = 14;                 % radar antennas (for separated, not used here)
NC     = N - NR;             % comm antennas (for separated, not used here)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

rng(1);                      % for reproducibility (same as Fig.3)

%% ============= Angle grid and steering matrices =============
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M         = numel(theta_deg);

% Full N-antenna ULA steering (A_full: N x M)
A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end

% Precompute S matrices for beampatterns: S(m) = a(theta_m) a(theta_m)^H
Sfull = cell(M,1);      % N  x N
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
theta0   = 0;         % main-beam center (deg)
bw_3dB   = 10;        % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2;   % left  3 dB point (-5)
theta2   = theta0 + bw_3dB/2;   % right 3 dB point (+5)

% Indices on the angle grid
[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

% Sidelobe region 𝕌: all angles outside [theta1, theta2]
idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

% Steering vectors for the three key angles (full array)
a_0 = A_full(:,idx0);   % θ0
a_1 = A_full(:,idx1);   % θ1
a_2 = A_full(:,idx2);   % θ2

fprintf('Solving radar-only 3dB (shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t_shared

    % Objective: maximize t  <=> minimize -t
    minimize( -t_shared )

    subject to
        % Per-antenna power on full array: diag(R2) = P0/N
        diag(R2) == (P0/N) * ones(N,1);

        % 3dB constraints:
        % a(θ1)^H R2 a(θ1) = 0.5 * a(θ0)^H R2 a(θ0)
        % a(θ2)^H R2 a(θ2) = 0.5 * a(θ0)^H R2 a(θ0)
        main0 = real( a_0' * R2 * a_0 );
        left  = real( a_1' * R2 * a_1 );
        right = real( a_2' * R2 * a_2 );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % Sidelobe suppression constraints:
        % a(θ0)^H R2 a(θ0) - a(θm)^H R2 a(θm) >= t, ∀θm ∈ 𝕌
        for mm = idx_sidelobe
            am    = A_full(:,mm);
            sidel = real( am' * R2 * am );
            main0 - sidel >= t_shared;
        end
cvx_end

% Radar-only beampattern (shared deployment): P_radar_only(theta) = a^H R2 a
P_radar_only = zeros(M,1);
for m = 1:M
    P_radar_only(m) = real(trace(R2 * Sfull{m}));
end

% =============================================================
%  Part 2: RadCom beampattern (Shared, Problem 20) - 3 dB case
% =============================================================

fprintf('Solving RadCom (shared, Problem 20) via CVX...\n');

cvx_begin sdp quiet
    % 这里 K=4，直接定义 4 个 T_i，Hermitian
    variable T1(N,N) hermitian semidefinite
    variable T2(N,N) hermitian semidefinite
    variable T3(N,N) hermitian semidefinite
    variable T4(N,N) hermitian semidefinite

    % sum_i T_i
    expression Tsum(N,N)
    Tsum = T1 + T2 + T3 + T4;

    % ===== (20a) 目标函数：|| Tsum - R2 ||_F^2 =====
    minimize( sum_square_abs( vec(Tsum - R2) ) )

    subject to
        % ===== (20c) per-antenna power 约束 =====
        % diag( sum_i T_i ) = P0/N * 1_N
        diag(Tsum) == (P0/N) * ones(N,1);

        % ===== (20d) PSD 约束：T_i ⪰ 0 =====
        T1 >= 0;
        T2 >= 0;
        T3 >= 0;
        T4 >= 0;

        % ===== (20b) SINR 约束：γ_i >= gamma，γ_i 用 (7) =====
        % γ_i = tr(HiHiH{i} * T_i) / ( sum_{k≠i} tr(HiHiH{i} * T_k) + N0 )

        % 用户 1：
        trace(HiHiH{1} * T1) >= gamma * ( ...
            trace(HiHiH{1} * T2) + ...
            trace(HiHiH{1} * T3) + ...
            trace(HiHiH{1} * T4) + N0 );

        % 用户 2：
        trace(HiHiH{2} * T2) >= gamma * ( ...
            trace(HiHiH{2} * T1) + ...
            trace(HiHiH{2} * T3) + ...
            trace(HiHiH{2} * T4) + N0 );

        % 用户 3：
        trace(HiHiH{3} * T3) >= gamma * ( ...
            trace(HiHiH{3} * T1) + ...
            trace(HiHiH{3} * T2) + ...
            trace(HiHiH{3} * T4) + N0 );

        % 用户 4：
        trace(HiHiH{4} * T4) >= gamma * ( ...
            trace(HiHiH{4} * T1) + ...
            trace(HiHiH{4} * T2) + ...
            trace(HiHiH{4} * T3) + N0 );
cvx_end

C_radcom = Tsum;   % RadCom 的联合协方差矩阵

% RadCom 情况下的方向图：P_radcom(theta) = a^H C_radcom a
P_radcom = zeros(M,1);
for m = 1:M
    P_radcom(m) = real(trace(C_radcom * Sfull{m}));
end

% =============================================================
%  Part 3: 使用“阵列增益”并转为 dBi，主瓣调到 ~5 dBi；计算 PSLR
% =============================================================

% --- 3.1 阵列增益（Array Gain） ---
% 定义：G(theta) = P(theta) / (P0/N)   (同一基准，因为都是全阵列发 P0)

G_radar_lin  = P_radar_only / (P0/N);   % M×1
G_radcom_lin = P_radcom     / (P0/N);   % M×1

% 转为 dBi
G_radar_dBi_raw  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi_raw = 10*log10(G_radcom_lin + 1e-12);

% --- 3.2 为了和论文 Fig.4(b) 纵轴风格一致：把 Radar-only 主瓣调到 ~5 dBi ---
[~, idx0_plot] = min(abs(theta_deg - 0));    % 主瓣中心索引
main_radar_raw = G_radar_dBi_raw(idx0_plot);
offset         = 5 - main_radar_raw;         % 让雷达主瓣 ≈ 5 dBi

G_radar_dBi  = G_radar_dBi_raw  + offset;
G_radcom_dBi = G_radcom_dBi_raw + offset;

% --- 3.3 计算 PSLR (RadCom Shared)，对齐论文定义 ---
% 旁瓣区域 = 非 [-5°, 5°]
sidelobe_idx = find( theta_deg < -5 | theta_deg > 5 );

main_peak_radcom = G_radcom_dBi(idx0_plot);
max_sidelobe_radcom = max(G_radcom_dBi(sidelobe_idx));

PSLR_radcom_shared = main_peak_radcom - max_sidelobe_radcom;
fprintf('RadCom (Shared) PSLR ≈ %.2f dB (目标 ~15 dB)\n', PSLR_radcom_shared);

% （可选）雷达-only PSLR
main_peak_radar = G_radar_dBi(idx0_plot);
max_sidelobe_radar = max(G_radar_dBi(sidelobe_idx));
PSLR_radar_shared = main_peak_radar - max_sidelobe_radar;
fprintf('Radar-only (Shared) PSLR ≈ %.2f dB\n', PSLR_radar_shared);

% --- 3.4 绘图（Fig.4(b) 风格） ---
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);

grid on; xlabel('Angle (Degree)');
ylabel('Array Gain (dBi)');
title('Shared Deployment - 3 dB Beampattern (Fig.4(b), Array Gain)');
legend('Radar-Only (Shared)','RadCom (Shared)','Location','Best');

ylim([-20 10]);    % 视觉上接近论文 Fig.4(b)
xlim([-90 90]);
