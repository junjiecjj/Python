% =============================================================
%  Fig.4(a) - Separated Deployment (3 dB Beampattern)
%  MU-MIMO Communications with MIMO Radar:
%  From Co-existence to Joint Transmission (Liu et al., TWC 2018)
%
%  - Radar-only: 3 dB main-beam centered at 0 deg (width 10 deg)
%                via Problem (13) + zero interference to users
%  - RadCom:     Zero-forcing separated deployment via Problem (19),
%                matching radar 3 dB beampattern
%
%  All other parameters are identical to Fig.3 in the paper.
% =============================================================
clear; clc; close all;

%% ============= System parameters (same as Fig.3) =============
P0_dBm = 20;                 % total BS power [dBm]
P0     = 10^(P0_dBm/10);     % linear
N      = 20;                 % total antennas at BS
NR     = 14;                 % radar antennas (separated)
NC     = N - NR;             % comm antennas (separated)
K      = 4;                  % number of users

N0_dBm = 0;                  % noise power [dBm]
N0     = 10^(N0_dBm/10);     % linear

lambda = 1;                  % normalized wavelength
d      = 0.5*lambda;         % half-wavelength spacing

% SINR target for all users (10 dB)
gamma_dB = 10;
gamma    = 10^(gamma_dB/10);

% Power split for separated deployment: half radar, half comm
PR = P0/2;                   % radar power (separated)
PC = P0/2;                   % comm power (separated)

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

% Separated deployment: first NR are radar, last NC are comm
A1 = A_full(1:NR ,:);   % radar array manifold
A2 = A_full(NR+1:end,:);% comm  array manifold

% Precompute S matrices for beampatterns: S(m) = a(theta_m) a(theta_m)^H
S1    = cell(M,1);      % NR x NR, for radar-only (separated)
S2    = cell(M,1);      % NC x NC, for comm (separated)
Sfull = cell(M,1);      % N  x N, for full array

for m = 1:M
    a1m = A1(:,m);
    a2m = A2(:,m);
    afm = A_full(:,m);
    S1{m}    = a1m * a1m';
    S2{m}    = a2m * a2m';
    Sfull{m} = afm * afm';
end

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% For separated deployment: split channel into radar/comm parts
F = H(1:NR ,:);   % radar -> users
G = H(NR+1:end,:);% comm  -> users

% Precompute outer-products for SINR / interference constraints
GiGiH = cell(K,1);   % NC x NC
FiFiH = cell(K,1);   % NR x NR
HiHiH = cell(K,1);   % N  x N (for shared, not used here but kept)
for i = 1:K
    gi = G(:,i);
    fi = F(:,i);
    hi = H(:,i);
    GiGiH{i} = gi * gi';
    FiFiH{i} = fi * fi';
    HiHiH{i} = hi * hi';
end

% =============================================================
%  Part 1: Radar-Only (Separated) via Problem (13) - 3 dB beam
% =============================================================

% ---- 1.1 Define main-beam and sidelobe regions ----
theta0   = 0;     % main-beam center (deg)
bw_3dB   = 10;    % 3 dB beamwidth (deg)
theta1   = theta0 - bw_3dB/2;   % left  3 dB point (-5)
theta2   = theta0 + bw_3dB/2;   % right 3 dB point (+5)

% Indices on the angle grid
[~, idx0] = min(abs(theta_deg - theta0));
[~, idx1] = min(abs(theta_deg - theta1));
[~, idx2] = min(abs(theta_deg - theta2));

% Sidelobe region 𝕌: all angles outside [theta1, theta2]
idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

% Steering vectors for the three key angles
a1_0 = A1(:,idx0);   % θ0
a1_1 = A1(:,idx1);   % θ1
a1_2 = A1(:,idx2);   % θ2

fprintf('Solving radar-only 3dB (separated, Problem 13) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable t_sep

    % Objective: maximize t  <=> minimize -t
    minimize( -t_sep )

    subject to
        % Per-antenna power on radar array: diag(R1) = PR/NR
        diag(R1) == (PR/NR) * ones(NR,1);

        % 3dB constraints:
        % a1(θ1)^H R1 a1(θ1) = 0.5 * a1(θ0)^H R1 a1(θ0)
        % a1(θ2)^H R1 a1(θ2) = 0.5 * a1(θ0)^H R1 a1(θ0)
        main0 = real( a1_0' * R1 * a1_0 );
        left  = real( a1_1' * R1 * a1_1 );
        right = real( a1_2' * R1 * a1_2 );
        left  == 0.5 * main0;
        right == 0.5 * main0;

        % Sidelobe suppression constraints:
        % a1(θ0)^H R1 a1(θ0) - a1(θm)^H R1 a1(θm) >= t, ∀θm ∈ 𝕌
        for mm = idx_sidelobe
            a1m   = A1(:,mm);
            sidel = real( a1m' * R1 * a1m );
            main0 - sidel >= t_sep;
        end

        % Zero interference from radar to users:
        % tr(f_i^* f_i^T R1) = 0, ∀i
        for i = 1:K
            real( trace( R1 * FiFiH{i} ) ) == 0;
        end
cvx_end

% Radar-only beampattern on full N-antenna aperture (for plotting)
C_radar_sep = blkdiag(R1, zeros(NC,NC));  % overall covariance
beamp_radar_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radar_sep(m) = real( af' * C_radar_sep * af );
end

% =============================================================
%  Part 2: RadCom (Separated, Problem 19) matching radar 3 dB
% =============================================================
fprintf('Solving RadCom (separated, Problem 19) via CVX...\n');

cvx_begin sdp quiet
    % Wi: 用户 i 的下行协方差矩阵（NC×NC）
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    % σ：论文中的缩放系数
    variable sig_sep

    % Σ Wi
    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    % A2^H (ΣWi) A2  —— 通信阵列在所有角度上的协方差
    C_comm  = A2' * sumW * A2;          % M×M
    % A1^H R1 A1  —— 雷达-only (3dB) 在所有角度上的协方差
    C_radar_A1 = A1' * R1 * A1;         % M×M

    % e(m) = [A2^H ΣWi A2]_{mm} - σ [A1^H R1 A1]_{mm}
    diff_vec = real( diag( C_comm - sig_sep * C_radar_A1 ) );   % M×1 实数

    % Objective:  ||diag(...)||_2^2
    minimize( sum_square( diff_vec ) )

    subject to
        % ---------- SINR 约束：β_i >= γ ---------- 
        % β_i = tr(GiGiH{i} * Wi) / (tr(GiGiH{i} * Σ_{k≠i} Wk) + tr(FiFiH{i}*R1) + N0)
        for i = 1:K
            switch i
                case 1
                    num = real( trace( W1 * GiGiH{i} ) );
                    den = real( trace( W2 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 2
                    num = real( trace( W2 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 3
                    num = real( trace( W3 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W2 * GiGiH{i} ) ...
                              + trace( W4 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
                case 4
                    num = real( trace( W4 * GiGiH{i} ) );
                    den = real( trace( W1 * GiGiH{i} ) ...
                              + trace( W2 * GiGiH{i} ) ...
                              + trace( W3 * GiGiH{i} ) ...
                              + trace( R1 * FiFiH{i} ) ) + N0;
            end
            num >= gamma * den;     % β_i >= γ
        end

        % ---------- 通信端总功率约束：Σ tr(Wi) <= PC ----------
        trace(sumW) <= PC;

        % ---------- σ >= 0 ----------
        sig_sep >= 0;
cvx_end

% Overall covariance matrix for RadCom (Separated)
C_radcom_sep = blkdiag(R1, full(W1+W2+W3+W4));

beamp_radcom_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radcom_sep(m) = real( af' * C_radcom_sep * af );
end

% =============================================================
% =============================================================
% =============================================================
% =============================================================
%  Part 3: dBi 计算，按 G(theta) = (P0 / 4pi) * a^H C a 实现
% =============================================================

% 线性方向功率（a^H C a），这里不做归一化
P_radar_sep_lin  = real(beamp_radar_sep).';   % M×1
P_radcom_sep_lin = real(beamp_radcom_sep).';  % M×1

% 等效全向天线相关的缩放因子：P0 / (4*pi)
scale_iso = P0 / (4*pi);

% 按 G(theta) = (P0 / 4π) * a^H C a 计算线性“增益”
G_radar_lin  = scale_iso * P_radar_sep_lin;
%G_radar_lin  = P_radar_sep_lin / scale_iso;
G_radcom_lin = scale_iso * P_radcom_sep_lin;
%G_radcom_lin = P_radcom_sep_lin/ scale_iso;
% 转为 dBi（实现 G_dBi(theta) = 10 log10( G(theta) )）
G_radar_dBi  = 10*log10(G_radar_lin  + 1e-12);
G_radcom_dBi = 10*log10(G_radcom_lin + 1e-12);


% 画图：绝对 dBi（实现了给定公式后的结果）
figure;
plot(theta_deg, G_radar_dBi,  'b--','LineWidth',1.8); hold on;
plot(theta_deg, G_radcom_dBi, 'r','LineWidth',1.8);

grid on;
xlabel('Angle (Degree)');
ylabel('Gain G(\theta) (dBi)');
title('Separated Deployment - 3 dB Beampattern (Fig.4(a), G(\theta) = (P0/4\pi)a^HCa)');

legend('Radar-Only','RadCom','Location','Best');

% 纵轴范围：以主瓣峰值为基准，往下看 40 dB 左右，可按需微调
max_peak = max([G_radar_dBi; G_radcom_dBi]);
ylim([max_peak-40, max_peak+5]);
xlim([-90 90]);
