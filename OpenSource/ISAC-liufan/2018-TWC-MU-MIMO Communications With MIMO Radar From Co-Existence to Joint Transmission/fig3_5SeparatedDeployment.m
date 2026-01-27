% =============================================================
%  Fig.3(a) - Separated Deployment (Radar-Only & RadCom)
%  MU-MIMO Communications with MIMO Radar: From Co-existence
%  to Joint Transmission (Liu et al., TSP 2018)
% =============================================================
clear; clc; close all;

%% ============= System parameters =============
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

%% ============= Desired multi-beam beampattern ===============
% 5 beams at [-60, -36, 0, 36, 60] degrees
beam_dirs_deg = [-60 -36 0 36 60];
main_bw_deg   = 2;           % half mainlobe width

P_tilde = zeros(M,1);        % desired pattern on angle grid
for b = 1:numel(beam_dirs_deg)
    idx = abs(theta_deg - beam_dirs_deg(b)) <= main_bw_deg;
    P_tilde(idx) = 2;
end

%% ============= Random user channels (Rayleigh) ==============
% H: N x K, BS-to-users channel, CN(0,1)
H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

% For separated deployment: split channel into radar/comm parts
F = H(1:NR ,:);   % radar -> users
G = H(NR+1:end,:);% comm  -> users

% Precompute outer-products for SINR constraints
GiGiH = cell(K,1);   % NC x NC
FiFiH = cell(K,1);   % NR x NR
HiHiH = cell(K,1);   % N  x N (shared)
for i = 1:K
    gi = G(:,i);
    fi = F(:,i);
    hi = H(:,i);
    GiGiH{i} = gi * gi';
    FiFiH{i} = fi * fi';
    HiHiH{i} = hi * hi';
end

% =============================================================
%  Part 1: Beampatterns (Separated) - Radar-only & RadCom
% =============================================================

%% ---- 1.1 Radar-Only (Eq.12) ----------------------
fprintf('Solving rad\_only (separated) via CVX...\n');
cvx_begin sdp quiet
    variable R1(NR,NR) hermitian semidefinite
    variable alpha_sep
    expression beamp_sep_rad(M)
    for m = 1:M
        beamp_sep_rad(m) = real( trace( R1 * S1{m} ) );
    end
    minimize( sum_square( alpha_sep * P_tilde - beamp_sep_rad ) )
    subject to
        % per-antenna power (radar array)
        diag(R1) == (PR/NR)*ones(NR,1);
        alpha_sep >= 0;
        % zero interference from radar to users
        for i = 1:K
            beamp_user = real( trace( R1 * FiFiH{i} ) );
            beamp_user == 0;
        end
cvx_end

% reference radar-only pattern (a1^H R1 a1) on angle grid
beamp_radar_sep_ref = beamp_sep_rad;    % M x 1

%% ---- 1.2 RadCom (Separated, Eq.19, strictly following paper form) ----
fprintf('Solving RadCom (separated, Eq.19) via CVX (matrix form)...\n');

cvx_begin sdp quiet
    % Wi: 用户 i 的下行协方差矩阵（NC×NC），半正定
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    % sigma：论文中的缩放系数 σ
    variable sig_sep

    % Σ Wi
    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    % ====== 论文式子里的两块矩阵 ======
    % A2^H (ΣWi) A2  —— 通信阵列在所有角度上的协方差
    C_comm  = A2' * sumW * A2;          % M×M
    % A1^H R1 A1  —— 雷达-only 在所有角度上的协方差
    C_radar = A1' * R1 * A1;            % M×M（R1 已由 1.1 求出，此处是常数）

    % ====== diag(...) 并取 2 范数平方（完全对应论文目标） ======
    % e(m) = [A2^H ΣWi A2]_{mm} - σ [A1^H R1 A1]_{mm}
    diff_vec = real( diag( C_comm - sig_sep * C_radar ) );   % M×1 实数
    minimize( sum_square( diff_vec ) )   %  ||diag(...)||_2^2

    subject to
        % ---------- SINR 约束：β_i >= γ ----------
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

%% ---- 1.3 用总协方差矩阵算 Separated 的 Radar-only / RadCom 波束图 ----
% Overall covariance matrices (Eq.14)
C_radar_sep  = blkdiag(R1, zeros(NC,NC));               % Radar-only
C_radcom_sep = blkdiag(R1, full(W1+W2+W3+W4));          % RadCom

beamp_radar_sep  = zeros(1,M);
beamp_radcom_sep = zeros(1,M);
for m = 1:M
    af = A_full(:,m);
    beamp_radar_sep(m)  = real( af' * C_radar_sep  * af );
    beamp_radcom_sep(m) = real( af' * C_radcom_sep * af );
end

% =============================================================
%  Part 2: Normalization & Plotting (Separated, Fig.3(a) style)
% =============================================================

% === 1) 各自单独归一化（每条曲线都用满功率） ===

% Ideal
P_ideal_sep  = abs(alpha_sep * P_tilde);
Pmax_ideal   = max(P_ideal_sep);
P_ideal_sep_n = P_ideal_sep / Pmax_ideal;

% Radar-only
P_radar_sep  = abs(beamp_radar_sep).';
Pmax_radar   = max(P_radar_sep);
P_radar_sep_n = P_radar_sep / Pmax_radar;

% RadCom
P_radcom_sep = abs(beamp_radcom_sep).';
Pmax_radcom  = max(P_radcom_sep);
P_radcom_sep_n = P_radcom_sep / Pmax_radcom;

% === 2) 画图 ===
figure;
plot(theta_deg, P_ideal_sep_n,  'k:','LineWidth',1.5); hold on;
plot(theta_deg, P_radar_sep_n,  'b--','LineWidth',1.8);
plot(theta_deg, P_radcom_sep_n, 'r','LineWidth',1.8);

grid on; xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
title('Separated Deployment (Each Curve Individually Normalized)');
legend('Ideal','Radar-Only','RadCom','Location','Best');
ylim([0 1.1]);  % 因为每条峰值 = 1
