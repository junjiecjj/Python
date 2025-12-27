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
    P_tilde(idx) = 1.8;
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

%% ---- 1.2 RadCom (Separated, Eq.19) ----------------------
fprintf('Solving RadCom (separated, Eq.19) via CVX...\n');
cvx_begin sdp quiet
    variable W1(NC,NC) hermitian semidefinite
    variable W2(NC,NC) hermitian semidefinite
    variable W3(NC,NC) hermitian semidefinite
    variable W4(NC,NC) hermitian semidefinite
    variable sig_sep

    expression sumW(NC,NC)
    sumW = W1 + W2 + W3 + W4;

    % comm-array beampattern: a2^H sumW a2 = tr(sumW * S2{m})
    expression beamp_comm_sep(M)
    for m = 1:M
        beamp_comm_sep(m) = real( trace( sumW * S2{m} ) );
    end

    % Objective: match comm beampattern to scaled radar-only beampattern
    %   min Σθ | P_comm(θ) - σ * P_radar_only(θ) |^2   (Eq.19)
    minimize( sum_square( beamp_comm_sep - sig_sep * beamp_radar_sep_ref ) )

    subject to
        % SINR constraints for each user (trace form, linear in Wk)
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
            num >= gamma * den;  % SINR_i >= gamma
        end

        % total comm transmit power
        trace(sumW) <= PC;
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

% 1) 线性功率，取绝对值保证非负
P_ideal_sep  = abs(alpha_sep * P_tilde);       % Ideal (α_sep * P~)
P_radar_sep  = abs(beamp_radar_sep).';        % Radar-only (full array)
P_radcom_sep = abs(beamp_radcom_sep).';       % RadCom

% 2) 统一归一化因子：三条曲线共用
Pmax_sep = max( [P_ideal_sep; P_radar_sep; P_radcom_sep] );

P_ideal_sep_n  = P_ideal_sep  / Pmax_sep;
P_radar_sep_n  = P_radar_sep  / Pmax_sep;
P_radcom_sep_n = P_radcom_sep / Pmax_sep;

% 3) 为了和论文视觉高度差不多，可以整体再放大一个常数（可调）
scale_sep = 3;   % 你可以微调 2.0~2.5
P_ideal_sep_plot  = scale_sep * P_ideal_sep_n;
P_radar_sep_plot  = scale_sep * P_radar_sep_n;
P_radcom_sep_plot = scale_sep * P_radcom_sep_n;

%% --------------------- 画 Fig.3(a) 的 Separated 子图 ------------------------
figure;
plot(theta_deg, P_ideal_sep_plot,  'k:','LineWidth',1.5); hold on;  % Ideal（黑点虚线）
plot(theta_deg, P_radar_sep_plot,  'b--','LineWidth',1.8);          % Radar-Only（蓝虚线）
plot(theta_deg, P_radcom_sep_plot, 'r','LineWidth',1.8);            % RadCom（红实线）

grid on;
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
title('Separated Deployment (Multi-beam)');
legend('Ideal','Radar-Only','RadCom','Location','Best');
xlim([-90 90]);
ylim([0 5]);          % 跟论文 Fig.3(a) 一样给个 0~5 的刻度
